# 环境依赖：pip install sentence-transformers faiss-cpu fastapi watchdog apscheduler
# ------------ 1. 文件扫描与索引构建（改进分块功能）------------
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class KnowledgeIndexer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_map = {}  # {idx: (file_path, chunk_content)}
        self.chunk_size = 1500
        self.overlap = 300

    def _split_into_chunks(self, text):
        """按指定大小和重叠分割文本"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.overlap)
        return chunks

    def scan_files(self, root_dir='./knowledge_base'):
        """递归扫描并分块存储"""
        self.file_map.clear()
        root = Path(os.path.expanduser(root_dir))
        for md_file in root.rglob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 分割文件内容为多个块
            for chunk in self._split_into_chunks(content):
                self.file_map[len(self.file_map)] = (str(md_file), chunk)
        return self

    def build_index(self):
        """基于文本块构建索引"""
        texts = [v[1] for v in self.file_map.values()]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.reset()
        self.index.add(np.array(embeddings).astype('float32'))
        return self


# ------------ 2. 查询服务（返回分块结果）------------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
indexer = KnowledgeIndexer()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/search")
async def semantic_search(request: QueryRequest):
    query_embed = indexer.model.encode([request.query])
    distances, indices = indexer.index.search(query_embed, request.top_k)

    results = []
    for idx in indices[0]:
        if idx >= 0 and idx in indexer.file_map:  # FAISS可能返回-1
            file_path, chunk_content = indexer.file_map[idx]
            results.append({
                "file": os.path.basename(file_path),
                "path": file_path,
                "content": chunk_content
            })
    return {"results": results}


# ------------ 3. 自动更新（保持原功能）------------
from apscheduler.schedulers.background import BackgroundScheduler


def auto_reindex():
    print("Rebuilding index...")
    indexer.scan_files().build_index()


scheduler = BackgroundScheduler()
scheduler.add_job(auto_reindex, 'cron', day_of_week='mon', hour=2)
scheduler.start()

# ------------ 4. 文件监听（保持原功能）------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            auto_reindex()


observer = Observer()
# file_dir = './knowledge_base'
file_dir = r"E:\AI\my_project\dnd5e\output2"
observer.schedule(FileChangeHandler(), path=file_dir, recursive=True)
observer.start()

if __name__ == "__main__":
    # 初始化索引
    indexer.scan_files(file_dir).build_index()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8123)