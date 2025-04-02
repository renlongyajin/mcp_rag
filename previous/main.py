# 环境依赖：pip install sentence-transformers faiss-cpu fastapi watchdog apscheduler

# ------------ 1. 文件扫描与索引构建 ------------
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class KnowledgeIndexer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_map = {}  # {idx: (file_path, content)}

    def scan_files(self, root_dir='./knowledge_base'):
        """递归扫描所有MD文件"""
        root = Path(os.path.expanduser(root_dir))
        for md_file in root.rglob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.file_map[len(self.file_map)] = (str(md_file), content)
        return self

    def build_index(self):
        """生成语义向量并构建FAISS索引"""
        texts = [v[1] for v in self.file_map.values()]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        return self


# ------------ 2. 查询服务 ------------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
indexer = KnowledgeIndexer()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/search")
async def semantic_search(request: QueryRequest):
    # 语义匹配核心逻辑
    query_embed = indexer.model.encode([request.query])
    distances, indices = indexer.index.search(query_embed, request.top_k)

    results = []
    for idx in indices[0]:
        if idx in indexer.file_map:
            file_path, content = indexer.file_map[idx]
            results.append({
                "path": file_path,
                "excerpt": _extract_key_segment(content, request.query)
            })
    return {"results": results}


def _extract_key_segment(text: str, query: str, window_size=300) -> str:
    """提取包含关键信息的段落（简化版）"""
    # 可升级为基于BERT的摘要模型
    return text[:window_size]  # 示例截取前300字符


# ------------ 3. 自动更新 ------------
from apscheduler.schedulers.background import BackgroundScheduler


def auto_reindex():
    print("Rebuilding index...")
    indexer.scan_files().build_index()


scheduler = BackgroundScheduler()
scheduler.add_job(auto_reindex, 'cron', day_of_week='mon', hour=2)  # 每周一凌晨2点
scheduler.start()

# ------------ 4. 文件监听 ------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            auto_reindex()  # 实时更新修改的文件


observer = Observer()
observer.schedule(FileChangeHandler(), path='../knowledge_base', recursive=True)
observer.start()

if __name__ == "__main__":
    # 首次启动时构建索引
    indexer.scan_files().build_index()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8123)