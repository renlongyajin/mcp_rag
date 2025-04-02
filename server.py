import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from FileLevelIndexer import OptimizedIndexer2
from index.indexer import OptimizedIndexer

load_dotenv()
app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


# ------------ 服务初始化 ------------
indexer = OptimizedIndexer(
    api_key=os.getenv("OPENAI_API_KEY"),
    proxy_url=os.getenv("OPENAI_BASE_URL"))


# ------------ 修改FastAPI端点 ------------
@app.post("/search")
async def semantic_search(request: QueryRequest):
    # 使用API生成查询向量
    query_embed = indexer.encode([request.query])
    distances, indices = indexer.index.search(query_embed, request.top_k)
    results = []
    for idx in indices[0]:
        if idx in indexer.file_map:
            file_path, content = indexer.file_map[idx]
            results.append({
                "path": file_path,
                "excerpt": content
            })
    return {"results": results}


class DummyModel:
    def encode(self, texts):
        return np.random.rand(len(texts), 384)


file_level_indexer = OptimizedIndexer2(
    api_key=os.getenv("OPENAI_API_KEY"),
    proxy_url=os.getenv("OPENAI_BASE_URL")
)


@app.post("/search2")
async def semantic_search(request: QueryRequest):
    # 使用API生成查询向量
    query_embed = file_level_indexer.encode([request.query])
    distances, indices = file_level_indexer.index.search(query_embed, request.top_k)
    results = []
    for idx in indices[0]:
        if idx in indexer.file_map:
            file_path, content = indexer.file_map[idx]
            results.append({
                "path": file_path,
                "excerpt": content
            })
    return {"results": results}


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


knowledge_dir = os.getenv("KNOWLEDGE_BASE")
assert knowledge_dir.strip() != ""

observer = Observer()
observer.schedule(FileChangeHandler(), path=knowledge_dir, recursive=True)
observer.start()

import uvicorn

if __name__ == "__main__":
    # 首次启动或强制重建时使用
    if os.getenv("REBUILD_INDEX"):
        print("Force rebuilding index...")
        # indexer.scan_files(knowledge_dir).build_index()
        file_level_indexer.scan_files(knowledge_dir).build_index()
    else:
        # 常规启动自动加载已有索引
        try:
            # indexer.build_index()
            file_level_indexer.build_index()
        except Exception as e:
            print(f"Index loading failed: {e}, rebuilding...")
    print("服务启动前检查：")
    print(f"已注册路由：{app.routes}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8123)
    except Exception as e:
        print(f"启动失败: {e}")
