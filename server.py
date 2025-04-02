import argparse
import os

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from FileLevelIndexer import OptimizedIndexer2
from config.config import global_config
from index.indexer import OptimizedIndexer

load_dotenv()
app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


def parse_args():
    parser = argparse.ArgumentParser(description='知识库服务配置')
    parser.add_argument('--knowledge_base_name', type=str, help='知识库名称')
    parser.add_argument('--port', type=int, help='服务运行端口号')
    parser.add_argument('--nochunk', action='store_true', help='禁用分块检索')
    return parser.parse_args()


args = parse_args()
knowledge_base_name = args.knowledge_base_name or os.getenv("KNOWLEDGE_BASE_NAME")
global_config.init_knowledge_base(knowledge_base_name)

knowledge_dir = global_config.knowledge_base.knowledge_dir
print(
    f"您正在启动知识库[{global_config.knowledge_base.knowledge_name}]"
    f", 该知识库的描述如下: \n{global_config.knowledge_base.description}")


port = args.port or os.getenv("PORT")
print(f"端口: {port}")

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


def main():
    # print(f"分块状态: {'禁用' if args.nochunk else '启用'}")

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
    observer.schedule(FileChangeHandler(), path=knowledge_dir, recursive=True)
    observer.start()

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
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()
