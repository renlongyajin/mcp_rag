import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from FileLevelIndexer import OptimizedIndexer2
from config.config import global_config
from index.indexer import OptimizedIndexer
from log.log import logger


def main():
    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        top_k: int = 3
        knowledge_name: str = global_config.knowledge_manager.knowledge_base.knowledge_name
        no_chunk: bool = False

        def my_print(self):
            return f"query:{self.query}\ntok_k:{self.top_k}"

    def parse_args():
        parser = argparse.ArgumentParser(description='知识库服务配置')
        parser.add_argument('--knowledge_base_name', type=str, help='知识库名称')
        parser.add_argument('--port', type=int, help='服务运行端口号')
        parser.add_argument('--nochunk', action='store_true', help='禁用分块检索')
        return parser.parse_args()

    args = parse_args()
    knowledge_base_name = args.knowledge_base_name or os.getenv("KNOWLEDGE_BASE_NAME")
    global_config.knowledge_manager.init_knowledge_base(knowledge_base_name)

    knowledge_dir = global_config.knowledge_manager.knowledge_base.knowledge_dir
    print(
        f"您正在启动知识库[{global_config.knowledge_manager.knowledge_base.knowledge_name}]"
        f", 该知识库的描述如下: \n{global_config.knowledge_manager.knowledge_base.description}\n,"
        f"注意：若请求自带数据库名称，将启动其他数据库")

    port = int(args.port or os.getenv("PORT"))
    print(f"端口: {port}")

    # ------------ 服务初始化 ------------
    indexer = OptimizedIndexer(
        api_key=os.getenv("OPENAI_API_KEY"),
        proxy_url=os.getenv("OPENAI_BASE_URL"))

    file_level_indexer = OptimizedIndexer2(
        api_key=os.getenv("OPENAI_API_KEY"),
        proxy_url=os.getenv("OPENAI_BASE_URL")
    )

    # ------------ 修改FastAPI端点 ------------
    def change_knowledge_base(_indexer, new_knowledge_name):
        global_config.knowledge_manager.init_knowledge_base(new_knowledge_name)
        _indexer.change_knowledge_base(new_knowledge_name)
        _indexer.hash_manager.change_knowledge_base(new_knowledge_name)
        global_config.knowledge_manager.init_knowledge_base(new_knowledge_name)

    def judge_change_knowledge(_indexer, request_knowledge_name):
        global_config.knowledge_manager.load_knowledge_base()
        if not (request_knowledge_name in global_config.knowledge_manager.all_knowledge_base_config):
            raise Exception(f"不存在名为{request_knowledge_name}的知识库 ×")
        if request_knowledge_name != global_config.knowledge_manager.knowledge_base.knowledge_name:
            change_knowledge_base(_indexer, request_knowledge_name)
            if not _indexer.hash_manager.is_hash_file_exist(request_knowledge_name):
                info = global_config.knowledge_manager.get_info(request_knowledge_name)
                new_knowledge_dir = info["knowledge_dir"]
                _indexer.scan_files(new_knowledge_dir).build_index()
            else:
                _indexer.build_index()

    @app.post("/search")
    async def semantic_search(request: QueryRequest):
        try:
            request_knowledge_name = request.knowledge_name
            no_chunk = request.no_chunk
            if no_chunk:
                now_index = file_level_indexer
            else:
                now_index = indexer
            judge_change_knowledge(now_index, request_knowledge_name)

            # 使用API生成查询向量
            query_embed = now_index.encode([request.query])
            distances, indices = now_index.index.search(query_embed, request.top_k)
            results = []
            for idx in indices[0]:
                if idx in now_index.file_map:
                    file_path, content = now_index.file_map[idx]
                    results.append({
                        "path": file_path,
                        "excerpt": content
                    })
            logger.log(logging.INFO, msg=f"search success: {request.my_print()}")
            return {"results": results}
        except Exception as e:
            print(f"Search error: {e}")
            return {"error": str(e)}

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
        indexer.scan_files(knowledge_dir).build_index()
        file_level_indexer.scan_files(knowledge_dir).build_index()
    else:
        # 常规启动自动加载已有索引
        try:
            indexer.build_index()
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
