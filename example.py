# 假设有一个实现encode方法的模型
import os
from pathlib import Path

import numpy as np

from FileLevelIndexer import FileLevelIndexer, OptimizedIndexer2
from SmartIncremental import QueryRequest
from index.indexer import OptimizedIndexer


class DummyModel:
    def encode(self, texts):
        return np.random.rand(len(texts), 384)


knowledge_path = r"./knowledge_base"

# 初始化索引器
# indexer = FileLevelIndexer(
#     model=DummyModel(),
#     index_path="./my_index.faiss",
#     hash_db_path="./file_hashes.db"
# )

# 扫描文件并构建索引
# indexer.scan_files(knowledge_path).build_index()

# 执行搜索
# results = indexer.search("机器学习", top_k=5)
# for path, content in results:
#     print(f"文件: {Path(path).name}\n内容摘要: {content[:100]}...\n")

# file_level_indexer = OptimizedIndexer2(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     proxy_url=os.getenv("OPENAI_BASE_URL")
# )

indexer = OptimizedIndexer(
    api_key=os.getenv("OPENAI_API_KEY"),
    proxy_url=os.getenv("OPENAI_BASE_URL"))


# ------------ 修改FastAPI端点 ------------
def semantic_search(request: QueryRequest):
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


# def semantic_search2(request: QueryRequest):
#     # 使用API生成查询向量
#     query_embed = file_level_indexer.encode([request.query])
#     distances, indices = file_level_indexer.index.search(query_embed, request.top_k)
#     results = []
#     for idx in indices:
#         if idx in indexer.file_map:
#             file_path, content = indexer.file_map[idx]
#             results.append({
#                 "path": file_path,
#                 "excerpt": content
#             })
#     return {"results": results}


import uvicorn

knowledge_dir = r"./knowledge_base/"

if __name__ == "__main__":
    # file_level_indexer.scan_files(knowledge_dir).build_index()
    indexer.scan_files().build_index()
    q = QueryRequest(query="LSTM", top_k=3)
    print(semantic_search(q))
    # print(semantic_search2(q))
