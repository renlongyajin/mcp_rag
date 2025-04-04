### 使用方法

首先找到knowledge_manage/knowledge_base.json，没有的话创建一个。

在其中按照规则添加一个知识库，键值对中键名必须与知识库名称一样。（需指定本地路径）

示例配置如下：这里有着**dnd5e不全书**和**LLM面试库**两个知识库。
```
{
    "dnd5e不全书": {
        "knowledge_dir": "/home/bladedragon/my_project/dnd5e",
        "description": "这是一个dnd5e的不全书（仅包含中文部分）",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "knowledge_name": "dnd5e不全书"
    },
    "LLM面试库":{
        "knowledge_dir": "/home/bladedragon/my_project/mcp_rag/knowledge_base",
        "description": "这是一个LLM面试题库",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "knowledge_name": "LLM面试库"
    }
}
```
找到.env文件，如果没有则创建一个
示例如下：
```
OPENAI_BASE_URL={这里填写你自己的base_url}
OPENAI_API_KEY={这里填写你自己的api-key}

FAISS_INDEX_DIR=./cache
FILE_MANAGER_DIR=./cache

REBUILD_INDEX=true
NO_CHUNK=True#用来标记是否需要分块

KNOWLEDGE_BASE_NAME=dnd5e不全书
PORT=8123
```

然后在.env中指定一个默认知识库的名称，即设置KNOWLEDGE_BASE_NAME

默认运行端口为8123，可以在.env中修改


安装环境
```
pip install - r requirments.txt
```
运行
```
python server.py
```
### 测试
```
curl -X POST http://localhost:8123/search \
-H "Content-Type: application/json" \
-d '{
  "query": "半龙是什么？",
  "top_k": 3,
  "knowledge_name": "dnd5e不全书"
}'
```
若返回结果则正常运行

美化输出，需下载jq
```
curl -s -X POST http://localhost:8123/search \
-H "Content-Type: application/json" \
-d '{"query":"半龙是什么？","top_k":3,"knowledge_name":"dnd5e不全书"}' | jq .
```
