# client.py
import argparse
import json
import sys
from typing import Dict, Any

import requests
from langchain.memory import ConversationBufferMemory

DEFAULT_ENDPOINT = "http://localhost:8123/search"


def send_query(
        query: str,
        top_k: int = 3,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 10
) -> Dict[str, Any]:
    """发送查询到MCP服务并返回结构化结果"""
    payload = {"query": query, "top_k": top_k}

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MCP知识库查询客户端")
    parser.add_argument("--query", type=str, help="直接输入查询语句")
    parser.add_argument("--top_k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--input", type=str, help="从JSON文件读取输入")
    parser.add_argument("--output", type=str, help="结果输出到文件路径")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--verbose", action="store_true", help="显示调试信息")
    parser.add_argument("--timeout", type=int, default=10, help="请求超时时间（秒）")
    parser.add_argument("--pretty", action="store_true", help="美化JSON输出")

    args = parser.parse_args()

    # 输入源处理
    input_data: Dict[str, Any] = {}
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    elif not sys.stdin.isatty():  # 支持管道输入
        input_data = json.load(sys.stdin)
    elif args.query:
        input_data = {"query": args.query, "top_k": args.top_k}
    else:
        parser.print_help()
        sys.exit(1)

    # 参数合并（命令行参数优先）
    final_query = args.query or input_data.get("query", "")
    final_top_k = args.top_k or input_data.get("top_k", 3)

    if args.verbose:
        print(f"[DEBUG] 请求端点: {args.endpoint}")
        print(f"[DEBUG] 请求参数: {json.dumps(input_data, indent=2)}")

    # 执行查询
    result = send_query(
        query=final_query,
        top_k=final_top_k,
        endpoint=args.endpoint,
        timeout=args.timeout
    )

    # 结果输出
    output_str = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"结果已保存到 {args.output}")
    else:
        print(output_str)


from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field


class MCPQueryInput(BaseModel):
    query: str = Field(description="搜索查询内容")
    top_k: Optional[int] = Field(default=3, description="返回的结果数量")


class MCPRAGTool(BaseTool):
    name: str = "mcp_rag_search"
    description: str = """
    当需要查询特定领域知识或最新信息时使用此工具。
    输入应该是详细的搜索查询语句。
    """
    args_schema: Type[BaseModel] = MCPQueryInput

    def _run(self, query: str, top_k: int = 3) -> str:
        # 调用你现有的MCP服务
        response = send_query(query, top_k=top_k)

        if "error" in response:
            return f"查询失败: {response['error']}"

        # 格式化返回结果
        results = []
        for item in response.get("results", []):
            results.append(
                f"文件路径: {item.get('path', '无')}\n"
                f"内容: {item.get('excerpt', '无')}\n"
            )

        return "\n\n".join(results) if results else "未找到相关结果"

    async def _arun(self, query: str, top_k: int = 10) -> str:
        # 异步支持（可选）
        return self._run(query, top_k)


from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

tools = [MCPRAGTool()]


# 修改后的 create_mcp_agent 函数
def create_mcp_agent():
    # 1. 定义工具集
    tools = [MCPRAGTool()]

    # 2. 创建带类型标注的内存对象
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,  # 关键修复点
        output_key="output"
    )

    # 3. 更新提示模板中的消息占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
                你是一个智能助手，你必须访问MCP知识库系统获取专业信息。
                请用中文友好地回答用户问题，请使用工具获取最新信息。
                如果使用工具查询，请确保查询语句精确且完整。
                如果MCP知识库中不存在答案，请回复“我不知道”，不要自己编造答案。
                """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    # 4. 选择LLM模型
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 5. 创建Agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # 6. 创建执行器（集成内存）
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,  # 新增关键参数
        handle_parsing_errors=True
    )


# 修改后的调用函数
def main2():
    agent = create_mcp_agent()
    while True:
        question = input("用户输入: ")
        response = agent.invoke({"input": question})
        print(f"助手回复: {response['output']}")


if __name__ == "__main__":
    main()
    # main2()

# python client.py --query LSTM --pretty
# python client.py --endpoint http://localhost:8123/search2 --query LSTM --pretty
