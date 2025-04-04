# client.py
import argparse
import json
import sys
from typing import Dict, Any, Optional

import requests
from langchain.memory import ConversationBufferMemory

DEFAULT_ENDPOINT = "http://localhost:8123/search"


def send_query(
        query: str,
        top_k: int = 3,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 10,
        knowledge_name: Optional[str] = None,
        no_chunk: bool = False
) -> Dict[str, Any]:
    """发送查询到MCP服务并返回结构化结果"""
    payload = {
        "query": query,
        "top_k": top_k,
        "no_chunk": no_chunk
    }
    if knowledge_name is not None:
        payload["knowledge_name"] = knowledge_name

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
    parser.add_argument("--knowledge_name", type=str, help="指定知识库名称")
    parser.add_argument("--no_chunk", action="store_true", help="启用不分块检索")
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
    # elif not sys.stdin.isatty():  # 支持管道输入
    #     input_data = json.load(sys.stdin)
    elif args.query:
        input_data = {"query": args.query, "top_k": args.top_k}
    else:
        parser.print_help()
        sys.exit(1)

    # 参数合并（命令行参数优先）
    final_query = args.query or input_data.get("query", "")
    final_top_k = args.top_k or input_data.get("top_k", 3)
    final_knowledge_name = args.knowledge_name if args.knowledge_name is not None else input_data.get("knowledge_name")
    final_no_chunk = args.no_chunk or input_data.get("no_chunk", False)

    if args.verbose:
        print(f"[DEBUG] 请求端点: {args.endpoint}")
        debug_params = {
            "query": final_query,
            "top_k": final_top_k,
            "knowledge_name": final_knowledge_name,
            "no_chunk": final_no_chunk
        }
        print(f"[DEBUG] 请求参数: {json.dumps(debug_params, indent=2, ensure_ascii=False)}")

    # 执行查询
    result = send_query(
        query=final_query,
        top_k=final_top_k,
        endpoint=args.endpoint,
        timeout=args.timeout,
        knowledge_name=final_knowledge_name,
        no_chunk=final_no_chunk
    )

    # 结果输出
    output_str = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"结果已保存到 {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()