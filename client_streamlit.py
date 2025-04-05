import os
import sys
import time
from typing import Optional, Tuple, Type, Union

import requests
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

from config.config import global_config
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_community.tools.tavily_search import TavilySearchResults  # TAVILY工具
import json
from typing import Dict, Any

# Streamlit页面配置
st.set_page_config(
    page_title="MCP知识库助手",
    page_icon=":books:",
    layout="centered"
)

st.write("""
<style>
/* 不同来源的视觉区分 */
div[data-testid="stMarkdownContainer"] ul {
    position: relative;
    padding-left: 1.5em;
}

div[data-testid="stMarkdownContainer"] ul:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0.4em;
    height: 80%;
    width: 2px;
    background: linear-gradient(#2ecc71, #3498db);
}

/* 网络结果样式 */
.web-result {
    border-left: 3px solid #3498db;
    padding-left: 1rem;
    margin: 1rem 0;
}

/* 知识库结果样式 */
.kb-result {
    border-left: 3px solid #2ecc71;
    padding-left: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.write("""
<style>
.md-box {
    padding: 1.2rem;
    border-left: 4px solid #2ecc71;
    background: #f8f9fa;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)

# # 强制禁用缓存
# @st.cache_resource(ttl=1)  # 1秒缓存周期
# def get_client():
#     return global_config.sync_proxy_client

DEFAULT_ENDPOINT = "http://localhost:8123/search"


class MCPQueryInput(BaseModel):
    query: str = Field(description="搜索查询内容")
    top_k: int = Field(default=3, description="返回的最大结果数量")
    knowledge_name: Optional[str] = Field(default=None)
    no_chunk: Optional[bool] = Field(default=False)


class MCPRAGTool(BaseTool):
    name: str = "mcp_rag_search"
    description: str = "当需要查询特定领域知识或最新信息时使用此工具"
    # 修正args_schema的类型注解
    args_schema: Type[BaseModel] = MCPQueryInput

    def _run(self, **params) -> str:
        response = requests.post(
            DEFAULT_ENDPOINT,
            json={k: v for k, v in params.items() if v is not None},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if not response.ok:
            return (f"查询失败: {response.text}", {})

        raw_data = response.json()
        formatted = self._format_response(raw_data)
        return (formatted, {"raw_results": raw_data})

        # return self._format_response(response) if response.ok else f"查询失败: {response.text}"
    def _format_response(self, raw_data: dict) -> str:
        """生成包含结构化数据的Markdown格式字符串"""
        results = []
        for idx, item in enumerate(raw_data.get("results", []), 1):  # 只取前3个结果
            path = item.get('path', '无路径信息')
            content = str(item.get('excerpt', '')).strip()
            
            results.append(
                f"### 结果 {idx}\n"
                f"**路径**: `{path}`\n"
                f"**内容**: {content}\n"
                "---"
            )
        return "\n\n".join(results) if results else "未找到相关结果"
    # def _format_response(self, response) -> str:
    #     results = [
    #         f"文件路径: {item.get('path', '无')}\n内容: {item.get('excerpt', '无')}"
    #         for item in response.json().get("results", [])
    #     ]
    #     return "\n\n".join(results) if results else "未找到相关结果"

def create_mcp_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3,
                     http_client=global_config.sync_proxy_client)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [MCPRAGTool(),TavilySearchResults(max_results=3)]
    my_prompt_template = """# 指令系统 v2.1
                            ## 知识库配置
                            - 当前库：{knowledge_name} 
                            - 返回数：{top_k}

                            ## 强制流程
                            1. 首次必须使用`mcp_rag_search`查询（参数必须包含：query+knowledge_name+top_k）
                            2. 当且仅当出现以下情况时使用`web_search`：
                            - 根据知识库的回答无法准确回答问题，相关信息不明确或者找不到,当前问题为{question}
                            - 用户明确要求实时信息
                            3. 最终回答必须包含：
                            - 📌 来源标记（知识库/网络）
                            - 🔍 检索关键词
                            - 📂 知识库路径（若最终采用知识库则必须罗列参考路径）
                            - 📂 对应网址（若最终采用网络搜索则必须罗列网址）

                            ## 错误处理
                            ❌ 禁止在知识库无结果时自行推理
                            ✅ 必须通过工具获取确切信息"""

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", my_prompt_template),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_tools_agent(llm, tools, prompt),
        tools=tools,
        verbose=True,
        return_intermediate_steps=True, #返回中间结果
        handle_parsing_errors="请重新表述您的问题"
    )


def knowledge_base_selector():
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])  # 调整列宽比例
    with col1:
        global_config.knowledge_manager.load_knowledge_base()
        kb_list = getattr(global_config.knowledge_manager, 'all_knowledge_base', [])
        kb_names = [kb.knowledge_name for kb in kb_list]
        st.session_state.selected_kb = st.selectbox(
            "选择知识库",
            options=kb_names,
            index=0
        )
    with col2:
        if st.button("＋", help="新建知识库"):
            st.session_state.show_create_kb = True

    with col3:
        st.session_state.top_k = st.number_input(
            "最大结果数",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="设置返回结果的最大数量"
        )

    if st.session_state.get("show_create_kb"):
        with st.expander("新建知识库", expanded=True):
            with st.form("create_kb_form"):
                name = st.text_input("名称")
                path = st.text_input("路径")
                description = st.text_input("知识库描述")
                chunk_size = st.number_input("分块大小", min_value=300, max_value=2000, value=1000, step=100,
                                             help="分块大小是指将文件分块再进行检索，分块越小，检索越细致")
                chunk_overlap = st.number_input("分块重叠", min_value=0, max_value=500, value=200, step=100,
                                                help="分块重叠是指每两个分块之间的重叠大小")
                if st.form_submit_button("创建") and name and path:
                    try:
                        global_config.knowledge_manager.create_knowledge_base(name, path, description, chunk_size,
                                                                              chunk_overlap)
                        st.success("创建成功")
                        st.session_state.show_create_kb = False
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"创建失败: {str(e)}")


def main():
    st.title("MCP专业知识问答系统 :books:")
    knowledge_base_selector()

    # 初始化会话状态
    if "agent" not in st.session_state:
        st.session_state.agent = create_mcp_agent()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # 处理用户输入
    if prompt := st.chat_input("请输入问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.agent.invoke({
                "input": prompt,
                "question": prompt,
                "knowledge_name": st.session_state.selected_kb,
                # "kb_list": ", ".join([kb.knowledge_name for kb in global_config.knowledge_manager.all_knowledge_base]),
                "top_k": st.session_state.top_k
            })

            # 显示中间步骤
            if response.get("intermediate_steps"):
                with st.expander("🧠 思考过程", expanded=False):
                    for i, step in enumerate(response["intermediate_steps"], 1):
                        action, result = step[0], step[1]
                        
                        step_content = [f"### 步骤 {i}"]
                        
                        if isinstance(action, ToolAgentAction): 
                            # 工具类型标识
                            tool_type = "联网搜索" if action.tool == "tavily_search_results_json" else "本地知识库查询"
                            step_content.append(f"**工具类型**: {tool_type}")
                            
                            # 通用信息展示
                            step_content.extend([
                                f"**工具名称**: `{action.tool}`",
                                "**参数**:",
                                "\n".join([f"- `{k}`: `{v}`" for k, v in action.tool_input.items()]),
                                f"**日志**:\n```\n{action.log.strip()}\n```"
                            ])
                            
                            # 结果特殊处理
                            step_content.append("**执行结果**:")
                            if action.tool == "tavily_search_results_json":
                                if isinstance(result, tuple):
                                    # 处理(content, raw_data)格式
                                    content, raw_data = result
                                    step_content.append(content)
                                    with st.expander("查看原始数据", expanded=False):
                                        st.json(raw_data["raw_results"])
                                elif isinstance(result, str) and result.startswith("联网搜索失败"):
                                    step_content.append(f"❌ {result}")
                                else:
                                    step_content.append(f"```\n{str(result)[:300]}\n```")
                            elif action.tool == "mcp_rag_search":
                                if isinstance(result, tuple) and len(result) == 2:
                                    content, raw_data = result
                                    step_content.append(content)
                                    # 使用容器替代expand
                               
                                    for idx, item in enumerate(raw_data.get("raw_results", {}).get("results", []), 1):
                                        step_content.append(f"**结果** {idx}\n")
                                        step_content.append(f"路径: {item.get('path', '无路径')}\n")
                                        step_content.append(f"内容摘要: {str(item.get('excerpt', '无内容'))[:300]}\n")
                                        step_content.append("---")
                            
                            else:
                                step_content.append(f"```\n{str(result)[:300]}\n```")
                        
                        st.markdown("\n\n".join(step_content))
                        st.divider()
                        

            full_response = response.get("output", "无法生成回答")
            display_text = ""
            placeholder = st.empty()

            sentences = sent_tokenize(full_response)
            display_text = ""

            for sent in sentences:
                display_text += sent + " "
                placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
                time.sleep(0.05)
            placeholder.markdown(display_text)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
