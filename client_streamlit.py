import os
import sys
import time
from typing import Optional, Type

import requests
import streamlit as st
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

from config.config import global_config

# 调试模式设置
if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    import debugpy

    debugpy.listen(5678)
    debugpy.wait_for_client()
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Streamlit页面配置
st.set_page_config(
    page_title="MCP知识库助手",
    page_icon=":books:",
    layout="centered"
)

# 在Streamlit中增加CSS注入
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

        return self._format_response(response) if response.ok else f"查询失败: {response.text}"

    def _format_response(self, response) -> str:
        results = [
            f"文件路径: {item.get('path', '无')}\n内容: {item.get('excerpt', '无')}"
            for item in response.json().get("results", [])
        ]
        return "\n\n".join(results) if results else "未找到相关结果"





def create_mcp_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [MCPRAGTool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "当前知识库：{knowledge_name}\n知识库的结果返回数：{top_k}可用知识库：{kb_list}\n请严格根据知识库内容回答。"
                   "请在严格在当前知识库中查找知识，如果没有答案，请返回‘知识库中不存在’，"
                   "请严格遵循工具调用规则，确保参数匹配。"
                   "最终请输出参考的文件的路径，确保可以访问。"
                   "请确保所有的文字输出都采用markdown格式，格式直观、简洁、优雅，记得在合适的地方分段。"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    return AgentExecutor.from_agent_and_tools(
        agent=create_openai_tools_agent(llm, tools, prompt),
        tools=tools,
        verbose=True,
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
                "knowledge_name": st.session_state.selected_kb,
                "kb_list": ", ".join([kb.knowledge_name for kb in global_config.knowledge_manager.all_knowledge_base]),
                "top_k": st.session_state.top_k
            })

            full_response = response.get("output", "无法生成回答")
            display_text = ""
            placeholder = st.empty()

            sentences = sent_tokenize(full_response)
            display_text = ""

            for sent in sentences:
                display_text += sent + " "
                placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
                time.sleep(0.05)

            # display_text = display_text.replace("\n", "\\\n")  # 保留换行符
            # 使用带CSS的HTML包装
            # placeholder.markdown(f"""
            # <div style="line-height:1.8;font-family: 'SF Mono', Consolas, monospace">
            # {display_text}
            # </div>
            # """, unsafe_allow_html=True)

            # # 流式显示效果
            # for word in full_response.split():
            #     display_text += word + " "
            #     placeholder.markdown(display_text + "▌")
            #     time.sleep(0.05)
            placeholder.markdown(display_text)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
