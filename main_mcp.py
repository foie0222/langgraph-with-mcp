"""
LangGraph ToolNode と 実際のMCP の統合サンプル（非同期処理版）

このサンプルでは、Streamable HTTP経由でローカルMCPサーバー (mcp_server.py) に接続し、
LangGraphのToolNodeで使用します。

重要なポイント:
1. MCPサーバーをStreamable HTTPで起動
2. MCPクライアントで非同期通信
3. LangGraphのToolNodeで使用

実行前の準備:
1. 別のターミナルでMCPサーバーを起動: python mcp_server.py
2. サーバーが http://localhost:8000/mcp で起動していることを確認
"""

import asyncio
from contextlib import asynccontextmanager

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from utils import print_messages

# =============================================================================
# MCPクライアントの設定
# =============================================================================


class MCPClientManager:
    """MCPサーバーとの接続を管理するクラス"""

    def __init__(self, base_url: str = "http://localhost:8000/mcp"):
        self.base_url = base_url
        self.session: ClientSession | None = None

    @asynccontextmanager
    async def connect(self):
        """MCPサーバーに接続（Streamable HTTP）"""
        async with streamablehttp_client(self.base_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                yield session


mcp_manager = MCPClientManager()


# =============================================================================
# LangGraphの設定
# =============================================================================

llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
)


def create_graph(tools: list[BaseTool]) -> CompiledStateGraph:
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: MessagesState) -> dict:
        """非同期でLLMを呼び出す"""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def main() -> None:
    async with mcp_manager.connect():
        tools = await load_mcp_tools(mcp_manager.session)
        graph = create_graph(tools)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="1.5と3の加算を行ってください。")]}
        )
        print_messages(result)


if __name__ == "__main__":
    asyncio.run(main())
