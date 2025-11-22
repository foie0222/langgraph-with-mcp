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
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

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

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """MCPツールを呼び出す"""
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        result = await self.session.call_tool(tool_name, arguments)

        # 結果を文字列に変換
        if result.content:
            return "\n".join(
                item.text if isinstance(item, TextContent) else str(item)
                for item in result.content
            )
        return ""


# グローバルなMCPクライアントマネージャー
mcp_manager = MCPClientManager()


# =============================================================================
# MCPツールをLangChainツールにラップ
# =============================================================================


async def calculate_mcp(operation: str, a: float, b: float) -> str:
    """
    ローカルMCPサーバーの計算機ツールを使用

    これは本物のMCPツールです！

    Args:
        operation: 実行する演算 (add, subtract, multiply, divide)
        a: 最初の数値
        b: 2番目の数値
    """
    print(f"  [MCP] Calculating: {a} {operation} {b}...")
    result = await mcp_manager.call_tool(
        "calculate", {"operation": operation, "a": a, "b": b}
    )
    return result


# =============================================================================
# LangChainツールへの変換
# =============================================================================

calculate_tool = StructuredTool.from_function(
    coroutine=calculate_mcp,
    name="calculate",
    description="2つの数値で四則演算（加算、減算、乗算、除算）を実行します。",
)

tools = [calculate_tool]

# =============================================================================
# LangGraphの設定
# =============================================================================

llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
)

llm_with_tools = llm.bind_tools(tools)


async def call_model(state: MessagesState) -> dict:
    """非同期でLLMを呼び出す"""
    messages = state["messages"]
    print("\n LLM呼び出し中...")
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    """ToolNodeを含むエージェントグラフを作成"""
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# =============================================================================
# メイン処理
# =============================================================================


async def main() -> None:
    """非同期のメイン関数"""

    print("=" * 70)
    print("LangGraph + ローカルMCPサーバー（Streamable HTTP）統合デモ")
    print("=" * 70)
    print("\n📝 このデモでは:")
    print("  - ローカルMCPサーバー (mcp_server.py) に接続")
    print("  - Streamable HTTP で通信")
    print("  - LangGraphのToolNodeで使用")
    print("  - 非同期処理で効率的に実行")
    print("=" * 70)

    # MCPサーバーに接続
    print("\n🔌 HTTP MCPサーバーに接続中 (http://localhost:8000/mcp)...")
    try:
        async with mcp_manager.connect() as session:
            print("✅ MCPサーバーに接続しました")

            # 利用可能なツールを確認
            tools_list = await session.list_tools()
            print(f"\n📋 利用可能なMCPツール: {len(tools_list.tools)}個")
            for tool in tools_list.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n" + "=" * 70)

            # グラフを作成
            graph = create_graph()

            # テストクエリ
            test_queries = [
                "1.5と3の加算を行ってください。",
            ]

            for query in test_queries:
                print(f"\n\n{'=' * 70}")
                print(f"Query: {query}")
                print("=" * 70)

                try:
                    result = await graph.ainvoke(
                        {"messages": [HumanMessage(content=query)]}
                    )

                    print("\n" + "=" * 70)
                    print("実行結果:")
                    print("=" * 70)
                    print_messages(result)

                except Exception as e:
                    print(f"\n❌ エラーが発生しました: {e}")
                    import traceback

                    traceback.print_exc()

    except Exception as e:
        print(f"\n❌ MCPサーバーへの接続に失敗しました: {e}")
        print("\n以下を確認してください:")
        print("  1. MCPサーバーが起動しているか: python mcp_server.py")
        print("  2. サーバーが http://localhost:8000/mcp でアクセス可能か")
        import traceback

        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("✅ デモ完了")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
