import operator

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from utils import print_messages


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """2つの数値で計算を実行する
    Args:
        operation (str): 操作の種類(add, subtract, multiply, divide)
        a (float): 1つ目の数値
        b (float): 2つ目の数値
    Returns:
        str: 計算結果またはエラーメッセージ
    """

    ops = {
        "add": (operator.add, "+"),
        "subtract": (operator.sub, "-"),
        "multiply": (operator.mul, "*"),
        "divide": (operator.truediv, "/"),
    }

    if operation not in ops:
        return f"Error: Unknown operation '{operation}'"

    func, symbol = ops[operation]

    if operation == "divide" and b == 0:
        return "Error: Division by zero"

    result = func(a, b)
    return f"Result: {a} {symbol} {b} = {result}"


tools = [calculate]

llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="ap-northeast-1",
)

llm_with_tools = llm.bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def main() -> None:
    graph = create_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="1.5と3の加算を行ってください。")]}
    )

    print_messages(result)


if __name__ == "__main__":
    main()
