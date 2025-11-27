from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from utils import print_messages

llm = ChatBedrockConverse(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="ap-northeast-1",
)


def call_model(state: MessagesState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


def main() -> None:
    graph = create_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="1.5と3の加算を行ってください。")]}
    )

    print_messages(result)


if __name__ == "__main__":
    main()
