def print_messages(result: dict) -> None:
    """メッセージを人間にとってわかりやすく出力する"""
    messages = result.get("messages", [])

    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage":
            print(f"\n[{i}] Human:")
            print(f"    {msg.content}")

        elif msg_type == "AIMessage":
            print(f"\n[{i}] AI:")
            if msg.content:
                print(f"    Content: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print("    Tool Calls:")
                for tool_call in msg.tool_calls:
                    print(
                        f"      - {tool_call.get('name', 'unknown')}({tool_call.get('args', {})})"
                    )

        elif msg_type == "ToolMessage":
            print(f"\n[{i}] Tool:")
            print(f"    Name: {msg.name if hasattr(msg, 'name') else 'unknown'}")
            print(f"    Result: {msg.content}")

        else:
            print(f"\n[{i}] ❓ {msg_type}:")
            print(f"    {msg}")
