import operator

from mcp.server.fastmcp import FastMCP  # Create MCP server instance

mcp = FastMCP("Calculate", host="localhost", port="8000", debug=True, log_level="INFO")


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
