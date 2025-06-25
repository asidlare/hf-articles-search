from crewai_tools import MCPServerAdapter


def tools_checker():
    server_params = {
        "url": "http://localhost:8000/mcp",
    }

    try:
        with MCPServerAdapter(server_params) as tools:
            print(f"Available tools from SSE MCP server: {[tool.name for tool in tools]}")


    except Exception as e:
        print(f"Error connecting to or using SSE MCP server (Managed): {e}")
        print("Ensure the SSE MCP server is running and accessible at the specified URL.")


if __name__ == "__main__":
    tools_checker()
