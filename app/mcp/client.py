import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClient:
    """Reusable client for calling tools on a local MCP server module.

    Spawns the server as a subprocess (python -m <server_module>) and speaks
    the real MCP protocol over its stdin/stdout -- proves the servers work
    end-to-end over the wire protocol, not just as directly-called Python.

    Standalone by design: not wired into the chat endpoint. Wiring tool
    calls into a live chat request would add an agentic tool-use loop,
    which is out of scope for this gateway (see README's Core Distinction).
    """

    def __init__(self, server_module: str, env: Optional[Dict[str, str]] = None):
        merged_env = {**os.environ, **(env or {})}
        self._server_params = StdioServerParameters(command=sys.executable, args=["-m", server_module], env=merged_env)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[str]:
        """Return the names of every tool this server exposes."""
        async with self.session() as session:
            result = await session.list_tools()
            return [tool.name for tool in result.tools]

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call a tool by name and return its structured result."""
        # Raise after the session context manager exits cleanly rather than inside it --
        # raising mid-block causes anyio's task group teardown to wrap the error in an
        # ExceptionGroup, which breaks straightforward `except RuntimeError` handling.
        async with self.session() as session:
            result = await session.call_tool(tool_name, arguments or {})

        if result.is_error:
            raise RuntimeError(f"Tool '{tool_name}' failed: {result.content}")

        value = result.structured_content if result.structured_content is not None else result.content
        # The server wraps non-object return values (lists, scalars) as {"result": ...}
        # per the tool's auto-generated output schema; unwrap that envelope for callers.
        if isinstance(value, dict) and set(value.keys()) == {"result"}:
            return value["result"]
        return value
