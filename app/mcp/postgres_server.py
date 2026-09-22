from typing import Any, Dict, List
from mcp.server.mcpserver import MCPServer
from app.mcp.db import QueryValidationError, SQLiteExecutor

server = MCPServer(
    name="postgres-mcp-server",
    instructions="Read-only access to the gateway's relational database: run SELECT queries and inspect the schema.",
)
_executor = SQLiteExecutor()


@server.tool()
def query_database(sql: str) -> List[Dict[str, Any]]:
    """Run a read-only SELECT (or WITH ... SELECT) query and return the matching rows."""
    try:
        return _executor.query(sql)
    except QueryValidationError as exc:
        raise ValueError(str(exc))


@server.tool()
def list_schema() -> Dict[str, List[Dict[str, str]]]:
    """List every table in the database and its columns."""
    return _executor.list_schema()


if __name__ == "__main__":
    server.run()
