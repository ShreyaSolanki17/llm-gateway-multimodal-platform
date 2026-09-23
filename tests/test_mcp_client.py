import sqlite3
import pytest
from app.mcp.client import MCPClient


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "e2e.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def postgres_client(seeded_db):
    return MCPClient("app.mcp.postgres_server", env={"MCP_DB_PATH": seeded_db})


@pytest.mark.asyncio
async def test_client_lists_tools_from_real_server_subprocess(postgres_client):
    tools = await postgres_client.list_tools()
    assert set(tools) == {"query_database", "list_schema"}


@pytest.mark.asyncio
async def test_client_calls_list_schema_tool(postgres_client):
    result = await postgres_client.call_tool("list_schema")
    assert "users" in result


@pytest.mark.asyncio
async def test_client_calls_query_database_tool(postgres_client):
    rows = await postgres_client.call_tool("query_database", {"sql": "SELECT * FROM users ORDER BY id"})
    assert rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@pytest.mark.asyncio
async def test_client_rejects_mutating_query_through_real_server(postgres_client):
    with pytest.raises(RuntimeError):
        await postgres_client.call_tool("query_database", {"sql": "DROP TABLE users"})
