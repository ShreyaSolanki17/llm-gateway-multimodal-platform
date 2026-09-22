import sqlite3
import pytest
from app.mcp.db import QueryValidationError, SQLiteExecutor, is_read_only_query


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM users",
        "select id, name from users where id = 1",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
        "  SELECT 1  ",
    ],
)
def test_is_read_only_query_accepts_select_and_with(sql):
    assert is_read_only_query(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO users (name) VALUES ('x')",
        "UPDATE users SET name = 'x'",
        "DELETE FROM users",
        "DROP TABLE users",
        "CREATE TABLE x (id INT)",
        "SELECT * FROM users; DROP TABLE users",
        "",
        "   ",
        "SELECT * FROM users WHERE name = 'x'; SELECT * FROM orders",
    ],
)
def test_is_read_only_query_rejects_mutations_and_stacked_statements(sql):
    assert is_read_only_query(sql) is False


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    conn.close()
    return db_path


def test_executor_query_returns_rows(seeded_db):
    executor = SQLiteExecutor(db_path=seeded_db)
    rows = executor.query("SELECT * FROM users ORDER BY id")
    assert rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


def test_executor_query_rejects_mutation(seeded_db):
    executor = SQLiteExecutor(db_path=seeded_db)
    with pytest.raises(QueryValidationError):
        executor.query("DELETE FROM users")


def test_executor_query_respects_row_limit(seeded_db):
    executor = SQLiteExecutor(db_path=seeded_db, row_limit=1)
    rows = executor.query("SELECT * FROM users ORDER BY id")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_executor_list_schema(seeded_db):
    executor = SQLiteExecutor(db_path=seeded_db)
    schema = executor.list_schema()
    assert "users" in schema
    column_names = {c["name"] for c in schema["users"]}
    assert column_names == {"id", "name"}


def test_executor_list_schema_empty_database(tmp_path):
    db_path = str(tmp_path / "empty.db")
    executor = SQLiteExecutor(db_path=db_path)
    assert executor.list_schema() == {}
