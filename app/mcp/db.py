import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings

_READ_ONLY_PREFIXES = ("select", "with")
_FORBIDDEN_KEYWORDS = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "create",
        "replace", "attach", "detach", "pragma", "vacuum", "truncate",
        "grant", "revoke", "exec", "execute",
    }
)


class QueryValidationError(ValueError):
    """Raised when a query is not a single, read-only SELECT/WITH statement."""


def is_read_only_query(sql: str) -> bool:
    """Reject anything that isn't a single SELECT/WITH statement.

    Defense in depth: checks the leading keyword, rejects stacked statements
    (semicolons), and scans for any mutating/DDL keyword anywhere in the text.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        return False

    first_word = stripped.split(None, 1)[0].lower()
    if first_word not in _READ_ONLY_PREFIXES:
        return False

    tokens = set(re.findall(r"[a-zA-Z_]+", stripped.lower()))
    if tokens & _FORBIDDEN_KEYWORDS:
        return False

    return True


class SQLiteExecutor:
    """Read-only database access for the PostgreSQL MCP server.

    ponytail: SQLite stand-in for a real Postgres connection -- swaps to
    asyncpg/psycopg against real Postgres at Milestone 17 once Docker infra
    exists. The query surface (read-only SELECT + schema introspection) is
    designed to carry over unchanged.
    """

    def __init__(self, db_path: Optional[str] = None, row_limit: Optional[int] = None):
        self._db_path = db_path or settings.MCP_DB_PATH
        self._row_limit = row_limit if row_limit is not None else settings.MCP_QUERY_ROW_LIMIT
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a read-only SELECT/WITH query and return up to row_limit rows."""
        if not is_read_only_query(sql):
            raise QueryValidationError("Only a single, read-only SELECT/WITH statement is allowed")

        conn = self._connect()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(self._row_limit)
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def list_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Return every table and its columns, keyed by table name."""
        conn = self._connect()
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()

            schema: Dict[str, List[Dict[str, str]]] = {}
            for table in tables:
                table_name = table["name"]
                # table_name comes only from sqlite_master (not user input), so
                # interpolating it into PRAGMA (which doesn't support parameters) is safe
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                schema[table_name] = [{"name": c["name"], "type": c["type"]} for c in columns]
            return schema
        finally:
            conn.close()
