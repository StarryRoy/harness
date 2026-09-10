"""Database access tools for agents, independent of application semantics."""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

DatabaseResult = dict[str, Any]


class DatabaseAdapter(ABC):
    """Backend contract used by :class:`DatabaseToolkit`.

    A PostgreSQL adapter can implement this contract without changing the tools or
    the agent integration.
    """

    @abstractmethod
    def list_tables(self) -> DatabaseResult: ...

    @abstractmethod
    def get_schema(self, table: str) -> DatabaseResult: ...

    @abstractmethod
    def execute_query(self, sql: str) -> DatabaseResult: ...

    @abstractmethod
    def execute_write(self, sql: str) -> DatabaseResult: ...

    def close(self) -> None:
        """Release adapter-owned resources, if any."""


@dataclass(frozen=True, slots=True)
class SQLiteConfig:
    """Configuration for an adapter-owned SQLite connection."""

    database: str | Path
    timeout: float = 5.0


_LEADING_COMMENTS = re.compile(r"\A(?:\s|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*", re.DOTALL)
_FIRST_WORD = re.compile(r"[A-Za-z]+")


def _operation(sql: str) -> str:
    remainder = _LEADING_COMMENTS.sub("", sql, count=1)
    match = _FIRST_WORD.match(remainder)
    return match.group(0).upper() if match else ""


def _error(kind: str, message: str) -> DatabaseResult:
    return {"ok": False, "error": {"type": kind, "message": message}}


class SQLiteAdapter(DatabaseAdapter):
    """SQLite implementation with query/write boundaries and managed transactions."""

    def __init__(self, source: sqlite3.Connection | SQLiteConfig | str | Path) -> None:
        if isinstance(source, sqlite3.Connection):
            self.connection = source
            self._owns_connection = False
        else:
            config = (
                source if isinstance(source, SQLiteConfig) else SQLiteConfig(source)
            )
            self.connection = sqlite3.connect(
                str(config.database), timeout=config.timeout
            )
            self._owns_connection = True
        self.connection.row_factory = sqlite3.Row

    @staticmethod
    def _validate_single_statement(sql: str) -> DatabaseResult | None:
        if not isinstance(sql, str) or not sql.strip():
            return _error("invalid_sql", "SQL must be a non-empty string.")
        return None

    @staticmethod
    def _translate_exception(exc: sqlite3.Error) -> DatabaseResult:
        if isinstance(exc, sqlite3.IntegrityError):
            return _error(
                "constraint_error",
                "The write violates a database constraint; check the supplied values.",
            )
        if isinstance(exc, sqlite3.OperationalError):
            return _error(
                "sql_error",
                "The database could not execute the SQL; check syntax, names, and locks.",
            )
        return _error("database_error", "The database operation failed safely.")

    def list_tables(self) -> DatabaseResult:
        try:
            rows = self.connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            return {"ok": True, "tables": [row[0] for row in rows]}
        except sqlite3.Error as exc:
            return self._translate_exception(exc)

    def get_schema(self, table: str) -> DatabaseResult:
        if not isinstance(table, str) or not table.strip():
            return _error("invalid_table", "Table must be a non-empty name.")
        try:
            exists = self.connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if exists is None:
                return _error("table_not_found", "The requested table does not exist.")
            quoted = table.replace('"', '""')
            rows = self.connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "nullable": not bool(row[3]),
                    "default": row[4],
                    "primary_key": bool(row[5]),
                }
                for row in rows
            ]
            return {"ok": True, "table": table, "columns": columns}
        except sqlite3.Error as exc:
            return self._translate_exception(exc)

    def execute_query(self, sql: str) -> DatabaseResult:
        invalid = self._validate_single_statement(sql)
        if invalid:
            return invalid
        if _operation(sql) not in {"SELECT", "WITH"}:
            return _error("read_only", "execute_query only accepts SELECT queries.")

        denied = {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_TRANSACTION,
        }

        def authorize(action: int, *_args: Any) -> int:
            return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

        try:
            self.connection.set_authorizer(authorize)
            cursor = self.connection.execute(sql)
            names = [item[0] for item in cursor.description or ()]
            rows = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
            return {"ok": True, "columns": names, "rows": rows, "row_count": len(rows)}
        except sqlite3.Error as exc:
            return self._translate_exception(exc)
        finally:
            self.connection.set_authorizer(None)

    def execute_write(self, sql: str) -> DatabaseResult:
        invalid = self._validate_single_statement(sql)
        if invalid:
            return invalid
        if _operation(sql) not in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            return _error(
                "unsafe_operation",
                "execute_write only accepts INSERT, UPDATE, DELETE, or REPLACE.",
            )
        try:
            cursor = self.connection.execute(sql)
            self.connection.commit()
            return {
                "ok": True,
                "rows_affected": cursor.rowcount,
                "last_row_id": cursor.lastrowid,
            }
        except sqlite3.Error as exc:
            self.connection.rollback()
            return self._translate_exception(exc)

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()


class DatabaseToolkit:
    """Expose a database adapter as standard LangChain tools.

    Writes are omitted by default. Set ``include_write=True`` explicitly and wrap
    the returned ``execute_write`` tool with Harness ``require_approval`` when a
    human decision is required.
    """

    def __init__(
        self,
        database: DatabaseAdapter | sqlite3.Connection | SQLiteConfig | str | Path,
        *,
        include_write: bool = False,
    ) -> None:
        self.adapter = (
            database
            if isinstance(database, DatabaseAdapter)
            else SQLiteAdapter(database)
        )
        self.include_write = include_write

    def get_tools(self) -> list[BaseTool]:
        tools = [
            StructuredTool.from_function(
                self.adapter.list_tables,
                name="list_tables",
                description="List the database tables available to query.",
            ),
            StructuredTool.from_function(
                self.adapter.get_schema,
                name="get_schema",
                description="Get column names, types, nullability, defaults, and keys for a table.",
            ),
            StructuredTool.from_function(
                self.adapter.execute_query,
                name="execute_query",
                description="Run one read-only SELECT query and return structured rows.",
            ),
        ]
        if self.include_write:
            tools.append(
                StructuredTool.from_function(
                    self.adapter.execute_write,
                    name="execute_write",
                    description="Run one INSERT, UPDATE, DELETE, or REPLACE transaction.",
                    metadata={"harness_sensitive": True},
                )
            )
        return tools

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self) -> DatabaseToolkit:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
