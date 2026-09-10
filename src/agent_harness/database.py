"""Database access tools with explicit read/write and transaction boundaries."""

from __future__ import annotations

import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from langchain_core.tools import BaseTool, StructuredTool

SQLParameters = Mapping[str, Any] | Sequence[Any] | None
DatabaseResult = dict[str, Any]


class DatabaseBackend(ABC):
    """Backend contract used by :class:`DatabaseToolkit`.

    A PostgreSQL or other backend can implement this interface without changing the
    public tools. Backend exceptions are deliberately handled at the toolkit boundary.
    """

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return user-visible table names."""

    @abstractmethod
    def get_schema(self, table: str) -> DatabaseResult:
        """Return the structure of one table."""

    @abstractmethod
    def execute_query(
        self, sql: str, parameters: SQLParameters, max_rows: int
    ) -> DatabaseResult:
        """Execute one read-only statement."""

    @abstractmethod
    def execute_write(self, sql: str, parameters: SQLParameters) -> DatabaseResult:
        """Execute one mutating statement."""

    def close(self) -> None:
        """Release backend-owned resources, if any."""


@dataclass(frozen=True, slots=True)
class SQLiteConfig:
    """Configuration for a Toolkit-owned SQLite connection."""

    database: str | Path = ":memory:"
    timeout: float = 5.0
    uri: bool = False


class DatabaseSQLValidationError(ValueError):
    """A stable, backend-independent SQL validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _masked_sql(sql: str) -> str:
    """Mask comments and quoted contents while retaining punctuation/depth."""

    output: list[str] = []
    index = 0
    state = "plain"
    quote = ""
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "plain":
            if char == "-" and following == "-":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if char in {"'", '"', "`"}:
                output.append(" ")
                quote = char
                state = "quoted"
            elif char == "[":
                output.append(" ")
                quote = "]"
                state = "quoted"
            else:
                output.append(char)
            index += 1
            continue
        if state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "plain"
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "plain"
            else:
                output.append(" ")
                index += 1
            continue
        output.append(" ")
        if char == quote:
            # SQL escapes a quote by doubling it.
            if following == quote and quote != "]":
                output.append(" ")
                index += 2
                continue
            state = "plain"
        index += 1
    if state == "block_comment":
        raise DatabaseSQLValidationError("invalid_sql", "SQL contains an open comment.")
    if state == "quoted":
        raise DatabaseSQLValidationError("invalid_sql", "SQL contains an open quote.")
    return "".join(output)


def _is_single_create_trigger(masked: str) -> bool:
    """Recognize one SQLite trigger while ignoring semicolons in its body."""

    matches = list(re.finditer(r"[A-Za-z_][A-Za-z0-9_$]*", masked.upper()))
    words = [match.group(0) for match in matches]
    if len(words) < 3 or words[0] != "CREATE":
        return False
    trigger_index = 2 if words[1] in {"TEMP", "TEMPORARY"} else 1
    if trigger_index >= len(words) or words[trigger_index] != "TRIGGER":
        return False

    try:
        begin_index = words.index("BEGIN", trigger_index + 1)
    except ValueError:
        return False

    case_depth = 0
    for index in range(begin_index + 1, len(words)):
        if words[index] == "CASE":
            case_depth += 1
        elif words[index] == "END":
            if case_depth:
                case_depth -= 1
                continue
            suffix = masked[matches[index].end() :].strip()
            return suffix in {"", ";"}
    return False


def _sql_tokens(sql: str) -> list[str]:
    masked = _masked_sql(sql)
    semicolons = [match.start() for match in re.finditer(";", masked)]
    has_multiple = semicolons and (
        len(semicolons) > 1 or masked[semicolons[0] + 1 :].strip()
    )
    if has_multiple and not _is_single_create_trigger(masked):
        raise DatabaseSQLValidationError(
            "multiple_statements", "Only one SQL statement is allowed per tool call."
        )
    return re.findall(r"[A-Za-z_][A-Za-z0-9_$]*|[(),]", masked.upper())


def _statement_kind(sql: str) -> str:
    tokens = _sql_tokens(sql)
    if not tokens:
        raise DatabaseSQLValidationError("invalid_sql", "SQL must not be empty.")
    if tokens[0] != "WITH":
        return tokens[0]

    depth = 0
    saw_cte_body = False
    for token in tokens[1:]:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise DatabaseSQLValidationError(
                    "invalid_sql", "SQL has unbalanced parentheses."
                )
            if depth == 0:
                saw_cte_body = True
        elif saw_cte_body and depth == 0 and token in {
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
        }:
            return token
    raise DatabaseSQLValidationError(
        "invalid_sql", "Unable to determine the operation after WITH."
    )


def _validate_query(sql: str) -> None:
    if _statement_kind(sql) != "SELECT":
        raise DatabaseSQLValidationError(
            "read_only", "execute_query only accepts SELECT statements."
        )


def _validate_write(sql: str, *, allow_dangerous: bool) -> None:
    leading = re.search(r"[A-Za-z_][A-Za-z0-9_$]*", _masked_sql(sql))
    dangerous = {"CREATE", "ALTER", "DROP"}
    if leading and leading.group(0).upper() in dangerous and not allow_dangerous:
        kind = leading.group(0).upper()
        raise DatabaseSQLValidationError(
            "dangerous_sql",
            f"{kind} is disabled. Set allow_dangerous=True to enable it.",
        )

    kind = _statement_kind(sql)
    if kind == "SELECT":
        raise DatabaseSQLValidationError(
            "write_only", "execute_write does not accept SELECT statements."
        )
    supported = {"INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "ALTER", "DROP"}
    if kind not in supported:
        raise DatabaseSQLValidationError(
            "unsupported_statement", f"Unsupported write statement type: {kind}."
        )
    if kind in dangerous and not allow_dangerous:
        raise DatabaseSQLValidationError(
            "dangerous_sql",
            f"{kind} is disabled. Set allow_dangerous=True to enable it.",
        )


class SQLiteBackend(DatabaseBackend):
    """SQLite implementation with caller-owned transaction preservation."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection | None = None,
        config: SQLiteConfig | None = None,
        allow_dangerous: bool = False,
    ) -> None:
        if connection is not None and config is not None:
            raise ValueError("Pass either connection or config, not both")
        self._owns_connection = connection is None
        if connection is None:
            resolved = config or SQLiteConfig()
            connection = sqlite3.connect(
                str(resolved.database),
                timeout=resolved.timeout,
                uri=resolved.uri,
                check_same_thread=False,
            )
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection")
        self._connection = connection
        self._allow_dangerous = allow_dangerous
        self._lock = threading.RLock()

    @property
    def owns_connection(self) -> bool:
        """Whether this backend owns commits, rollbacks, and connection lifetime."""

        return self._owns_connection

    def list_tables(self) -> list[str]:
        with self._lock:
            cursor = self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            return [str(row[0]) for row in cursor.fetchall()]

    def get_schema(self, table: str) -> DatabaseResult:
        if not table or not table.strip():
            raise DatabaseSQLValidationError(
                "invalid_table", "Table name must not be empty."
            )
        with self._lock:
            found = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if found is None:
                raise DatabaseSQLValidationError(
                    "table_not_found", f"Table '{table}' does not exist."
                )
            quoted = table.replace('"', '""')
            rows = self._connection.execute(f'PRAGMA table_info("{quoted}")').fetchall()
        columns = [
            {
                "name": row[1],
                "type": row[2] or None,
                "nullable": not bool(row[3]),
                "default": row[4],
                "primary_key": bool(row[5]),
            }
            for row in rows
        ]
        return {"table": table, "columns": columns}

    @staticmethod
    def _parameters(parameters: SQLParameters) -> Mapping[str, Any] | Sequence[Any]:
        if parameters is None:
            return ()
        if isinstance(parameters, (str, bytes, bytearray)):
            raise DatabaseSQLValidationError(
                "invalid_parameters", "SQL parameters must be an object or an array."
            )
        return parameters

    def execute_query(
        self, sql: str, parameters: SQLParameters, max_rows: int
    ) -> DatabaseResult:
        _validate_query(sql)
        with self._lock:
            cursor = self._connection.execute(sql, self._parameters(parameters))
            raw_rows = cursor.fetchmany(max_rows + 1)
            names: list[str] = []
            seen: dict[str, int] = {}
            for item in cursor.description or ():
                original = str(item[0])
                count = seen.get(original, 0) + 1
                seen[original] = count
                names.append(original if count == 1 else f"{original}_{count}")
        truncated = len(raw_rows) > max_rows
        rows = [dict(zip(names, row, strict=False)) for row in raw_rows[:max_rows]]
        return {
            "columns": names,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "max_rows": max_rows,
        }

    def execute_write(self, sql: str, parameters: SQLParameters) -> DatabaseResult:
        _validate_write(sql, allow_dangerous=self._allow_dangerous)
        with self._lock:
            try:
                cursor = self._connection.execute(sql, self._parameters(parameters))
                result = {
                    "rows_affected": max(cursor.rowcount, 0),
                    "last_insert_id": cursor.lastrowid,
                    "committed": self._owns_connection,
                }
                if self._owns_connection:
                    self._connection.commit()
                return result
            except Exception:
                if self._owns_connection:
                    self._connection.rollback()
                raise

    def close(self) -> None:
        if self._owns_connection:
            with self._lock:
                self._connection.close()


class DatabaseToolkit:
    """Expose backend-neutral database operations as standard LangChain tools.

    Read tools are returned by default. Set ``include_write=True`` to expose the
    separate write tool. Caller-provided SQLite connections remain caller-owned:
    writes are neither committed nor rolled back by this toolkit.
    """

    def __init__(
        self,
        connection: sqlite3.Connection | None = None,
        *,
        config: SQLiteConfig | None = None,
        backend: DatabaseBackend | None = None,
        max_rows: int = 100,
        include_write: bool = False,
        require_write_approval: bool = False,
        allow_dangerous: bool = False,
        tool_name_prefix: str = "",
    ) -> None:
        supplied = sum(item is not None for item in (connection, config, backend))
        if supplied > 1:
            raise ValueError("Pass only one of connection, config, or backend")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        if require_write_approval and not include_write:
            raise ValueError("require_write_approval requires include_write=True")
        if backend is None:
            backend = SQLiteBackend(
                connection=connection,
                config=config,
                allow_dangerous=allow_dangerous,
            )
        elif allow_dangerous:
            raise ValueError("allow_dangerous must be configured on a custom backend")
        self.backend = backend
        self.max_rows = max_rows
        self.include_write = include_write
        self.require_write_approval = require_write_approval
        self.tool_name_prefix = tool_name_prefix.strip("_")

    def _name(self, operation: str) -> str:
        return (
            f"{self.tool_name_prefix}_{operation}"
            if self.tool_name_prefix
            else operation
        )

    @staticmethod
    def _error(operation: str, error: Exception) -> DatabaseResult:
        retryable = False
        if isinstance(error, DatabaseSQLValidationError):
            code, message = error.code, str(error)
        elif isinstance(error, sqlite3.IntegrityError):
            code, message = (
                "constraint_violation",
                "The write violates a database constraint; inspect the schema and values.",
            )
        elif isinstance(error, sqlite3.OperationalError):
            lowered = str(error).lower()
            if "no such table" in lowered:
                code, message = "table_not_found", "A referenced table does not exist."
            elif "no such column" in lowered:
                code, message = "column_not_found", "A referenced column does not exist."
            elif "syntax error" in lowered or "incomplete input" in lowered:
                code, message = "invalid_sql", "The SQL syntax is invalid or incomplete."
            elif "locked" in lowered or "busy" in lowered:
                code, message, retryable = (
                    "database_busy",
                    "The database is temporarily busy; retry the operation later.",
                    True,
                )
            else:
                code, message = "database_error", "The database rejected the operation."
        elif isinstance(error, (sqlite3.DatabaseError, sqlite3.ProgrammingError)):
            code, message = "database_error", "The database rejected the operation."
        else:
            code, message = "database_error", "The database operation could not be completed."
        return {
            "ok": False,
            "operation": operation,
            "error": {"type": code, "message": message, "retryable": retryable},
        }

    def _run(self, operation: str, function: Any, *args: Any) -> DatabaseResult:
        try:
            payload = function(*args)
            if operation == "list_tables":
                payload = {"tables": payload, "count": len(payload)}
            return {"ok": True, "operation": operation, **payload}
        except Exception as error:  # noqa: BLE001 - stable tool error boundary
            return self._error(operation, error)

    def list_tables(self) -> DatabaseResult:
        """List available user tables."""

        return self._run("list_tables", self.backend.list_tables)

    def get_schema(self, table: str) -> DatabaseResult:
        """Get column metadata for a table."""

        return self._run("get_schema", self.backend.get_schema, table)

    def execute_query(
        self, sql: str, parameters: dict[str, Any] | list[Any] | None = None
    ) -> DatabaseResult:
        """Execute one SELECT statement with a bounded result set."""

        return self._run(
            "execute_query", self.backend.execute_query, sql, parameters, self.max_rows
        )

    def execute_write(
        self, sql: str, parameters: dict[str, Any] | list[Any] | None = None
    ) -> DatabaseResult:
        """Execute one non-SELECT statement using the configured transaction owner."""

        return self._run("execute_write", self.backend.execute_write, sql, parameters)

    async def _alist_tables(self) -> DatabaseResult:
        return self.list_tables()

    async def _aget_schema(self, table: str) -> DatabaseResult:
        return self.get_schema(table)

    async def _aexecute_query(
        self, sql: str, parameters: dict[str, Any] | list[Any] | None = None
    ) -> DatabaseResult:
        return self.execute_query(sql, parameters)

    async def _aexecute_write(
        self, sql: str, parameters: dict[str, Any] | list[Any] | None = None
    ) -> DatabaseResult:
        return self.execute_write(sql, parameters)

    def get_tools(self, *, include_write: bool | None = None) -> list[BaseTool]:
        """Build standard LangChain tools, with writes excluded by default."""

        expose_write = self.include_write if include_write is None else include_write
        common = {"database_toolkit": True}
        tools: list[BaseTool] = [
            StructuredTool.from_function(
                self.list_tables,
                coroutine=self._alist_tables,
                name=self._name("list_tables"),
                description="List the user tables available in the database.",
                metadata={**common, "database_operation": "read"},
            ),
            StructuredTool.from_function(
                self.get_schema,
                coroutine=self._aget_schema,
                name=self._name("get_schema"),
                description="Get column names, types, nullability, defaults, and keys for a table.",
                metadata={**common, "database_operation": "read"},
            ),
            StructuredTool.from_function(
                self.execute_query,
                coroutine=self._aexecute_query,
                name=self._name("execute_query"),
                description=(
                    "Execute one parameterized SELECT query. Results are capped and report "
                    "whether rows were truncated."
                ),
                metadata={**common, "database_operation": "read"},
            ),
        ]
        if expose_write:
            metadata = {**common, "database_operation": "write"}
            if self.require_write_approval:
                metadata.update(
                    {
                        "harness_approval": True,
                        "harness_approval_message": "Approve this database write?",
                    }
                )
            tools.append(
                StructuredTool.from_function(
                    self.execute_write,
                    coroutine=self._aexecute_write,
                    name=self._name("execute_write"),
                    description=(
                        "Execute one parameterized INSERT, UPDATE, DELETE, REPLACE, or "
                        "allowed schema statement."
                    ),
                    metadata=metadata,
                )
            )
        return tools

    def close(self) -> None:
        """Close resources owned by the backend."""

        self.backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
