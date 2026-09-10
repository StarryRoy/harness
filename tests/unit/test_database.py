import asyncio
import json
import sqlite3

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import Field

from agent_harness import DatabaseToolkit, SQLiteConfig, create_agent


class DatabaseToolModel(FakeMessagesListChatModel):
    bound_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_names = [tool.name for tool in tools]
        return self


def _tool_call(name, arguments, identifier):
    return {
        "name": name,
        "args": arguments,
        "id": identifier,
        "type": "tool_call",
    }


def _connection():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            score INTEGER DEFAULT 0
        );
        INSERT INTO users (name, score) VALUES
            ('Ada', 10), ('Grace', 20), ('Linus', 30);
        """
    )
    connection.commit()
    return connection


def test_list_schema_query_write_and_default_read_only_tools():
    connection = _connection()
    toolkit = DatabaseToolkit(connection, include_write=True, max_rows=2)
    try:
        listed = toolkit.list_tables()
        assert listed == {
            "ok": True,
            "operation": "list_tables",
            "tables": ["users"],
            "count": 1,
        }

        schema = toolkit.get_schema("users")
        assert schema["ok"] is True
        assert [column["name"] for column in schema["columns"]] == [
            "id",
            "name",
            "score",
        ]
        assert schema["columns"][0]["primary_key"] is True
        assert schema["columns"][1]["nullable"] is False

        queried = toolkit.execute_query("SELECT id, name FROM users ORDER BY id")
        assert queried["rows"] == [
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Grace"},
        ]
        assert queried["row_count"] == 2
        assert queried["max_rows"] == 2
        assert queried["truncated"] is True

        written = toolkit.execute_write(
            "UPDATE users SET score = ? WHERE name = ?", [42, "Ada"]
        )
        assert written["ok"] is True
        assert written["rows_affected"] == 1
        assert written["committed"] is False
        assert connection.execute(
            "SELECT score FROM users WHERE name = 'Ada'"
        ).fetchone() == (42,)

        read_only = DatabaseToolkit(connection).get_tools()
        assert [tool.name for tool in read_only] == [
            "list_tables",
            "get_schema",
            "execute_query",
        ]
        assert all(isinstance(tool, BaseTool) for tool in read_only)
    finally:
        connection.close()


def test_sql_errors_are_structured_and_do_not_expose_driver_exceptions():
    connection = _connection()
    toolkit = DatabaseToolkit(connection, include_write=True)
    try:
        syntax = toolkit.execute_query("SELECT FROM users")
        assert syntax["ok"] is False
        assert syntax["error"]["type"] == "invalid_sql"
        assert "OperationalError" not in json.dumps(syntax)

        wrong_boundary = toolkit.execute_query("DELETE FROM users")
        assert wrong_boundary["error"]["type"] == "read_only"

        dangerous = toolkit.execute_write("DROP TABLE users")
        assert dangerous["error"]["type"] == "dangerous_sql"
        assert toolkit.list_tables()["tables"] == ["users"]

        create_table = toolkit.execute_write("CREATE TABLE audit (id INTEGER)")
        assert create_table["error"]["type"] == "dangerous_sql"
        assert "audit" not in toolkit.list_tables()["tables"]

        create_trigger = toolkit.execute_write(
            "CREATE TRIGGER users_audit AFTER INSERT ON users "
            "BEGIN UPDATE users SET score = score; END"
        )
        assert create_trigger["error"]["type"] == "dangerous_sql"

        multiple = toolkit.execute_query("SELECT 1; SELECT 2")
        assert multiple["error"]["type"] == "multiple_statements"
    finally:
        connection.close()


def test_toolkit_owned_connection_commits_writes(tmp_path):
    database = tmp_path / "owned.sqlite"
    toolkit = DatabaseToolkit(
        config=SQLiteConfig(database), include_write=True, allow_dangerous=True
    )
    try:
        assert toolkit.execute_write(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT)"
        )["committed"] is True
        assert toolkit.list_tables()["tables"] == ["events"]
        assert toolkit.execute_write(
            "INSERT INTO events (name) VALUES (?)", ["created"]
        )["committed"] is True
        with sqlite3.connect(database) as observer:
            assert observer.execute("SELECT name FROM events").fetchall() == [
                ("created",)
            ]
    finally:
        toolkit.close()


def test_allow_dangerous_create_trigger_accepts_internal_semicolons():
    connection = _connection()
    toolkit = DatabaseToolkit(
        connection, include_write=True, allow_dangerous=True
    )
    try:
        created = toolkit.execute_write(
            "CREATE TRIGGER increment_new_user_score AFTER INSERT ON users "
            "BEGIN "
            "UPDATE users SET score = score + 1 WHERE id = NEW.id; "
            "UPDATE users SET score = score + 1 WHERE id = NEW.id; "
            "END"
        )
        assert created["ok"] is True

        inserted = toolkit.execute_write(
            "INSERT INTO users (name, score) VALUES (?, ?)", ["Margaret", 40]
        )
        assert inserted["ok"] is True
        assert connection.execute(
            "SELECT score FROM users WHERE name = ?", ("Margaret",)
        ).fetchone() == (42,)

        appended_statement = toolkit.execute_write(
            "CREATE TRIGGER invalid_extra AFTER INSERT ON users "
            "BEGIN UPDATE users SET score = score; END; SELECT 1"
        )
        assert appended_statement["error"]["type"] == "multiple_statements"
    finally:
        connection.close()


def test_default_write_scope_still_allows_insert_update_and_delete():
    connection = _connection()
    toolkit = DatabaseToolkit(connection, include_write=True)
    try:
        inserted = toolkit.execute_write(
            "INSERT INTO users (name, score) VALUES (?, ?)", ["Margaret", 40]
        )
        assert inserted["ok"] is True
        assert inserted["rows_affected"] == 1

        updated = toolkit.execute_write(
            "UPDATE users SET score = ? WHERE name = ?", [41, "Margaret"]
        )
        assert updated["ok"] is True
        assert updated["rows_affected"] == 1

        deleted = toolkit.execute_write("DELETE FROM users WHERE name = ?", ["Margaret"])
        assert deleted["ok"] is True
        assert deleted["rows_affected"] == 1
    finally:
        connection.close()


def test_external_connection_keeps_caller_transaction_boundary(tmp_path):
    database = tmp_path / "external.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    connection.commit()
    toolkit = DatabaseToolkit(connection, include_write=True)
    try:
        result = toolkit.execute_write("INSERT INTO items (name) VALUES (?)", ["pending"])
        assert result["ok"] is True
        assert result["committed"] is False
        assert connection.in_transaction is True
        with sqlite3.connect(database) as observer:
            assert observer.execute("SELECT name FROM items").fetchall() == []

        failed = toolkit.execute_write(
            "INSERT INTO items (name) VALUES (?)", ["pending"]
        )
        assert failed["error"]["type"] == "constraint_violation"
        assert connection.in_transaction is True
        assert connection.execute("SELECT name FROM items").fetchall() == [("pending",)]

        connection.rollback()
        assert connection.execute("SELECT name FROM items").fetchall() == []
    finally:
        connection.close()


def test_write_tool_can_be_marked_for_harness_approval():
    connection = _connection()
    try:
        tools = DatabaseToolkit(
            connection,
            include_write=True,
            require_write_approval=True,
        ).get_tools()
        write_tool = next(tool for tool in tools if tool.name == "execute_write")
        assert write_tool.metadata["database_operation"] == "write"
        assert write_tool.metadata["harness_approval"] is True
    finally:
        connection.close()


def test_tools_integrate_with_create_agent_and_async_invocation(persistent_defaults):
    connection = _connection()
    toolkit = DatabaseToolkit(connection, max_rows=10)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call(
                    "execute_query",
                    {"sql": "SELECT name FROM users ORDER BY id LIMIT 1"},
                    "database-1",
                )
            ],
        ),
        AIMessage(content="Ada"),
    ]
    model = DatabaseToolModel(responses=responses)
    agent = create_agent(
        name="database-agent",
        instructions="Use database tools.",
        model=model,
        tools=toolkit.get_tools(),
    )
    try:
        result = agent.invoke("Who is first?")
        assert model.bound_names == ["list_tables", "get_schema", "execute_query"]
        observation = next(
            message for message in result["messages"] if isinstance(message, ToolMessage)
        )
        assert json.loads(observation.content)["rows"] == [{"name": "Ada"}]

        async def invoke_tool():
            query_tool = next(
                tool for tool in toolkit.get_tools() if tool.name == "execute_query"
            )
            return await query_tool.ainvoke({"sql": "SELECT COUNT(*) AS count FROM users"})

        async_result = asyncio.run(invoke_tool())
        assert async_result["ok"] is True
        assert async_result["rows"] == [{"count": 3}]
    finally:
        connection.close()


def test_async_agent_can_call_database_tool(persistent_defaults):
    connection = _connection()
    toolkit = DatabaseToolkit(connection)
    model = DatabaseToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "list_tables",
                        {},
                        "database-async-1",
                    )
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(
        name="async-database-agent",
        instructions="Inspect the database.",
        model=model,
        tools=toolkit.get_tools(),
    )
    try:
        result = asyncio.run(agent.ainvoke("List tables"))
        observation = next(
            message for message in result["messages"] if isinstance(message, ToolMessage)
        )
        assert json.loads(observation.content)["tables"] == ["users"]
    finally:
        connection.close()
