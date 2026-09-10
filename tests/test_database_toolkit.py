import sqlite3

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from agent_harness import DatabaseToolkit, create_agent, require_approval


class ToolBindingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _tools(toolkit):
    return {tool.name: tool for tool in toolkit.get_tools()}


def test_sqlite_database_operations_and_errors():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)"
    )
    connection.execute("INSERT INTO users (name) VALUES ('Ada')")
    connection.commit()

    readonly = _tools(DatabaseToolkit(connection))
    assert set(readonly) == {"list_tables", "get_schema", "execute_query"}
    assert readonly["list_tables"].invoke({}) == {"ok": True, "tables": ["users"]}

    schema = readonly["get_schema"].invoke({"table": "users"})
    assert schema["ok"] is True
    assert [column["name"] for column in schema["columns"]] == ["id", "name"]

    result = readonly["execute_query"].invoke(
        {"sql": "SELECT id, name FROM users ORDER BY id"}
    )
    assert result == {
        "ok": True,
        "columns": ["id", "name"],
        "rows": [{"id": 1, "name": "Ada"}],
        "row_count": 1,
    }
    error = readonly["execute_query"].invoke({"sql": "SELECT missing FROM users"})
    assert error["ok"] is False
    assert error["error"]["type"] == "sql_error"
    assert "missing" not in error["error"]["message"]

    writable = _tools(DatabaseToolkit(connection, include_write=True))
    inserted = writable["execute_write"].invoke(
        {"sql": "INSERT INTO users (name) VALUES ('Grace')"}
    )
    assert inserted["ok"] is True
    assert inserted["rows_affected"] == 1
    assert connection.execute("SELECT count(*) FROM users").fetchone()[0] == 2
    assert writable["execute_write"].invoke({"sql": "DROP TABLE users"}) == {
        "ok": False,
        "error": {
            "type": "unsafe_operation",
            "message": "execute_write only accepts INSERT, UPDATE, DELETE, or REPLACE.",
        },
    }


def test_query_cannot_hide_a_write_in_cte():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE items (value TEXT)")
    query = _tools(DatabaseToolkit(connection))["execute_query"]
    result = query.invoke(
        {"sql": "WITH item AS (SELECT 'safe' AS value) SELECT value FROM item"}
    )
    assert result["rows"] == [{"value": "safe"}]
    assert query.invoke({"sql": "DELETE FROM items"})["error"]["type"] == "read_only"


def test_tools_integrate_with_create_agent_and_hitl(persistent_defaults):
    toolkit = DatabaseToolkit(sqlite3.connect(":memory:"), include_write=True)
    tools = toolkit.get_tools()
    write = next(tool for tool in tools if tool.name == "execute_write")
    require_approval(write, message="Approve database mutation")

    model = ToolBindingModel(responses=[AIMessage(content="ready")])
    agent = create_agent(
        name="database-agent",
        instructions="Use database tools when needed.",
        model=model,
        tools=tools,
    )
    assert all(isinstance(tool, BaseTool) for tool in agent.definition.tools)
    assert {tool.name for tool in agent.definition.tools} == {
        "list_tables",
        "get_schema",
        "execute_query",
        "execute_write",
    }
    assert write.metadata["harness_approval"] is True
