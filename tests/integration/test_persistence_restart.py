from conftest import PersistentSaverStub, PersistentStoreStub
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import Field

from agent_harness import (
    MemoryConfig,
    configure_default_persistence,
    create_agent,
    require_approval,
)


def tool_call(name, arguments, identifier):
    return {
        "name": name,
        "args": arguments,
        "id": identifier,
        "type": "tool_call",
    }


class ToolCallingFakeModel(FakeMessagesListChatModel):
    bound_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_names = [
            (
                tool.name
                if hasattr(tool, "name")
                else tool["function"]["name"]
                if isinstance(tool, dict)
                else tool.__name__
            )
            for tool in tools
        ]
        return self


def test_recreated_agent_reads_file_checkpoint_after_new_backend_instances(
    persistent_defaults,
):
    saver, store = persistent_defaults
    first = create_agent(
        name="restartable",
        instructions="answer",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="first")]),
    )
    assert first.invoke("one", session_id="restart-session").output == "first"

    restarted_saver = PersistentSaverStub(saver.path)
    restarted_store = PersistentStoreStub(store.path)
    configure_default_persistence(
        checkpointer=restarted_saver,
        store=restarted_store,
    )
    try:
        second = create_agent(
            name="restartable",
            instructions="answer",
            model=FakeMessagesListChatModel(responses=[AIMessage(content="second")]),
        )
        result = second.invoke("two", session_id="restart-session")
        assert result.output == "second"
        assert len(result.state["messages"]) >= 3
    finally:
        restarted_saver.close()
        restarted_store.close()
        configure_default_persistence(checkpointer=saver, store=store)


def test_hitl_interrupt_survives_agent_and_sqlite_checkpointer_restart(tmp_path):
    checkpoint_path = tmp_path / "hitl-checkpoints.sqlite"
    store_path = tmp_path / "hitl-store.sqlite"
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return f"acted:{value}"

    def build_agent(saver, store, responses):
        tool = require_approval(
            StructuredTool.from_function(
                sensitive, name="sensitive", description="Perform one sensitive action."
            )
        )
        return create_agent(
            name="restartable-hitl",
            instructions="Perform the requested action.",
            model=ToolCallingFakeModel(responses=responses),
            tools=[tool],
            checkpointer=saver,
            store=store,
        )

    first_saver = PersistentSaverStub(checkpoint_path)
    first_store = PersistentStoreStub(store_path)
    first = build_agent(
        first_saver,
        first_store,
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("sensitive", {"value": "once"}, "sensitive-call-1")
                ],
            )
        ],
    )
    paused = first.invoke("go", session_id="durable-hitl")
    assert paused.status == "paused"
    assert first.runtime.is_paused(session_id="durable-hitl")
    assert calls == []
    first_saver.close()
    first_store.close()

    restarted_saver = PersistentSaverStub(checkpoint_path)
    restarted_store = PersistentStoreStub(store_path)
    restarted = build_agent(
        restarted_saver, restarted_store, [AIMessage(content="finished")]
    )
    try:
        assert restarted.runtime.is_paused(session_id="durable-hitl")
        assert restarted.runtime.pending_interrupts(session_id="durable-hitl")[0][
            "tool"
        ] == "sensitive"

        completed = restarted.resume(
            session_id="durable-hitl", decision="approve"
        )
        assert completed.status == "completed"
        assert completed.output == "finished"
        assert calls == ["once"]
        assert not restarted.runtime.is_paused(session_id="durable-hitl")

        checkpoint_config, internal_thread_id, _ = restarted.runtime._config(
            None, "durable-hitl"
        )
        assert restarted_saver.get_tuple(checkpoint_config) is not None

        restarted.clear_session("durable-hitl")
        assert restarted_saver.get_tuple(checkpoint_config) is None
        assert restarted_saver.deleted[-1] == internal_thread_id
    finally:
        restarted_saver.close()
        restarted_store.close()


def test_long_term_memory_survives_store_and_agent_restart_with_isolation(tmp_path):
    checkpoint_path = tmp_path / "memory-checkpoints.sqlite"
    store_path = tmp_path / "long-term-memory.sqlite"

    first_saver = PersistentSaverStub(checkpoint_path)
    first_store = PersistentStoreStub(store_path)
    first = create_agent(
        name="memory-agent",
        instructions="Remember user preferences.",
        model=ToolCallingFakeModel(
            responses=[
                AIMessage(content="I will remember that."),
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "Memory",
                            {"content": "User prefers concise reports"},
                            "memory-write-1",
                        )
                    ],
                ),
            ]
        ),
        memory=MemoryConfig(instructions="Store explicit user preferences."),
        checkpointer=first_saver,
        store=first_store,
    )
    first.invoke(
        "I prefer concise reports", session_id="memory-write", memory_id="user-1"
    )
    assert first_store.search(("memory", "memory-agent", "user-1"))
    first_saver.close()
    first_store.close()

    restarted_saver = PersistentSaverStub(checkpoint_path)
    restarted_store = PersistentStoreStub(store_path)
    restarted = create_agent(
        name="memory-agent",
        instructions="Remember user preferences.",
        model=ToolCallingFakeModel(
            responses=[
                AIMessage(content="I will use the saved preference."),
                AIMessage(content=""),
            ]
        ),
        memory=MemoryConfig(instructions="Store explicit user preferences."),
        checkpointer=restarted_saver,
        store=restarted_store,
    )
    try:
        result = restarted.invoke(
            "How should you answer?",
            session_id="memory-read",
            memory_id="user-1",
        )
        assert result.state["long_term_memories"] == [
            {"content": "User prefers concise reports"}
        ]
        assert restarted.runtime.memory.load("memory-agent", "user-2", "reports") == []
        assert restarted.runtime.memory.load("other-agent", "user-1", "reports") == []
    finally:
        restarted_saver.close()
        restarted_store.close()
