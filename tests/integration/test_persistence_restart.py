from conftest import PersistentSaverStub, PersistentStoreStub
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_harness import configure_default_persistence, create_agent


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
