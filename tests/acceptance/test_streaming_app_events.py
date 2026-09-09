from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_harness import StreamEvent, create_agent


def test_application_stream_is_graph_agnostic(persistent_defaults):
    agent = create_agent(
        name="acceptance-stream",
        instructions="answer",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="hello")]),
    )

    events = list(agent.stream("hi", session_id="acceptance-stream"))

    assert all(isinstance(event, StreamEvent) for event in events)
    assert events[-1].event_type == "final"
    assert events[-1].data["output"] == "hello"
