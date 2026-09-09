import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_harness import RuntimeEvent, StreamEvent, create_agent


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_invoke_emits_traceable_events_and_metrics(persistent_defaults):
    recorder = Recorder()
    agent = create_agent(
        name="observed",
        instructions="answer",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
        event_sink=recorder,
    )

    result = agent.invoke("go", session_id="contract-session")

    assert result.output == "done"
    assert all(isinstance(event, RuntimeEvent) for event in recorder.events)
    assert {event.event_type for event in recorder.events} >= {
        "agent.start",
        "model.start",
        "model.end",
        "agent.end",
    }
    assert len({event.trace_id for event in recorder.events}) == 1
    model_start = next(e for e in recorder.events if e.event_type == "model.start")
    agent_start = next(e for e in recorder.events if e.event_type == "agent.start")
    assert model_start.parent_span_id == agent_start.span_id
    assert agent.metrics.model_calls == 1
    assert agent.metrics.agent_calls == 1


def test_stream_is_stable_and_raw_stream_remains_available(persistent_defaults):
    stable = create_agent(
        name="stable-stream",
        instructions="answer",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="hello")]),
    )
    events = list(stable.stream("go", session_id="stable"))

    assert all(isinstance(event, StreamEvent) for event in events)
    assert [event.event_type for event in events][-1] == "final"
    assert any(event.event_type == "text_delta" for event in events)
    assert events[-1].data["output"] == "hello"

    raw = create_agent(
        name="raw-stream",
        instructions="answer",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="raw")]),
    )
    chunks = list(raw.raw_stream("go", session_id="raw"))
    assert chunks
    assert not isinstance(chunks[0], StreamEvent)


def test_async_stream_matches_sync_contract(persistent_defaults):
    async def run():
        agent = create_agent(
            name="async-stable-stream",
            instructions="answer",
            model=FakeMessagesListChatModel(responses=[AIMessage(content="async")]),
        )
        return [event async for event in agent.astream("go", session_id="async-stable")]

    events = asyncio.run(run())
    assert all(isinstance(event, StreamEvent) for event in events)
    assert events[-1].event_type == "final"
    assert events[-1].data["output"] == "async"
