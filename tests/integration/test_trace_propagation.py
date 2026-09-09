from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent_harness import create_agent, require_approval


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def tool_call(name, arguments, identifier):
    return {
        "name": name,
        "args": arguments,
        "id": identifier,
        "type": "tool_call",
    }


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_subagent_inherits_trace_and_parent_span(persistent_defaults):
    recorder = Recorder()
    child = create_agent(
        name="trace-child",
        description="child",
        instructions="answer child",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="child done")]),
    )
    main = create_agent(
        name="trace-main",
        instructions="delegate",
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("trace-child", {"task": "work"}, "delegate-1")
                    ],
                ),
                AIMessage(content="main done"),
            ]
        ),
        subagents=[child],
        event_sink=recorder,
    )

    assert main.invoke("go", session_id="trace").output == "main done"

    assert len({event.trace_id for event in recorder.events}) == 1
    subagent_start = next(
        event for event in recorder.events if event.event_type == "subagent.start"
    )
    child_agent = next(
        event
        for event in recorder.events
        if event.event_type == "agent.start" and event.agent_name == "trace-child"
    )
    assert child_agent.parent_span_id == subagent_start.span_id
    assert any(
        event.event_type == "model.start" and event.agent_name == "trace-child"
        for event in recorder.events
    )


def test_hitl_resume_keeps_trace_and_does_not_double_count_pause(
    persistent_defaults,
):
    recorder = Recorder()
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return value

    agent = create_agent(
        name="trace-hitl",
        instructions="use tool",
        model=ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("sensitive", {"value": "ok"}, "call-1")],
                ),
                AIMessage(content="done"),
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="sensitive"
                )
            )
        ],
        event_sink=recorder,
    )

    paused = agent.invoke("go", session_id="hitl-trace")
    assert paused.status == "paused"
    completed = agent.resume(session_id="hitl-trace", decision="approve")

    assert completed.status == "completed"
    assert calls == ["ok"]
    assert len({event.trace_id for event in recorder.events}) == 1
    assert sum(event.event_type == "hitl.pause" for event in recorder.events) == 1
    assert {event.event_type for event in recorder.events} >= {
        "hitl.resume",
        "hitl.approve",
    }
