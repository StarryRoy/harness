import asyncio
import time
from typing import TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import Field

from agent_harness import (
    AgentMiddleware,
    ContextPolicy,
    PlanExecuteStrategy,
    RuntimeConfig,
    Skill,
    SkillMetadata,
    SkillRegistry,
    create_agent,
    require_approval,
)
from agent_harness.enterprise import (
    GuardrailMiddleware,
    GuardrailResult,
    ModelFallbackMiddleware,
)
from agent_harness.middleware import (
    AgentExecution,
    CallLimitMiddleware,
    MiddlewarePipeline,
    ModelRequest,
    RetryMiddleware,
    TimeoutMiddleware,
    ToolRequest,
)

pytestmark = pytest.mark.usefixtures("persistent_defaults")

_MEANINGFUL_CONTEXT = "context " * 350
_SECOND_CONTEXT = "second " * 350


class RecordingModel(FakeMessagesListChatModel):
    seen: list[list[object]] = Field(default_factory=list)
    structured_seen: list[list[object]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    tool_choices: list[object] = Field(default_factory=list)
    structured_outputs: list[object] = Field(default_factory=list)
    structured_index: int = 0

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_names = [
            tool.get("function", {}).get("name", "")
            if isinstance(tool, dict)
            else tool.name
            for tool in tools
        ]
        self.tool_choices.append(tool_choice)
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(value, config=None):
            self.structured_seen.append(list(value))
            if self.structured_index >= len(self.structured_outputs):
                raise AssertionError("missing fake structured output")
            result = self.structured_outputs[self.structured_index]
            self.structured_index += 1
            return result

        return RunnableLambda(run)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RecordingDebug:
    def __init__(self):
        self.events = []

    def emit(self, event, **details):
        self.events.append((event, details))


class FallbackModel:
    def invoke(self, messages, config):
        return "fallback"


def tool_call(name, arguments, identifier):
    return {
        "name": name,
        "args": arguments,
        "id": identifier,
        "type": "tool_call",
    }


def test_token_threshold_triggers_langmem_summary():
    policy = ContextPolicy(
        summary_token_threshold=1_000,
        summary_keep_recent=1,
    )
    model = RecordingModel(
        responses=[
            AIMessage(content="turn one"),
            AIMessage(content="conversation summary"),
            AIMessage(content="turn two"),
        ]
    )
    agent = create_agent(
        name="summary-regression",
        instructions="Assist the user.",
        model=model,
        runtime_config=RuntimeConfig(context_policy=policy),
    )

    agent.invoke(_MEANINGFUL_CONTEXT, session_id="session")
    result = agent.invoke(_SECOND_CONTEXT, session_id="session")

    assert len(model.seen) == 3
    assert result["messages"][-1].content == "turn two"
    assert any(
        isinstance(message, SystemMessage) and "conversation summary" in message.content
        for message in result["summarized_messages"]
    )


def test_token_threshold_triggers_async_langmem_summary():
    policy = ContextPolicy(
        summary_token_threshold=1_000,
        summary_keep_recent=1,
    )
    model = RecordingModel(
        responses=[
            AIMessage(content="turn one"),
            AIMessage(content="conversation summary"),
            AIMessage(content="turn two"),
        ]
    )
    agent = create_agent(
        name="async-summary-regression",
        instructions="Assist the user.",
        model=model,
        runtime_config=RuntimeConfig(context_policy=policy),
    )

    async def run():
        await agent.ainvoke(_MEANINGFUL_CONTEXT, session_id="session")
        return await agent.ainvoke(_SECOND_CONTEXT, session_id="session")

    result = asyncio.run(run())

    assert len(model.seen) == 3
    assert result["messages"][-1].content == "turn two"
    assert result["summary"] == "conversation summary"


def test_guardrail_emits_required_debug_event():
    debug = RecordingDebug()
    execution = AgentExecution("assistant", {}, debug=debug)
    middleware = GuardrailMiddleware(input=lambda value: GuardrailResult(action="pass"))

    middleware.before_agent(execution)

    assert debug.events == [
        ("GUARDRAIL", {"stage": "input", "action": "pass", "reason": None})
    ]


def test_model_fallback_emits_required_debug_event():
    debug = RecordingDebug()
    execution = AgentExecution("assistant", {}, debug=debug)
    pipeline = MiddlewarePipeline([ModelFallbackMiddleware([FallbackModel()])], debug)
    request = ModelRequest(execution, {}, [], {})

    def fail(_request):
        raise RuntimeError("primary unavailable")

    assert pipeline.model(request, fail) == "fallback"
    assert debug.events[-1] == (
        "MODEL FALLBACK",
        {
            "index": 1,
            "model": "FallbackModel",
            "primary_error": "primary unavailable",
        },
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_summary_context_keeps_later_tool_observation(async_mode):
    policy = ContextPolicy(
        summary_token_threshold=1_000,
        summary_keep_recent=1,
    )

    def observe(value: str) -> str:
        return f"fresh-observation:{value}"

    model = RecordingModel(
        responses=[
            AIMessage(content="turn one"),
            AIMessage(content="conversation summary"),
            AIMessage(
                content="",
                tool_calls=[tool_call("observe", {"value": "new"}, "observe-1")],
            ),
            AIMessage(content="used observation"),
        ]
    )
    agent = create_agent(
        name=f"summary-tool-{async_mode}",
        instructions="Use tool observations.",
        model=model,
        tools=[
            StructuredTool.from_function(
                observe, name="observe", description="Return an observation."
            )
        ],
        runtime_config=RuntimeConfig(context_policy=policy),
    )

    async def run_async():
        await agent.ainvoke(_MEANINGFUL_CONTEXT, session_id="session")
        return await agent.ainvoke(_SECOND_CONTEXT, session_id="session")

    if async_mode:
        result = asyncio.run(run_async())
    else:
        agent.invoke(_MEANINGFUL_CONTEXT, session_id="session")
        result = agent.invoke(_SECOND_CONTEXT, session_id="session")

    assert result["messages"][-1].content == "used observation"
    assert any(
        isinstance(message, ToolMessage) and message.content == "fresh-observation:new"
        for message in model.seen[-1]
    )


def test_plan_prepare_and_summary_run_before_planner_for_new_turn():
    policy = ContextPolicy(
        summary_token_threshold=1_000,
        summary_keep_recent=1,
    )
    model = RecordingModel(
        responses=[
            AIMessage(content="step-1"),
            AIMessage(content="final-1"),
            AIMessage(content="summary-1"),
            AIMessage(content="step-2"),
            AIMessage(content="final-2"),
            AIMessage(content="step-3"),
            AIMessage(content="final-3"),
        ],
        structured_outputs=[
            {"steps": [{"description": "one"}]},
            {"steps": [{"description": "two"}]},
            {"steps": [{"description": "three"}]},
        ],
    )
    agent = create_agent(
        name="planner-summary-order",
        instructions="Plan from current context.",
        model=model,
        strategy=PlanExecuteStrategy(max_steps=1),
        runtime_config=RuntimeConfig(context_policy=policy),
    )

    agent.invoke(_MEANINGFUL_CONTEXT, session_id="session")
    agent.invoke(_MEANINGFUL_CONTEXT, session_id="session")
    agent.invoke("latest third request", session_id="session")

    latest_plan_context = model.structured_seen[-1]
    assert any(
        isinstance(message, HumanMessage) and message.content == "latest third request"
        for message in latest_plan_context
    )
    assert any(
        isinstance(message, SystemMessage) and "summary-1" in message.content
        for message in latest_plan_context
    )


@pytest.mark.parametrize(
    ("decision", "expected_sensitive"),
    [
        ("approve", ["original"]),
        ("reject", []),
        ({"decision": "edit", "args": {"value": "edited"}}, ["edited"]),
    ],
)
def test_hitl_does_not_replay_an_earlier_tool(decision, expected_sensitive):
    ordinary_calls = []
    sensitive_calls = []

    def ordinary(value: str) -> str:
        ordinary_calls.append(value)
        return f"ordinary:{value}"

    def sensitive(value: str) -> str:
        sensitive_calls.append(value)
        return f"sensitive:{value}"

    sensitive_tool = require_approval(
        StructuredTool.from_function(
            sensitive, name="sensitive", description="Sensitive action."
        )
    )
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("ordinary", {"value": "once"}, "ordinary-1"),
                    tool_call("sensitive", {"value": "original"}, "sensitive-1"),
                ],
            ),
            AIMessage(content="finished"),
        ]
    )
    agent = create_agent(
        name=f"multi-hitl-{len(str(decision))}",
        instructions="Run both tools.",
        model=model,
        tools=[
            StructuredTool.from_function(
                ordinary, name="ordinary", description="Ordinary side effect."
            ),
            sensitive_tool,
        ],
    )

    agent.invoke("go", session_id="session")
    assert ordinary_calls == ["once"]
    assert sensitive_calls == []

    result = agent.resume(session_id="session", decision=decision)
    assert ordinary_calls == ["once"]
    assert sensitive_calls == expected_sensitive
    assert result["messages"][-1].content == "finished"


@pytest.mark.parametrize("planning", [False, True])
def test_main_resume_continues_sensitive_subagent(planning):
    sensitive_calls = []
    subagent_name = f"sensitive_sub_{planning}"

    def sensitive(value: str) -> str:
        sensitive_calls.append(value)
        return f"approved:{value}"

    child_model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("sensitive", {"value": "child"}, "child-1")],
            ),
            AIMessage(content="child complete"),
        ]
    )
    child = create_agent(
        name=subagent_name,
        description="Sensitive specialist.",
        instructions="Perform the delegated task.",
        model=child_model,
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive child action."
                )
            )
        ],
    )
    if planning:
        main_responses = [
            AIMessage(
                content="",
                tool_calls=[tool_call(subagent_name, {"task": "do it"}, "delegate-1")],
            ),
            AIMessage(content="step complete"),
            AIMessage(content="main complete"),
        ]
        structured_outputs = [{"steps": [{"description": "delegate"}]}]
        strategy = PlanExecuteStrategy(max_steps=1)
    else:
        main_responses = [
            AIMessage(
                content="",
                tool_calls=[tool_call(subagent_name, {"task": "do it"}, "delegate-1")],
            ),
            AIMessage(content="main complete"),
        ]
        structured_outputs = []
        strategy = None
    main_model = RecordingModel(
        responses=main_responses,
        structured_outputs=structured_outputs,
    )
    main = create_agent(
        name=f"main-sub-hitl-{planning}",
        instructions="Delegate and finish.",
        model=main_model,
        subagents=[child],
        strategy=strategy,
    )

    main.invoke("goal", session_id="main-session")
    pending = main.runtime.pending_interrupts(session_id="main-session")
    assert pending and pending[0]["subagent"] == subagent_name

    result = main.resume(session_id="main-session", decision="approve")
    assert sensitive_calls == ["child"]
    assert result["messages"][-1].content == "main complete"
    if planning:
        assert result["plan"]["status"] == "completed"


@pytest.mark.parametrize(
    ("async_mode", "second_decision", "expected_second_calls"),
    [
        (False, "approve", ["B"]),
        (True, "approve", ["B"]),
        (False, "reject", []),
    ],
)
def test_subagent_continues_through_multiple_hitl_decisions(
    async_mode, second_decision, expected_second_calls
):
    first_calls = []
    second_calls = []
    child_name = f"sequential_child_{async_mode}_{second_decision}"

    def sensitive_a(value: str) -> str:
        first_calls.append(value)
        return f"A:{value}"

    def sensitive_b(value: str) -> str:
        second_calls.append(value)
        return f"B:{value}"

    child = create_agent(
        name=child_name,
        description="Runs two approved actions.",
        instructions="Run each required sensitive action in order.",
        model=RecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("sensitive_a", {"value": "A"}, "child-a")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[tool_call("sensitive_b", {"value": "B"}, "child-b")],
                ),
                AIMessage(content="child final"),
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive_a,
                    name="sensitive_a",
                    description="First sensitive action.",
                )
            ),
            require_approval(
                StructuredTool.from_function(
                    sensitive_b,
                    name="sensitive_b",
                    description="Second sensitive action.",
                )
            ),
        ],
    )
    main = create_agent(
        name=f"sequential_main_{async_mode}_{second_decision}",
        instructions="Delegate and return the final result.",
        model=RecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(child_name, {"task": "run both"}, "delegate")
                    ],
                ),
                AIMessage(content="main final"),
            ]
        ),
        subagents=[child],
    )

    async def run_async():
        await main.ainvoke("goal", session_id="main-session")
        assert first_calls == []
        await main.aresume(session_id="main-session", decision="approve")
        assert main.runtime.is_paused(session_id="main-session")
        assert first_calls == ["A"]
        assert second_calls == []
        return await main.aresume(session_id="main-session", decision=second_decision)

    if async_mode:
        result = asyncio.run(run_async())
    else:
        main.invoke("goal", session_id="main-session")
        assert first_calls == []
        main.resume(session_id="main-session", decision="approve")
        assert main.runtime.is_paused(session_id="main-session")
        assert first_calls == ["A"]
        assert second_calls == []
        result = main.resume(session_id="main-session", decision=second_decision)

    assert first_calls == ["A"]
    assert second_calls == expected_second_calls
    assert not main.runtime.is_paused(session_id="main-session")
    assert result["messages"][-1].content == "main final"
    assert any(
        isinstance(message, ToolMessage) and "child final" in message.content
        for message in result["messages"]
    )


def test_plan_execute_with_structured_output_runs_full_graph():
    expected = {"answer": "formatted plan result"}
    model = RecordingModel(
        responses=[AIMessage(content="step result")],
        structured_outputs=[
            {"steps": [{"description": "execute the only step"}]},
            expected,
        ],
    )
    agent = create_agent(
        name="plan-structured-combination",
        instructions="Plan, execute, synthesize, and format.",
        model=model,
        strategy=PlanExecuteStrategy(max_steps=1),
        response_format=dict,
    )

    result = agent.invoke("complete the structured task")

    assert result["plan"]["status"] == "completed"
    assert result["plan"]["steps"][0]["result"] == "step result"
    assert result["messages"][-1].content == "step result"
    assert result["structured_response"] == expected
    assert len(model.seen) == 1
    assert len(model.structured_seen) == 2


def test_react_structured_output_preserves_multiple_tool_rounds():
    expected = {"answer": "structured tool result"}
    response_tool = "agent_harness_structured_response"
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("lookup", {"value": "x"}, "lookup-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("lookup", {"value": "y"}, "lookup-2")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call(response_tool, expected, "response-1")],
            ),
        ],
    )

    def lookup(value: str) -> str:
        """Look up a value."""
        return f"found:{value}"

    agent = create_agent(
        name="react-structured-tool",
        instructions="Use the tool and return structured output.",
        model=model,
        tools=[StructuredTool.from_function(lookup)],
        response_format=dict,
    )

    result = agent.invoke("look it up")

    assert result["structured_response"] == expected
    assert len(model.seen) == 3
    assert model.structured_seen == []
    assert model.bound_tool_names == ["lookup", response_tool]
    assert model.tool_choices == ["any"]
    assert any(
        isinstance(message, ToolMessage) and message.content == "found:x"
        for message in model.seen[1]
    )
    assert any(
        isinstance(message, ToolMessage) and message.content == "found:y"
        for message in model.seen[2]
    )


def test_react_can_return_structured_output_on_first_call_with_tools():
    expected = {"answer": "no tool needed"}
    response_tool = "agent_harness_structured_response"
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call(response_tool, expected, "response-1")],
            )
        ]
    )

    def lookup(value: str) -> str:
        """Look up a value."""
        return f"found:{value}"

    agent = create_agent(
        name="react-structured-direct",
        instructions="Return structured output directly when no tool is needed.",
        model=model,
        tools=[StructuredTool.from_function(lookup)],
        response_format=dict,
    )

    result = agent.invoke("answer directly")

    assert result["structured_response"] == expected
    assert len(model.seen) == 1
    assert model.structured_seen == []


def test_async_react_structured_output_uses_terminal_tool():
    expected = {"answer": "async structured result"}
    response_tool = "agent_harness_structured_response"
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("lookup", {"value": "async"}, "lookup-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call(response_tool, expected, "response-1")],
            ),
        ]
    )

    def lookup(value: str) -> str:
        """Look up a value."""
        return f"found:{value}"

    agent = create_agent(
        name="async-react-structured-tool",
        instructions="Use tools and return structured output.",
        model=model,
        tools=[StructuredTool.from_function(lookup)],
        response_format=dict,
    )

    result = asyncio.run(agent.ainvoke("look it up asynchronously"))

    assert result["structured_response"] == expected
    assert len(model.seen) == 2
    assert model.structured_seen == []


class NamespaceMemory:
    def __init__(self):
        self.values = {}
        self.loads = []

    def load(self, agent_name, memory_id, query):
        self.loads.append((agent_name, memory_id, query))
        return list(self.values.get((agent_name, memory_id), []))

    def update(self, agent_name, memory_id, messages):
        if agent_name == "memory_sub_a":
            self.values[(agent_name, memory_id)] = [
                {"preference": "subagent-A-private"}
            ]


class RecordingMemory:
    def __init__(self):
        self.updates = []

    def load(self, agent_name, memory_id, query):
        return []

    def update(self, agent_name, memory_id, messages):
        self.updates.append(list(messages))

    async def aupdate(self, agent_name, memory_id, messages):
        self.updates.append(list(messages))


@pytest.mark.parametrize("async_mode", [False, True])
def test_long_term_memory_only_processes_messages_from_current_turn(async_mode):
    model = RecordingModel(
        responses=[AIMessage(content="first reply"), AIMessage(content="second reply")]
    )
    agent = create_agent(
        name=f"incremental-memory-{async_mode}",
        instructions="Remember without reprocessing history.",
        model=model,
        memory=True,
    )
    memory = RecordingMemory()
    agent.runtime.memory = memory

    if async_mode:
        async def run():
            await agent.ainvoke("first request", session_id="session", memory_id="user")
            await agent.ainvoke("second request", session_id="session", memory_id="user")

        asyncio.run(run())
    else:
        agent.invoke("first request", session_id="session", memory_id="user")
        agent.invoke("second request", session_id="session", memory_id="user")

    assert [
        [message.content for message in messages] for messages in memory.updates
    ] == [
        ["first request", "first reply"],
        ["second request", "second reply"],
    ]


def test_subagent_memory_identity_propagates_and_agent_names_isolate_it():
    memory = NamespaceMemory()
    sub_a_model = RecordingModel(
        responses=[AIMessage(content="A saved"), AIMessage(content="A reused")]
    )
    sub_b_model = RecordingModel(responses=[AIMessage(content="B isolated")])
    sub_a = create_agent(
        name="memory_sub_a",
        description="A",
        instructions="A memory",
        model=sub_a_model,
        memory=True,
    )
    sub_b = create_agent(
        name="memory_sub_b",
        description="B",
        instructions="B memory",
        model=sub_b_model,
        memory=True,
    )
    sub_a.runtime.memory = memory
    sub_b.runtime.memory = memory
    main_model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("memory_sub_a", {"task": "save"}, "a-1")],
            ),
            AIMessage(content="main-1"),
            AIMessage(
                content="",
                tool_calls=[tool_call("memory_sub_a", {"task": "reuse"}, "a-2")],
            ),
            AIMessage(content="main-2"),
            AIMessage(
                content="",
                tool_calls=[tool_call("memory_sub_b", {"task": "check"}, "b-1")],
            ),
            AIMessage(content="main-3"),
        ]
    )
    main = create_agent(
        name="memory-main",
        instructions="Delegate.",
        model=main_model,
        subagents=[sub_a, sub_b],
        memory=True,
    )
    main.runtime.memory = memory

    main.invoke("first", session_id="session-A", memory_id="user-1")
    main.invoke("second", session_id="session-B", memory_id="user-1")
    main.invoke("third", session_id="session-C", memory_id="user-1")

    assert any(
        isinstance(message, SystemMessage) and "subagent-A-private" in message.content
        for message in sub_a_model.seen[1]
    )
    assert all(
        "subagent-A-private" not in str(message.content)
        for message in sub_b_model.seen[0]
        if isinstance(message, SystemMessage)
    )
    assert ("memory_sub_a", "user-1", "reuse") in memory.loads
    assert ("memory_sub_b", "user-1", "check") in memory.loads


def test_retry_attempts_are_each_counted_by_call_limit():
    execution = AgentExecution("assistant", {})
    pipeline = MiddlewarePipeline(
        [RetryMiddleware(max_attempts=3), CallLimitMiddleware(max_calls=2)],
        RecordingDebug(),
    )
    attempts = []

    def fail(request):
        attempts.append(request)
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="call limit=2"):
        pipeline.model(ModelRequest(execution, {}, [], {}), fail)

    assert len(attempts) == 2
    assert execution.metadata["middleware_calls"] == 3


def test_tool_retry_attempts_are_each_counted_by_call_limit():
    execution = AgentExecution("assistant", {})
    pipeline = MiddlewarePipeline(
        [
            RetryMiddleware(max_attempts=3, retry_tools=True),
            CallLimitMiddleware(max_calls=2),
        ],
        RecordingDebug(),
    )
    attempts = []
    tool = type("Tool", (), {"name": "retried_tool"})()

    def fail(request):
        attempts.append(request)
        raise RuntimeError("tool provider failed")

    with pytest.raises(RuntimeError, match="call limit=2"):
        pipeline.tool(ToolRequest(execution, tool, {}, {}, "call-1"), fail)

    assert len(attempts) == 2
    assert execution.metadata["middleware_calls"] == 3


@pytest.mark.parametrize("async_mode", [False, True])
def test_tool_middleware_receives_state_and_preserves_command(async_mode):
    class BusinessState(TypedDict, total=False):
        marker: str

    class StateCommandMiddleware(AgentMiddleware):
        def __init__(self):
            self.seen = []

        def _command(self, request):
            self.seen.append(request.state)
            assert request.state["marker"] in {"before", "after"}
            assert request.state["messages"][-1].tool_calls
            return Command(update={"marker": "after"})

        def wrap_tool_call(self, request, call_next):
            return self._command(request)

        async def awrap_tool_call(self, request, call_next):
            return self._command(request)

    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("lookup", {"value": "x"}, "lookup-1"),
                    tool_call("lookup", {"value": "y"}, "lookup-2"),
                ],
            ),
            AIMessage(content="done"),
        ]
    )

    def lookup(value: str) -> str:
        """Look up a value."""
        return value

    middleware = StateCommandMiddleware()
    agent = create_agent(
        name=f"command-{async_mode}",
        instructions="Use the tool.",
        model=model,
        tools=[StructuredTool.from_function(lookup)],
        state_schema=BusinessState,
        middleware=[middleware],
        middleware_mode="replace",
    )
    input_state = {"messages": [HumanMessage(content="go")], "marker": "before"}
    result = (
        asyncio.run(agent.ainvoke(input_state))
        if async_mode
        else agent.invoke(input_state)
    )

    assert result["marker"] == "after"
    assert result["messages"][-1].content == "done"
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["pending_tool_calls"] == []
    assert result["tool_call_index"] == 0
    assert len(middleware.seen) == 2
    assert middleware.seen[1]["pending_tool_calls"]
    assert middleware.seen[1]["tool_call_index"] == 1


def test_fallback_reenters_downstream_model_middleware():
    fallback = FallbackModel()
    execution = AgentExecution("assistant", {})
    seen_models = []

    class ObserveModel(AgentMiddleware):
        def wrap_model_call(self, request, call_next):
            seen_models.append(request.model_override)
            return call_next(request)

    pipeline = MiddlewarePipeline(
        [
            ModelFallbackMiddleware([fallback]),
            RetryMiddleware(max_attempts=1),
            CallLimitMiddleware(max_calls=4),
            ObserveModel(),
        ],
        RecordingDebug(),
    )

    def fail(_request):
        raise RuntimeError("primary failed")

    assert pipeline.model(ModelRequest(execution, {}, [], {}), fail) == "fallback"
    assert seen_models == [None, fallback]
    assert execution.metadata["middleware_calls"] == 2


def test_async_fallback_is_still_constrained_by_timeout():
    class SlowFallback:
        async def ainvoke(self, messages, config):
            await asyncio.sleep(0.05)
            return "late"

    execution = AgentExecution("assistant", {})
    pipeline = MiddlewarePipeline(
        [
            ModelFallbackMiddleware([SlowFallback()]),
            RetryMiddleware(max_attempts=1),
            CallLimitMiddleware(max_calls=4),
            TimeoutMiddleware(timeout_seconds=0.001),
        ],
        RecordingDebug(),
    )

    async def fail(_request):
        raise RuntimeError("primary failed")

    async def run():
        with pytest.raises(TimeoutError, match="Model call timed out"):
            await pipeline.amodel(ModelRequest(execution, {}, [], {}), fail)

    asyncio.run(run())
    assert execution.metadata["middleware_calls"] == 2


def test_sync_timeout_runs_inline_once_without_background_side_effect():
    calls = []
    tool = type("Tool", (), {"name": "side_effect"})()
    pipeline = MiddlewarePipeline(
        [TimeoutMiddleware(timeout_seconds=0.001)], RecordingDebug()
    )

    def invoke(_request):
        calls.append("called")
        time.sleep(0.01)
        return "done"

    result = pipeline.tool(
        ToolRequest(AgentExecution("assistant", {}), tool, {}, {}, "call-1"),
        invoke,
    )
    assert result == "done"
    assert calls == ["called"]


def test_skill_registry_mixes_unversioned_and_versioned_entries():
    registry = SkillRegistry()
    unversioned = Skill(SkillMetadata("analysis", "default"), "default")
    version_one = Skill(SkillMetadata("analysis", "one", version="1.0"), "version one")
    version_two = Skill(SkillMetadata("analysis", "two", version="2.0"), "version two")
    for skill in (unversioned, version_one, version_two):
        registry.register(skill)

    assert registry.get("analysis") is version_two
    assert registry.get("analysis@1.0") is version_one
    assert registry.get("analysis@2.0") is version_two
