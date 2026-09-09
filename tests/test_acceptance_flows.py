import asyncio
import sys
from typing import TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import Field

from agent_harness import (
    Agent,
    AgentMiddleware,
    PlanExecuteStrategy,
    Skill,
    SkillError,
    SkillLoader,
    SkillMetadata,
    SkillRegistry,
    SkillScriptRunner,
    create_agent,
    load_mcp_tools,
    require_approval,
)

pytestmark = pytest.mark.usefixtures("persistent_defaults")


class HarnessFakeModel(FakeMessagesListChatModel):
    bound_names: list[str] = Field(default_factory=list)
    seen: list[list[object]] = Field(default_factory=list)
    structured_outputs: list[object] = Field(default_factory=list)
    structured_index: int = 0

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_names = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(_value, config=None):
            if self.structured_index >= len(self.structured_outputs):
                raise AssertionError("missing fake structured output")
            result = self.structured_outputs[self.structured_index]
            self.structured_index += 1
            return result

        return RunnableLambda(run)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name, arguments, identifier):
    return {
        "name": name,
        "args": arguments,
        "id": identifier,
        "type": "tool_call",
    }


def value_tool(name):
    def run(value: str) -> str:
        return f"{name}:{value}"

    return StructuredTool.from_function(run, name=name, description=f"Run {name}.")


def test_phase1_react_tool_loops_and_progressive_skill():
    plain_model = HarnessFakeModel(responses=[AIMessage(content="plain-final")])
    plain = create_agent(name="plain", instructions="plain", model=plain_model)
    assert isinstance(plain, Agent)
    assert plain.invoke("hello")["messages"][-1].content == "plain-final"

    single_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("one", {"value": "x"}, "one-1")],
            ),
            AIMessage(content="single-final"),
        ]
    )
    single = create_agent(
        name="single",
        instructions="single",
        model=single_model,
        tools=[value_tool("one")],
    )
    single_result = single.invoke("go")
    assert single_result["messages"][-1].content == "single-final"
    assert any(
        isinstance(message, ToolMessage) and message.content == "one:x"
        for message in single_result["messages"]
    )

    multi_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("one", {"value": "x"}, "multi-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("two", {"value": "y"}, "multi-2")],
            ),
            AIMessage(content="multi-final"),
        ]
    )
    multi = create_agent(
        name="multi",
        instructions="multi",
        model=multi_model,
        tools=[value_tool("one"), value_tool("two")],
    )
    multi_result = multi.invoke("go")
    assert multi_result["messages"][-1].content == "multi-final"
    assert (
        sum(isinstance(message, ToolMessage) for message in multi_result["messages"])
        == 2
    )

    skill = Skill(
        SkillMetadata(
            name="expert",
            description="Expert method",
            required_tools=("one",),
        ),
        "PRIVATE SKILL INSTRUCTIONS",
    )
    skill_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("load_skill", {"name": "expert"}, "skill-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("one", {"value": "z"}, "skill-2")],
            ),
            AIMessage(content="skill-final"),
        ]
    )
    skilled = create_agent(
        name="skilled",
        instructions="skilled",
        model=skill_model,
        tools=[value_tool("one")],
        skills=[skill],
    )
    skill_result = skilled.invoke("expert task")
    first_system = next(
        message for message in skill_model.seen[0] if isinstance(message, SystemMessage)
    )
    second_system = next(
        message for message in skill_model.seen[1] if isinstance(message, SystemMessage)
    )
    assert "PRIVATE SKILL INSTRUCTIONS" not in first_system.content
    assert "expert: Expert method" in first_system.content
    assert "PRIVATE SKILL INSTRUCTIONS" in second_system.content
    assert skill_result["messages"][-1].content == "skill-final"


def test_phase1_sync_async_and_stream_apis():
    async def run_async():
        async_agent = create_agent(
            name="async",
            instructions="async",
            model=HarnessFakeModel(responses=[AIMessage(content="async-final")]),
        )
        async_result = await async_agent.ainvoke("hello")

        async_stream_agent = create_agent(
            name="async-stream",
            instructions="async stream",
            model=HarnessFakeModel(responses=[AIMessage(content="astream-final")]),
        )
        chunks = [
            chunk
            async for chunk in async_stream_agent.astream(
                "hello", session_id="async-stream-session"
            )
        ]
        return async_result, chunks

    result, async_chunks = asyncio.run(run_async())
    assert result["messages"][-1].content == "async-final"
    assert async_chunks

    stream_agent = create_agent(
        name="stream",
        instructions="stream",
        model=HarnessFakeModel(responses=[AIMessage(content="stream-final")]),
    )
    assert list(stream_agent.stream("hello", session_id="stream-session"))


def test_phase2_subagents_sessions_and_skill_isolation():
    private_skill = Skill(
        SkillMetadata("sub-skill", "SubAgent-only skill"),
        "PRIVATE SUBAGENT SKILL",
    )
    sub_a_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("load_skill", {"name": "sub-skill"}, "sub-skill-1")
                ],
            ),
            AIMessage(content="A-result"),
        ]
    )
    sub_b_model = HarnessFakeModel(responses=[AIMessage(content="B-result")])
    sub_a = create_agent(
        name="sub_a",
        description="SubAgent A",
        instructions="A",
        model=sub_a_model,
        skills=[private_skill],
    )
    sub_b = create_agent(
        name="sub_b",
        description="SubAgent B",
        instructions="B",
        model=sub_b_model,
    )
    main_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("sub_a", {"task": "task-A"}, "main-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("sub_b", {"task": "task-B"}, "main-2")],
            ),
            AIMessage(content="main-final"),
        ]
    )
    main = create_agent(
        name="main",
        instructions="main",
        model=main_model,
        subagents=[sub_a, sub_b],
    )

    result = main.invoke("goal")
    observations = [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert result["messages"][-1].content == "main-final"
    assert "A-result" in observations[0]
    assert "B-result" in observations[1]
    assert "PRIVATE SUBAGENT SKILL" not in observations[0]
    assert result["loaded_skills"] == []

    session_model = HarnessFakeModel(
        responses=[
            AIMessage(content="turn-1"),
            AIMessage(content="turn-2"),
            AIMessage(content="other-session"),
        ]
    )
    session_agent = create_agent(
        name="session-agent",
        instructions="session",
        model=session_model,
    )
    session_agent.invoke("A1", session_id="A")
    same_session = session_agent.invoke("A2", session_id="A")
    other_session = session_agent.invoke("B1", session_id="B")
    assert (
        sum(isinstance(message, HumanMessage) for message in same_session["messages"])
        == 2
    )
    assert (
        sum(isinstance(message, HumanMessage) for message in other_session["messages"])
        == 1
    )


def test_phase2_middleware_order_and_business_state():
    events = []

    class OrderedMiddleware(AgentMiddleware):
        def __init__(self, name):
            self.name = name

        def before_agent(self, execution):
            events.append(f"before-agent-{self.name}")

        def before_model(self, request):
            events.append(f"before-model-{self.name}")

        def wrap_model_call(self, request, call_next):
            events.append(f"enter-{self.name}")
            result = call_next(request)
            events.append(f"exit-{self.name}")
            return result

        def after_model(self, request, response):
            events.append(f"after-model-{self.name}")
            return response

        def after_agent(self, execution, result):
            events.append(f"after-agent-{self.name}")
            return result

    agent = create_agent(
        name="middleware",
        instructions="middleware",
        model=HarnessFakeModel(responses=[AIMessage(content="ok")]),
        middleware=[OrderedMiddleware("1"), OrderedMiddleware("2")],
        middleware_mode="replace",
    )
    agent.invoke("go")
    assert events == [
        "before-agent-1",
        "before-agent-2",
        "before-model-1",
        "before-model-2",
        "enter-1",
        "enter-2",
        "exit-2",
        "exit-1",
        "after-model-2",
        "after-model-1",
        "after-agent-2",
        "after-agent-1",
    ]

    class BusinessState(TypedDict, total=False):
        project_id: str

    state_agent = create_agent(
        name="state",
        instructions="state",
        model=HarnessFakeModel(responses=[AIMessage(content="ok")]),
        state_schema=BusinessState,
    )
    result = state_agent.invoke(
        {"messages": [HumanMessage(content="go")], "project_id": "P-001"}
    )
    assert result["project_id"] == "P-001"

    class InvalidState(TypedDict, total=False):
        messages: list

    with pytest.raises(ValueError, match="cannot override Harness fields"):
        create_agent(
            name="invalid-state",
            instructions="state",
            model=HarnessFakeModel(responses=[AIMessage(content="unused")]),
            state_schema=InvalidState,
        )


def test_phase2_advanced_skills(tmp_path):
    common_dir = tmp_path / "common"
    current_dir = tmp_path / "analysis-v2"
    common_dir.mkdir()
    (current_dir / "references").mkdir(parents=True)
    (current_dir / "resources").mkdir()
    (current_dir / "scripts").mkdir()
    (common_dir / "SKILL.md").write_text(
        "---\nname: common\ndescription: Common rules\nversion: '1.0'\n---\nCommon.",
        encoding="utf-8",
    )
    (current_dir / "SKILL.md").write_text(
        "---\n"
        "name: analysis\n"
        "description: Analyze measurements\n"
        "version: '2.0'\n"
        "required_tools: [double]\n"
        "dependencies: [common@1.0]\n"
        "scripts: [calculate.py]\n"
        "---\n"
        "Current instructions.",
        encoding="utf-8",
    )
    (current_dir / "references" / "standard.md").write_text(
        "Reference text.", encoding="utf-8"
    )
    (current_dir / "resources" / "template.txt").write_text(
        "Template text.", encoding="utf-8"
    )
    (current_dir / "scripts" / "calculate.py").write_text(
        "import json\n"
        "import sys\n"
        "arguments = json.load(sys.stdin)\n"
        "print(json.dumps({'result': arguments['value'] * 2}))\n",
        encoding="utf-8",
    )

    loader = SkillLoader()
    common = loader.load(common_dir)
    current = loader.load(current_dir)
    registry = SkillRegistry()
    registry.register(common)
    registry.register(current)
    registry.validate()

    assert [
        skill.identifier for skill in registry.load_order("analysis", {"double"})
    ] == ["common@1.0", "analysis@2.0"]
    with pytest.raises(SkillError, match="double"):
        registry.load_order("analysis", set())
    assert current.read_reference("standard.md") == "Reference text."
    assert current.read_resource("template.txt")["content"] == "Template text."
    assert SkillScriptRunner().run(current, "calculate.py", {"value": 6}).result == {
        "result": 12
    }
    assert registry.summaries(query="analyze measurements", limit=1)[0]["name"] == (
        "analysis@2.0"
    )

    cyclic = SkillRegistry()
    cyclic.register(Skill(SkillMetadata("a", "a", dependencies=("b",)), "a"))
    cyclic.register(Skill(SkillMetadata("b", "b", dependencies=("a",)), "b"))
    with pytest.raises(SkillError, match="Circular skill dependency"):
        cyclic.validate()


def test_phase3_plan_replan_and_structured_output():
    plan_model = HarnessFakeModel(
        responses=[
            AIMessage(content="step-one"),
            AIMessage(content="step-two"),
            AIMessage(content="synth-final"),
        ],
        structured_outputs=[
            {"steps": [{"description": "first"}, {"description": "second"}]}
        ],
    )
    planner = create_agent(
        name="planner",
        instructions="plan",
        model=plan_model,
        strategy=PlanExecuteStrategy(max_steps=3, max_replans=1),
    )
    result = planner.invoke("complex goal")
    assert result["plan"]["status"] == "completed"
    assert [step["result"] for step in result["plan"]["steps"]] == [
        "step-one",
        "step-two",
    ]
    assert result["messages"][-1].content == "synth-final"

    replan_model = HarnessFakeModel(
        responses=[
            AIMessage(content=""),
            AIMessage(content="recovered"),
            AIMessage(content="replan-final"),
        ],
        structured_outputs=[
            {"steps": [{"description": "bad"}]},
            {"steps": [{"description": "recover"}]},
        ],
    )
    replanner = create_agent(
        name="replanner",
        instructions="plan",
        model=replan_model,
        strategy=PlanExecuteStrategy(max_steps=2, max_replans=1),
    )
    replanned = replanner.invoke("complex goal")
    assert replanned["plan"]["replan_count"] == 1
    assert replanned["plan"]["steps"][0]["description"] == "recover"

    structured_model = HarnessFakeModel(
        responses=[AIMessage(content="draft")],
        structured_outputs=[{"answer": "structured"}],
    )
    structured = create_agent(
        name="structured",
        instructions="structured",
        model=structured_model,
        response_format=dict,
    )
    assert structured.invoke("go")["structured_response"] == {"answer": "structured"}


@pytest.mark.parametrize(
    ("decision", "expected_value"),
    [
        ("approve", "original"),
        ({"decision": "edit", "args": {"value": "changed"}}, "changed"),
    ],
)
def test_phase3_hitl_approve_and_edit(decision, expected_value):
    called = []

    def sensitive(value: str) -> str:
        called.append(value)
        return f"acted:{value}"

    tool = require_approval(
        StructuredTool.from_function(
            sensitive, name="sensitive", description="Sensitive action."
        )
    )
    model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("sensitive", {"value": "original"}, "sensitive-1")
                ],
            ),
            AIMessage(content="final"),
        ]
    )
    agent = create_agent(
        name=f"hitl-{expected_value}",
        instructions="hitl",
        model=model,
        tools=[tool],
    )

    agent.invoke("go", session_id="session")
    result = agent.resume(session_id="session", decision=decision)

    assert called == [expected_value]
    assert result["messages"][-1].content == "final"


def test_phase3_hitl_reject_and_plan_state_recovery():
    called = []

    def sensitive(value: str) -> str:
        called.append(value)
        return value

    tool = require_approval(
        StructuredTool.from_function(
            sensitive, name="sensitive", description="Sensitive action."
        )
    )
    reject_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("sensitive", {"value": "x"}, "reject-1")],
            ),
            AIMessage(content="rejected-final"),
        ]
    )
    reject_agent = create_agent(
        name="hitl-reject",
        instructions="hitl",
        model=reject_model,
        tools=[tool],
    )
    reject_agent.invoke("go", session_id="reject")
    rejected = reject_agent.resume(session_id="reject", decision="reject")
    observations = [
        message.content
        for message in rejected["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert not called
    assert any("rejected by the user" in observation for observation in observations)

    plan_model = HarnessFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("sensitive", {"value": "plan"}, "plan-1")],
            ),
            AIMessage(content="step-done"),
            AIMessage(content="plan-final"),
        ],
        structured_outputs=[{"steps": [{"description": "sensitive step"}]}],
    )
    plan_agent = create_agent(
        name="hitl-plan",
        instructions="plan",
        model=plan_model,
        tools=[tool],
        strategy=PlanExecuteStrategy(),
    )
    paused = plan_agent.invoke("go", session_id="plan")
    resumed = plan_agent.resume(session_id="plan", decision="approve")
    assert paused["plan"]["current_step"] == 0
    assert resumed["plan"]["status"] == "completed"
    assert resumed["plan"]["steps"][0]["result"] == "step-done"


def test_phase3_long_term_memory_cross_session():
    model = HarnessFakeModel(
        responses=[
            AIMessage(content="remembered"),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "Memory",
                        {"content": "User prefers concise reports"},
                        "memory-1",
                    )
                ],
            ),
            AIMessage(content="used-memory"),
            AIMessage(content=""),
        ]
    )
    agent = create_agent(
        name="memory-agent",
        instructions="help",
        model=model,
        memory=True,
    )

    agent.invoke(
        "I prefer concise reports",
        session_id="session-A",
        memory_id="user-1",
    )
    result = agent.invoke(
        "How should you answer?",
        session_id="session-B",
        memory_id="user-1",
    )

    assert result["long_term_memories"] == [{"content": "User prefers concise reports"}]


def test_phase3_mcp_stdio_tool_enters_common_runtime():
    pytest.importorskip("langchain_mcp_adapters")
    pytest.importorskip("mcp")

    async def run():
        server_code = (
            "from mcp.server.fastmcp import FastMCP; "
            "m=FastMCP('verify'); ns={}; "
            "exec('def double(x: int) -> int:\\n"
            '    \\"Double a number.\\"\\n'
            "    return x * 2',ns); "
            "m.tool()(ns['double']); m.run()"
        )
        tools = await load_mcp_tools(
            {
                "verify": {
                    "command": sys.executable,
                    "args": ["-c", server_code],
                    "transport": "stdio",
                }
            }
        )
        model = HarnessFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[tool_call("double", {"x": 21}, "mcp-1")],
                ),
                AIMessage(content="mcp-final"),
            ]
        )
        agent = create_agent(
            name="mcp-agent",
            instructions="mcp",
            model=model,
            tools=tools,
        )
        return model, await agent.ainvoke("double")

    model, result = asyncio.run(run())
    observations = [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert model.bound_names == ["double"]
    assert any("42" in observation for observation in observations)
    assert result["messages"][-1].content == "mcp-final"
