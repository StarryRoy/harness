import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from agent_harness import (
    AgentResult,
    LexicalSkillSelector,
    PersistenceError,
    PlanExecuteStrategy,
    RuntimeConfig,
    SessionError,
    Skill,
    SkillLoader,
    SkillMetadata,
    SkillRegistry,
    SkillScriptRunner,
    configure_default_persistence,
    create_agent,
    require_approval,
)


class ResultModel(FakeMessagesListChatModel):
    seen: list[list[object]] = Field(default_factory=list)
    structured_seen: list[list[object]] = Field(default_factory=list)
    structured_outputs: list[object] = Field(default_factory=list)
    structured_index: int = 0

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(value, config=None):
            self.structured_seen.append(list(value))
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


def test_agent_requires_persistent_checkpointer_and_rejects_memory_backends(
    persistent_defaults,
):
    saver, _ = persistent_defaults
    configure_default_persistence(checkpointer=None, store=None)
    model = ResultModel(responses=[AIMessage(content="unused")])

    with pytest.raises(PersistenceError, match="persistent LangGraph checkpointer"):
        create_agent(name="missing-persistence", instructions="x", model=model)
    with pytest.raises(PersistenceError, match="In-memory LangGraph checkpointers"):
        create_agent(
            name="memory-checkpointer",
            instructions="x",
            model=model,
            checkpointer=InMemorySaver(),
        )
    with pytest.raises(PersistenceError, match="In-memory LangGraph checkpointers"):
        configure_default_persistence(checkpointer=InMemorySaver())

    explicit = create_agent(
        name="explicit-persistence",
        instructions="x",
        model=ResultModel(responses=[AIMessage(content="ok")]),
        checkpointer=saver,
    )
    assert explicit.invoke("go").output == "ok"


def test_memory_requires_persistent_store_and_rejects_in_memory_store(
    persistent_defaults,
):
    saver, persistent_store = persistent_defaults
    configure_default_persistence(checkpointer=saver, store=None)
    model = ResultModel(responses=[AIMessage(content="unused")])

    with pytest.raises(PersistenceError, match="persistent LangGraph Store"):
        create_agent(name="missing-store", instructions="x", model=model, memory=True)
    with pytest.raises(PersistenceError, match="In-memory LangGraph stores"):
        create_agent(
            name="memory-store",
            instructions="x",
            model=model,
            memory=True,
            store=InMemoryStore(),
        )
    explicit = create_agent(
        name="persistent-memory",
        instructions="x",
        model=ResultModel(responses=[AIMessage(content="unused")]),
        memory=True,
        checkpointer=saver,
        store=persistent_store,
    )
    assert explicit.runtime.memory is not None


def test_agent_result_auto_session_can_resume_hitl(persistent_defaults):
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return value

    model = ResultModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("sensitive", {"value": "ok"}, "call-1")],
            ),
            AIMessage(content="finished"),
        ]
    )
    agent = create_agent(
        name="auto-session-hitl",
        instructions="act",
        model=model,
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive action."
                )
            )
        ],
    )

    paused = agent.invoke("go")
    assert isinstance(paused, AgentResult)
    assert paused.status == "paused"
    assert paused.session_id.startswith("session-")
    assert paused.interrupts[0]["tool"] == "sensitive"
    assert "thread_id" not in paused.metadata

    completed = agent.resume(session_id=paused.session_id, decision="approve")
    assert completed.status == "completed"
    assert completed.session_id == paused.session_id
    assert completed.output == "finished"
    assert completed.interrupts == ()
    assert completed.state["messages"][-1].content == "finished"
    assert calls == ["ok"]


def test_async_agent_result_auto_session_can_resume_hitl(persistent_defaults):
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return value

    agent = create_agent(
        name="async-auto-session-hitl",
        instructions="act",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("sensitive", {"value": "async"}, "async-call")
                    ],
                ),
                AIMessage(content="async finished"),
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive action."
                )
            )
        ],
    )

    async def run():
        paused = await agent.ainvoke("go")
        completed = await agent.aresume(
            session_id=paused.session_id, decision="approve"
        )
        return paused, completed

    paused, completed = asyncio.run(run())
    assert isinstance(paused, AgentResult)
    assert paused.status == "paused"
    assert isinstance(completed, AgentResult)
    assert completed.status == "completed"
    assert completed.output == "async finished"
    assert completed.session_id == paused.session_id
    assert calls == ["async"]


def test_agent_result_prefers_structured_output(persistent_defaults):
    expected = {"answer": "structured"}
    agent = create_agent(
        name="result-structured",
        instructions="format",
        model=ResultModel(
            responses=[AIMessage(content="draft")], structured_outputs=[expected]
        ),
        response_format=dict,
    )

    result = agent.invoke("go", session_id="known-session")
    assert result.output == expected
    assert result.structured_output == expected
    assert result.session_id == "known-session"
    assert result.status == "completed"


def test_stream_requires_public_session_and_hitl_can_resume_and_clear(
    persistent_defaults,
):
    saver, _ = persistent_defaults
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return value

    agent = create_agent(
        name="stream-session",
        instructions="act",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("sensitive", {"value": "stream"}, "stream-call")
                    ],
                ),
                AIMessage(content="stream finished"),
                AIMessage(content="fresh session"),
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive action."
                )
            )
        ],
    )

    with pytest.raises(SessionError, match="explicit non-empty session_id"):
        agent.stream("would be hidden")
    assert list(saver.backend.list(None)) == []

    chunks = list(agent.stream("go", session_id="public-stream"))
    assert chunks
    assert agent.runtime.is_paused(session_id="public-stream")
    completed = agent.resume(session_id="public-stream", decision="approve")
    assert completed.status == "completed"
    assert completed.output == "stream finished"
    assert calls == ["stream"]

    agent.clear_session("public-stream")
    fresh = agent.invoke("again", session_id="public-stream")
    assert fresh.output == "fresh session"
    assert sum(isinstance(item, HumanMessage) for item in fresh.state["messages"]) == 1


def test_astream_requires_public_session_and_hitl_can_resume_and_clear(
    persistent_defaults,
):
    saver, _ = persistent_defaults
    calls = []

    def sensitive(value: str) -> str:
        calls.append(value)
        return value

    agent = create_agent(
        name="astream-session",
        instructions="act",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("sensitive", {"value": "astream"}, "astream-call")
                    ],
                ),
                AIMessage(content="astream finished"),
                AIMessage(content="fresh async session"),
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive action."
                )
            )
        ],
    )

    async def run():
        with pytest.raises(SessionError, match="explicit non-empty session_id"):
            agent.astream("would be hidden")
        assert list(saver.backend.list(None)) == []
        chunks = [
            chunk async for chunk in agent.astream("go", session_id="public-astream")
        ]
        assert chunks
        assert await agent.runtime.ais_paused(session_id="public-astream")
        completed = await agent.aresume(session_id="public-astream", decision="approve")
        await agent.aclear_session("public-astream")
        fresh = await agent.ainvoke("again", session_id="public-astream")
        return completed, fresh

    completed, fresh = asyncio.run(run())
    assert completed.status == "completed"
    assert completed.output == "astream finished"
    assert fresh.output == "fresh async session"
    assert sum(isinstance(item, HumanMessage) for item in fresh.state["messages"]) == 1
    assert calls == ["astream"]


def test_clear_session_sync_and_async_hide_thread_id(persistent_defaults):
    saver, _ = persistent_defaults
    sync_agent = create_agent(
        name="clear-sync",
        instructions="remember session",
        model=ResultModel(
            responses=[AIMessage(content="first"), AIMessage(content="fresh")]
        ),
    )
    sync_agent.invoke("before", session_id="public-sync")
    sync_agent.clear_session("public-sync")
    fresh = sync_agent.invoke("after", session_id="public-sync")
    assert sum(isinstance(item, HumanMessage) for item in fresh.state["messages"]) == 1

    async_agent = create_agent(
        name="clear-async",
        instructions="remember session",
        model=ResultModel(
            responses=[AIMessage(content="first"), AIMessage(content="fresh")]
        ),
    )

    async def run():
        await async_agent.ainvoke("before", session_id="public-async")
        await async_agent.aclear_session("public-async")
        return await async_agent.ainvoke("after", session_id="public-async")

    async_fresh = asyncio.run(run())
    assert (
        sum(isinstance(item, HumanMessage) for item in async_fresh.state["messages"])
        == 1
    )
    assert saver.deleted
    assert "public-sync" not in saver.deleted
    assert "public-async" not in saver.deleted


def test_clear_session_wraps_backend_failure(persistent_defaults):
    saver, _ = persistent_defaults
    agent = create_agent(
        name="clear-error",
        instructions="x",
        model=ResultModel(responses=[AIMessage(content="ok")]),
    )

    def fail(_thread_id):
        raise OSError("database unavailable")

    saver.delete_thread = fail
    with pytest.raises(PersistenceError) as raised:
        agent.clear_session("session")
    assert isinstance(raised.value.cause, OSError)
    with pytest.raises(PersistenceError):
        agent.clear_session("")


def test_clear_main_session_cascades_only_its_subagent_checkpoint(
    persistent_defaults,
):
    saver, _ = persistent_defaults
    child = create_agent(
        name="cleanup-child",
        instructions="finish delegated work",
        model=ResultModel(
            responses=[AIMessage(content="child A"), AIMessage(content="child B")]
        ),
    )
    main = create_agent(
        name="cleanup-main",
        instructions="delegate",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("cleanup-child", {"task": "A"}, "child-call-A")
                    ],
                ),
                AIMessage(content="main A"),
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("cleanup-child", {"task": "B"}, "child-call-B")
                    ],
                ),
                AIMessage(content="main B"),
            ]
        ),
        subagents=[child],
    )

    main.invoke("A", session_id="main-A")
    main.invoke("B", session_id="main-B")

    def persisted_threads():
        return {
            item.config["configurable"]["thread_id"]
            for item in saver.backend.list(None)
        }

    assert len(persisted_threads()) == 4
    main.clear_session("main-A")
    assert len(persisted_threads()) == 2
    main.clear_session("main-B")
    assert persisted_threads() == set()


def test_aclear_main_session_removes_paused_subagent_hitl_checkpoint(
    persistent_defaults,
):
    saver, _ = persistent_defaults

    def sensitive(value: str) -> str:
        return value

    child = create_agent(
        name="paused-cleanup-child",
        instructions="use the sensitive tool",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("sensitive", {"value": "x"}, "sensitive-call")
                    ],
                )
            ]
        ),
        tools=[
            require_approval(
                StructuredTool.from_function(
                    sensitive, name="sensitive", description="Sensitive action."
                )
            )
        ],
    )
    main = create_agent(
        name="paused-cleanup-main",
        instructions="delegate",
        model=ResultModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "paused-delegate",
                            {"task": "pause"},
                            "paused-child-call",
                        )
                    ],
                )
            ]
        ),
        tools=[child.as_tool(name="paused-delegate")],
    )

    async def run():
        paused = await main.ainvoke("go", session_id="paused-main")
        assert paused.status == "paused"
        threads = {
            item.config["configurable"]["thread_id"]
            for item in saver.backend.list(None)
        }
        assert len(threads) == 2
        await main.aclear_session("paused-main")

    asyncio.run(run())
    assert list(saver.backend.list(None)) == []


def test_skill_selector_is_replaceable_and_lexical_remains_default(
    persistent_defaults,
):
    skills = [
        Skill(SkillMetadata("alpha", "Alpha work"), "alpha private"),
        Skill(SkillMetadata("beta", "Beta work"), "beta private"),
    ]

    class LastSelector:
        def select(self, candidates, *, query=None, limit=None):
            return list(candidates[-1:])

    custom = create_agent(
        name="custom-selector",
        instructions="select",
        model=ResultModel(responses=[AIMessage(content="ok")]),
        skills=skills,
        skill_selector=LastSelector(),
    )
    result = custom.invoke("alpha")
    assert result.state["available_skills"] == [
        {"name": "beta", "description": "Beta work"}
    ]

    registry = SkillRegistry(selector=LexicalSkillSelector())
    for skill in skills:
        registry.register(skill)
    assert registry.summaries(query="alpha", limit=1)[0]["name"] == "alpha"


def test_skill_assets_and_script_environment_allowlist(tmp_path, monkeypatch):
    root = tmp_path / "safe-skill"
    (root / "assets").mkdir(parents=True)
    (root / "resources").mkdir()
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: safe\ndescription: Safe skill\nscripts: [env.py]\n---\nUse assets.",
        encoding="utf-8",
    )
    (root / "assets" / "template.txt").write_text("asset", encoding="utf-8")
    (root / "resources" / "legacy.txt").write_text("resource", encoding="utf-8")
    (root / "scripts" / "env.py").write_text(
        "import json, os\n"
        "print(json.dumps({'allowed': os.getenv('HARNESS_ALLOWED'), "
        "'secret': os.getenv('HARNESS_SECRET')}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESS_ALLOWED", "visible")
    monkeypatch.setenv("HARNESS_SECRET", "must-not-leak")

    skill = SkillLoader().load(root)
    assert skill.read_asset("template.txt")["content"] == "asset"
    assert skill.read_resource("legacy.txt")["content"] == "resource"
    result = SkillScriptRunner(env_allowlist=("HARNESS_ALLOWED",)).run(skill, "env.py")
    assert result.result == {"allowed": "visible", "secret": None}
    assert RuntimeConfig(
        script_env_allowlist=["HARNESS_ALLOWED"]
    ).script_env_allowlist == ("HARNESS_ALLOWED",)


def test_plan_failure_and_partial_statuses_are_preserved(persistent_defaults):
    def fail_tool() -> str:
        raise RuntimeError("tool failed")

    failed_model = ResultModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[tool_call("fail_tool", {}, "failed-call")],
            ),
            AIMessage(content="Error: unrecoverable"),
            AIMessage(content="failure synthesis"),
        ],
        structured_outputs=[{"steps": [{"description": "failing step"}]}],
    )
    failed_agent = create_agent(
        name="failed-plan",
        instructions="plan",
        model=failed_model,
        tools=[
            StructuredTool.from_function(
                fail_tool, name="fail_tool", description="Always fails."
            )
        ],
        strategy=PlanExecuteStrategy(max_steps=1, max_replans=0),
    )
    failed = failed_agent.invoke("fail")
    assert failed.status == "error"
    assert failed.state["plan"]["status"] == "failed"
    assert failed.state["plan"]["steps"][0]["status"] == "failed"

    partial_model = ResultModel(
        responses=[
            AIMessage(content="step one complete"),
            AIMessage(
                content="",
                tool_calls=[tool_call("fail_tool", {}, "partial-call")],
            ),
            AIMessage(content="Error: second step failed"),
            AIMessage(content="partial synthesis"),
        ],
        structured_outputs=[
            {
                "steps": [
                    {"description": "successful step"},
                    {"description": "failing step"},
                    {"description": "unreachable step"},
                ]
            }
        ],
    )
    partial_agent = create_agent(
        name="partial-plan",
        instructions="plan",
        model=partial_model,
        tools=[
            StructuredTool.from_function(
                fail_tool, name="fail_tool", description="Always fails."
            )
        ],
        strategy=PlanExecuteStrategy(max_steps=3, max_replans=0),
    )
    partial = partial_agent.invoke("partially finish")
    assert partial.state["plan"]["status"] == "partial"
    assert [step["status"] for step in partial.state["plan"]["steps"]] == [
        "completed",
        "failed",
        "skipped",
    ]
    assert partial.output == "partial synthesis"


def test_planner_and_replanner_controls_use_system_messages(persistent_defaults):
    model = ResultModel(
        responses=[
            AIMessage(content=""),
            AIMessage(content="recovered"),
            AIMessage(content="final"),
        ],
        structured_outputs=[
            {"steps": [{"description": "initial"}]},
            {"steps": [{"description": "recovered"}]},
        ],
    )
    agent = create_agent(
        name="planner-roles",
        instructions="plan",
        model=model,
        strategy=PlanExecuteStrategy(max_steps=1, max_replans=1),
    )
    agent.invoke("goal")

    assert any(
        isinstance(message, SystemMessage) and "Create a plan" in message.content
        for message in model.structured_seen[0]
    )
    assert any(
        isinstance(message, SystemMessage)
        and "Revise only the unfinished" in message.content
        for message in model.structured_seen[1]
    )
    assert not any(
        isinstance(message, AIMessage)
        and (
            "Create a plan" in str(message.content)
            or "Revise only the unfinished" in str(message.content)
        )
        for request in model.structured_seen
        for message in request
    )
