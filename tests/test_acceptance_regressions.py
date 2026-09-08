import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import Field

from agent_harness import ContextPolicy, RuntimeConfig, create_agent
from agent_harness.enterprise import (
    GuardrailMiddleware,
    GuardrailResult,
    ModelFallbackMiddleware,
)
from agent_harness.middleware import AgentExecution, MiddlewarePipeline, ModelRequest


class RecordingModel(FakeMessagesListChatModel):
    seen: list[list[object]] = Field(default_factory=list)

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


def test_message_threshold_triggers_langmem_summary():
    policy = ContextPolicy(
        summary_threshold=2,
        summary_token_threshold=100_000,
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

    agent.invoke("first", session_id="session")
    result = agent.invoke("second", session_id="session")

    assert len(model.seen) == 3
    assert result["messages"][-1].content == "turn two"
    assert any(
        isinstance(message, SystemMessage) and "conversation summary" in message.content
        for message in result["summarized_messages"]
    )
    assert result["summarized_messages"][-1].content == "second"


def test_message_threshold_triggers_async_langmem_summary():
    policy = ContextPolicy(
        summary_threshold=2,
        summary_token_threshold=100_000,
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
        await agent.ainvoke("first", session_id="session")
        return await agent.ainvoke("second", session_id="session")

    result = asyncio.run(run())

    assert len(model.seen) == 3
    assert result["messages"][-1].content == "turn two"
    assert result["summary"] == "conversation summary"
    assert result["summarized_messages"][-1].content == "second"


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
