import asyncio

import pytest

from agent_harness.enterprise import ModelFallbackMiddleware
from agent_harness.middleware import (
    AgentExecution,
    MiddlewarePipeline,
    ModelRequest,
    RetryMiddleware,
)


class Debug:
    def emit(self, *args, **kwargs):
        pass


class Runnable:
    def __init__(self, events, label):
        self.events = events
        self.label = label

    def invoke(self, messages, config):
        self.events.append(self.label)
        return self.label

    async def ainvoke(self, messages, config):
        self.events.append(self.label)
        return self.label


class FallbackModel(Runnable):
    def bind_tools(self, tools):
        return Runnable(self.events, "fallback-tools")

    def with_structured_output(self, schema):
        return Runnable(self.events, "fallback-structured")


def request(*, tools=(), response_format=None, purpose="agent"):
    return ModelRequest(
        execution=AgentExecution("assistant", {}),
        state={},
        messages=[],
        config={},
        purpose=purpose,
        tools=tools,
        response_format=response_format,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "fallback"),
        ({"tools": (object(),)}, "fallback-tools"),
        ({"response_format": dict}, "fallback-structured"),
        ({"response_format": dict, "purpose": "planner"}, "fallback-structured"),
    ],
)
def test_primary_retries_before_capable_fallback(kwargs, expected):
    events = []
    fallback = FallbackModel(events, "fallback")
    pipeline = MiddlewarePipeline(
        [ModelFallbackMiddleware([fallback]), RetryMiddleware(max_attempts=2)], Debug()
    )

    def primary(req):
        events.append("primary")
        raise RuntimeError("temporary")

    assert pipeline.model(request(**kwargs), primary) == expected
    assert events == ["primary", "primary", expected]


def test_async_primary_retries_before_fallback():
    events = []
    pipeline = MiddlewarePipeline(
        [
            ModelFallbackMiddleware([FallbackModel(events, "fallback")]),
            RetryMiddleware(2),
        ],
        Debug(),
    )

    async def primary(req):
        events.append("primary")
        raise RuntimeError("temporary")

    assert asyncio.run(pipeline.amodel(request(), primary)) == "fallback"
    assert events == ["primary", "primary", "fallback"]
