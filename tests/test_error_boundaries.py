import asyncio

import pytest

from agent_harness.errors import MiddlewareError
from agent_harness.middleware import (
    AgentExecution,
    AgentMiddleware,
    MiddlewarePipeline,
    ModelRequest,
)


class Debug:
    def emit(self, *args, **kwargs):
        pass


class BrokenHook(AgentMiddleware):
    def __init__(self, hook):
        self.hook = hook

    def before_agent(self, execution):
        if self.hook == "before_agent":
            raise LookupError("broken agent hook")

    def before_model(self, request):
        if self.hook == "before_model":
            raise LookupError("broken model hook")

    def after_model(self, request, response):
        if self.hook == "after_model":
            raise LookupError("broken model hook")
        return response

    def after_agent(self, execution, result):
        if self.hook == "after_agent":
            raise LookupError("broken agent hook")
        return result


def execution():
    return AgentExecution("assistant", {})


def request(run):
    return ModelRequest(run, {}, [], {})


@pytest.mark.parametrize("hook", ["before_agent", "after_agent"])
def test_agent_hook_failures_have_stable_error_type(hook):
    pipeline = MiddlewarePipeline([BrokenHook(hook)], Debug())
    run = execution()

    with pytest.raises(MiddlewareError) as raised:
        if hook == "before_agent":
            pipeline.before_agent(run)
        else:
            pipeline.after_agent(run, {})

    assert isinstance(raised.value.cause, LookupError)
    assert hook in str(raised.value)


@pytest.mark.parametrize("hook", ["before_model", "after_model"])
def test_model_hook_failures_have_stable_error_type(hook):
    pipeline = MiddlewarePipeline([BrokenHook(hook)], Debug())
    model_request = request(execution())

    with pytest.raises(MiddlewareError) as raised:
        pipeline.model(model_request, lambda _: "answer")

    assert isinstance(raised.value.cause, LookupError)
    assert hook in str(raised.value)


@pytest.mark.parametrize("hook", ["before_agent", "after_agent"])
def test_async_agent_hook_failures_have_stable_error_type(hook):
    async def run():
        pipeline = MiddlewarePipeline([BrokenHook(hook)], Debug())
        agent_execution = execution()
        if hook == "before_agent":
            await pipeline.abefore_agent(agent_execution)
        else:
            await pipeline.aafter_agent(agent_execution, {})

    with pytest.raises(MiddlewareError) as raised:
        asyncio.run(run())

    assert isinstance(raised.value.cause, LookupError)


@pytest.mark.parametrize("hook", ["before_model", "after_model"])
def test_async_model_hook_failures_have_stable_error_type(hook):
    async def run():
        pipeline = MiddlewarePipeline([BrokenHook(hook)], Debug())
        await pipeline.amodel(request(execution()), lambda _: _answer())

    async def _answer():
        return "answer"

    with pytest.raises(MiddlewareError) as raised:
        asyncio.run(run())

    assert isinstance(raised.value.cause, LookupError)
