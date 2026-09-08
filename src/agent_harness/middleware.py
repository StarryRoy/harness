"""Ordered middleware used by the agent, model, and tool execution paths."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .errors import AgentError, MiddlewareError


@dataclass(slots=True)
class AgentExecution:
    agent_name: str
    input: dict[str, Any]
    session_id: str | None = None
    business_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None


@dataclass(slots=True)
class ModelRequest:
    execution: AgentExecution
    state: Mapping[str, Any]
    messages: list[Any]
    config: Mapping[str, Any]
    purpose: str = "agent"
    tools: tuple[Any, ...] = ()
    response_format: Any | None = None

    def runnable_for(self, model: Any) -> Any:
        """Recreate the current call shape on another base chat model."""
        if self.response_format is not None:
            return model.with_structured_output(self.response_format)
        if self.tools:
            return model.bind_tools(list(self.tools))
        return model

    def invoke_with(self, model: Any) -> Any:
        return self.runnable_for(model).invoke(self.messages, self.config)

    async def ainvoke_with(self, model: Any) -> Any:
        return await self.runnable_for(model).ainvoke(self.messages, self.config)


@dataclass(slots=True)
class ToolRequest:
    execution: AgentExecution
    tool: Any
    arguments: Mapping[str, Any]
    config: Mapping[str, Any]
    tool_call_id: str


ModelHandler = Callable[[ModelRequest], Any]
AsyncModelHandler = Callable[[ModelRequest], Awaitable[Any]]
ToolHandler = Callable[[ToolRequest], Any]
AsyncToolHandler = Callable[[ToolRequest], Awaitable[Any]]


class AgentMiddleware:
    """No-op base class for the stable Phase 2 middleware contract.

    Before hooks run in registration order. Wrappers are nested in registration
    order (the first middleware is outermost). After hooks run in reverse order.
    """

    def before_agent(self, execution: AgentExecution) -> None:
        pass

    async def abefore_agent(self, execution: AgentExecution) -> None:
        self.before_agent(execution)

    def before_model(self, request: ModelRequest) -> None:
        pass

    async def abefore_model(self, request: ModelRequest) -> None:
        self.before_model(request)

    def wrap_model_call(self, request: ModelRequest, call_next: ModelHandler) -> Any:
        return call_next(request)

    async def awrap_model_call(
        self, request: ModelRequest, call_next: AsyncModelHandler
    ) -> Any:
        result = self.wrap_model_call(request, call_next)  # type: ignore[arg-type]
        return await result if inspect.isawaitable(result) else result

    def after_model(self, request: ModelRequest, response: Any) -> Any:
        return response

    async def aafter_model(self, request: ModelRequest, response: Any) -> Any:
        return self.after_model(request, response)

    def wrap_tool_call(self, request: ToolRequest, call_next: ToolHandler) -> Any:
        return call_next(request)

    async def awrap_tool_call(
        self, request: ToolRequest, call_next: AsyncToolHandler
    ) -> Any:
        result = self.wrap_tool_call(request, call_next)  # type: ignore[arg-type]
        return await result if inspect.isawaitable(result) else result

    def after_agent(self, execution: AgentExecution, result: Any) -> Any:
        return result

    async def aafter_agent(self, execution: AgentExecution, result: Any) -> Any:
        return self.after_agent(execution, result)


class RetryMiddleware(AgentMiddleware):
    """Retry model calls and, when explicitly enabled, tool calls."""

    def __init__(
        self,
        max_attempts: int = 2,
        delay_seconds: float = 0.0,
        *,
        retry_tools: bool = False,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.retry_tools = retry_tools
        self.exceptions = exceptions

    def _run(self, request: Any, call_next: Callable[[Any], Any]) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return call_next(request)
            except self.exceptions:
                if attempt == self.max_attempts:
                    raise
                if self.delay_seconds:
                    time.sleep(self.delay_seconds)
        raise AssertionError("unreachable")

    async def _arun(self, request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await call_next(request)
            except self.exceptions:
                if attempt == self.max_attempts:
                    raise
                if self.delay_seconds:
                    await asyncio.sleep(self.delay_seconds)
        raise AssertionError("unreachable")

    def wrap_model_call(self, request: ModelRequest, call_next: ModelHandler) -> Any:
        return self._run(request, call_next)

    async def awrap_model_call(
        self, request: ModelRequest, call_next: AsyncModelHandler
    ) -> Any:
        return await self._arun(request, call_next)

    def wrap_tool_call(self, request: ToolRequest, call_next: ToolHandler) -> Any:
        return self._run(request, call_next) if self.retry_tools else call_next(request)

    async def awrap_tool_call(
        self, request: ToolRequest, call_next: AsyncToolHandler
    ) -> Any:
        return await self._arun(request, call_next) if self.retry_tools else await call_next(request)


class TimeoutMiddleware(AgentMiddleware):
    """Apply cancellable timeouts to async model and tool calls.

    Sync calls run inline because Python cannot safely cancel an arbitrary
    running thread. This avoids returning a timeout while work with possible
    side effects continues in the background.
    """

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def wrap_model_call(self, request: ModelRequest, call_next: ModelHandler) -> Any:
        return call_next(request)

    async def awrap_model_call(
        self, request: ModelRequest, call_next: AsyncModelHandler
    ) -> Any:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Model call timed out after {self.timeout_seconds:g}s") from exc

    def wrap_tool_call(self, request: ToolRequest, call_next: ToolHandler) -> Any:
        return call_next(request)

    async def awrap_tool_call(
        self, request: ToolRequest, call_next: AsyncToolHandler
    ) -> Any:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Tool '{request.tool.name}' timed out after {self.timeout_seconds:g}s"
            ) from exc


class CallLimitMiddleware(AgentMiddleware):
    """Limit total model and tool calls within one agent execution scope."""

    def __init__(self, max_calls: int = 48) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls

    def _count(self, execution: AgentExecution) -> None:
        calls = int(execution.metadata.get("middleware_calls", 0)) + 1
        execution.metadata["middleware_calls"] = calls
        if calls > self.max_calls:
            raise RuntimeError(f"Agent exceeded middleware call limit={self.max_calls}")

    def wrap_model_call(self, request: ModelRequest, call_next: ModelHandler) -> Any:
        self._count(request.execution)
        return call_next(request)

    async def awrap_model_call(
        self, request: ModelRequest, call_next: AsyncModelHandler
    ) -> Any:
        self._count(request.execution)
        return await call_next(request)

    def wrap_tool_call(self, request: ToolRequest, call_next: ToolHandler) -> Any:
        self._count(request.execution)
        return call_next(request)

    async def awrap_tool_call(
        self, request: ToolRequest, call_next: AsyncToolHandler
    ) -> Any:
        self._count(request.execution)
        return await call_next(request)


_EXECUTION: contextvars.ContextVar[AgentExecution | None] = contextvars.ContextVar(
    "agent_harness_execution", default=None
)


class MiddlewarePipeline:
    def __init__(self, middleware: Sequence[AgentMiddleware], debug: Any) -> None:
        self.middleware = tuple(middleware)
        self.debug = debug

    @contextmanager
    def execution_scope(self, execution: AgentExecution) -> Iterator[None]:
        token = _EXECUTION.set(execution)
        try:
            yield
        finally:
            _EXECUTION.reset(token)

    @staticmethod
    def current_execution() -> AgentExecution:
        execution = _EXECUTION.get()
        if execution is None:
            raise RuntimeError("No active agent execution scope")
        return execution

    def before_agent(self, execution: AgentExecution) -> None:
        for item in self.middleware:
            self.debug.emit("MIDDLEWARE", hook="before_agent", name=type(item).__name__)
            try:
                item.before_agent(execution)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.before_agent failed", cause=exc
                ) from exc

    async def abefore_agent(self, execution: AgentExecution) -> None:
        for item in self.middleware:
            self.debug.emit("MIDDLEWARE", hook="before_agent", name=type(item).__name__)
            try:
                await item.abefore_agent(execution)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.before_agent failed", cause=exc
                ) from exc

    def model(self, request: ModelRequest, handler: ModelHandler) -> Any:
        for item in self.middleware:
            self.debug.emit("MIDDLEWARE", hook="before_model", name=type(item).__name__)
            try:
                item.before_model(request)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.before_model failed", cause=exc
                ) from exc
        call = handler
        for item in reversed(self.middleware):
            next_call = call

            def wrapped_model(
                req: ModelRequest, mw: AgentMiddleware = item, nxt: ModelHandler = next_call
            ) -> Any:
                return mw.wrap_model_call(req, nxt)

            call = wrapped_model
        response = call(request)
        for item in reversed(self.middleware):
            try:
                updated = item.after_model(request, response)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.after_model failed", cause=exc
                ) from exc
            response = response if updated is None else updated
        return response

    async def amodel(self, request: ModelRequest, handler: AsyncModelHandler) -> Any:
        for item in self.middleware:
            self.debug.emit("MIDDLEWARE", hook="before_model", name=type(item).__name__)
            try:
                await item.abefore_model(request)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.before_model failed", cause=exc
                ) from exc
        call = handler
        for item in reversed(self.middleware):
            next_call = call

            async def wrapped(req: ModelRequest, mw: AgentMiddleware = item, nxt: AsyncModelHandler = next_call) -> Any:
                return await mw.awrap_model_call(req, nxt)

            call = wrapped
        response = await call(request)
        for item in reversed(self.middleware):
            try:
                updated = await item.aafter_model(request, response)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.after_model failed", cause=exc
                ) from exc
            response = response if updated is None else updated
        return response

    def tool(self, request: ToolRequest, handler: ToolHandler) -> Any:
        call = handler
        for item in reversed(self.middleware):
            next_call = call

            def wrapped_tool(
                req: ToolRequest, mw: AgentMiddleware = item, nxt: ToolHandler = next_call
            ) -> Any:
                return mw.wrap_tool_call(req, nxt)

            call = wrapped_tool
        return call(request)

    async def atool(self, request: ToolRequest, handler: AsyncToolHandler) -> Any:
        call = handler
        for item in reversed(self.middleware):
            next_call = call

            async def wrapped(req: ToolRequest, mw: AgentMiddleware = item, nxt: AsyncToolHandler = next_call) -> Any:
                return await mw.awrap_tool_call(req, nxt)

            call = wrapped
        return await call(request)

    def after_agent(self, execution: AgentExecution, result: Any) -> Any:
        for item in reversed(self.middleware):
            self.debug.emit("MIDDLEWARE", hook="after_agent", name=type(item).__name__)
            try:
                updated = item.after_agent(execution, result)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.after_agent failed", cause=exc
                ) from exc
            result = result if updated is None else updated
        return result

    async def aafter_agent(self, execution: AgentExecution, result: Any) -> Any:
        for item in reversed(self.middleware):
            self.debug.emit("MIDDLEWARE", hook="after_agent", name=type(item).__name__)
            try:
                updated = await item.aafter_agent(execution, result)
            except AgentError:
                raise
            except Exception as exc:
                raise MiddlewareError(
                    f"{type(item).__name__}.after_agent failed", cause=exc
                ) from exc
            result = result if updated is None else updated
        return result


def default_middleware(
    *, retry_attempts: int = 2, timeout_seconds: float = 60.0, call_limit: int = 48
) -> tuple[AgentMiddleware, ...]:
    return (
        CallLimitMiddleware(call_limit),
        RetryMiddleware(retry_attempts),
        TimeoutMiddleware(timeout_seconds),
    )
