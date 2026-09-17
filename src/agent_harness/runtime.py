"""Runtime facade providing sessions around a strategy-owned LangGraph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .context import AgentContextManager
from .debug import DebugHandler
from .definition import AgentDefinition
from .errors import (
    AgentError,
    HITLError,
    PersistenceError,
    SessionError,
    StrategyError,
)
from .memory import LongTermMemory
from .middleware import AgentExecution, MiddlewarePipeline
from .observability import (
    BufferedEventSink,
    EventSink,
    EventType,
    MetricsEventSink,
    RuntimeEvent,
    RuntimeMetrics,
    RuntimeObserver,
    StreamEvent,
    StreamEventType,
)
from .persistence import validate_checkpointer
from .skills import SkillRegistry
from .state import HARNESS_STATE_FIELDS
from .strategy import AgentStrategy


_MEMORY_MESSAGE_START = "_memory_message_start"


def _derive_child_session_id(
    parent_thread_id: str, subagent_name: str, tool_call_id: str
) -> str:
    digest = hashlib.sha256(
        f"{parent_thread_id}:{subagent_name}:{tool_call_id}".encode()
    ).hexdigest()
    return f"parent-call-{digest}"


class AgentRuntime:
    def __init__(
        self,
        definition: AgentDefinition,
        strategy: AgentStrategy,
        skills: SkillRegistry,
        checkpointer: Any,
        *,
        session_namespace: str | None = None,
        memory: LongTermMemory | None = None,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        self.definition = definition
        self.metrics_sink = MetricsEventSink()
        debug_sink = DebugHandler(
            definition.runtime_config.debug, definition.runtime_config.debug_format
        )
        sinks: tuple[EventSink, ...] = (self.metrics_sink, *tuple(event_sinks))
        if debug_sink.enabled:
            sinks = (*sinks, debug_sink)
        self.observer = RuntimeObserver(sinks, definition.runtime_config.observability)
        # ``debug`` remains as a compatibility alias for strategies and integrations.
        self.debug = self.observer
        self.skills = skills
        self.context = AgentContextManager(
            definition.instructions,
            skills,
            definition.runtime_config.context_policy,
            model=definition.model,
        )
        self.middleware = MiddlewarePipeline(definition.middleware, self.debug)
        self.memory = memory
        self.checkpointer = validate_checkpointer(checkpointer)
        if session_namespace is not None and (
            not isinstance(session_namespace, str) or not session_namespace.strip()
        ):
            raise ValueError("session_namespace must be a non-empty string")
        self._namespace = (
            session_namespace.strip()
            if session_namespace is not None
            else definition.name
        )
        try:
            self.graph = strategy.build_graph(
                definition,
                skills,
                self.debug,
                context=self.context,
                middleware=self.middleware,
                checkpointer=self.checkpointer,
            )
        except (TypeError, ValueError, AgentError):
            raise
        except Exception as exc:
            raise StrategyError(
                "Unable to build agent strategy graph", cause=exc
            ) from exc

    @property
    def metrics(self) -> RuntimeMetrics:
        return self.metrics_sink.snapshot()

    @staticmethod
    def _trace_values(
        saved_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, str | None]:
        saved = dict(saved_metadata or {})
        parent = RuntimeObserver.current_context()
        trace_id = str(
            saved.get("trace_id") or (parent.trace_id if parent else uuid.uuid4().hex)
        )
        previous_span = saved.get("agent_span_id")
        parent_span_id = (
            parent.span_id
            if parent is not None and parent.trace_id == trace_id
            else str(previous_span)
            if previous_span
            else None
        )
        return trace_id, uuid.uuid4().hex, parent_span_id

    @staticmethod
    def _execution_metadata(
        thread_id: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        *,
        resume_pending: bool = False,
    ) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "resume_pending": resume_pending,
        }

    def _emit_agent_end(
        self,
        started: float,
        *,
        status: str,
        resumed: bool = False,
        error: BaseException | None = None,
    ) -> None:
        event_type = EventType.AGENT_ERROR if error is not None else EventType.AGENT_END
        self.observer.emit(
            event_type,
            status="error" if error is not None else status,
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"resumed": resumed},
            error=error,
        )

    def _emit_skill_discover(self, state: Mapping[str, Any]) -> None:
        available = list(state.get("available_skills", []))
        self.observer.emit(
            EventType.SKILL_DISCOVER,
            status="success",
            metadata={
                "count": len(available),
                "skills": [item.get("name") for item in available],
            },
        )

    def _input(self, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, str):
            state: dict[str, Any] = {"messages": [HumanMessage(content=value)]}
        elif isinstance(value, dict) and "messages" in value:
            forbidden = sorted((set(value) & HARNESS_STATE_FIELDS) - {"messages"})
            if forbidden:
                raise ValueError(
                    "Input cannot set Harness-managed state fields: "
                    + ", ".join(forbidden)
                )
            state = dict(value)
        else:
            raise TypeError(
                "input must be a string or a state mapping containing 'messages'"
            )
        state["iteration"] = 0
        state["runtime_metadata"] = {}
        state["available_skills"] = self.context.discover(value)
        return state

    def _config(
        self, config: RunnableConfig | None, session_id: str | None
    ) -> tuple[RunnableConfig, str, str]:
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise ValueError("session_id must be a non-empty string")
        public_id = session_id.strip() if session_id else f"session-{uuid.uuid4().hex}"
        thread_id = hashlib.sha256(
            f"{self._namespace}:{public_id}".encode()
        ).hexdigest()
        merged: dict[str, Any] = dict(config or {})
        configurable = dict(merged.get("configurable", {}))
        configurable["thread_id"] = thread_id
        merged["configurable"] = configurable
        merged.setdefault(
            "recursion_limit",
            self.definition.runtime_config.max_iterations * 4
            + self.definition.runtime_config.call_limit
            + 16,
        )
        return merged, thread_id, public_id

    @staticmethod
    def _message_count(snapshot: Any) -> int:
        return len(snapshot.values.get("messages", []))

    def invoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        graph_config, thread_id, public_id = self._config(config, session_id)
        trace_id, agent_span_id, parent_span_id = self._trace_values()
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
            "session_id": public_id,
            "trace_id": trace_id,
            "agent_span_id": agent_span_id,
            "parent_span_id": parent_span_id,
            _MEMORY_MESSAGE_START: self._message_count(
                self.graph.get_state(graph_config)
            ),
        }
        with self.observer.trace(
            agent_name=self.definition.name,
            session_id=public_id,
            trace_id=trace_id,
            span_id=agent_span_id,
            parent_span_id=parent_span_id,
        ):
            started = time.perf_counter()
            self.observer.emit(
                EventType.AGENT_START,
                status="started",
                metadata={"resumed": False},
            )
            execution = AgentExecution(
                self.definition.name,
                state,
                public_id,
                dict(context or {}),
                self._execution_metadata(
                    thread_id, trace_id, agent_span_id, parent_span_id
                ),
                debug=self.observer,
            )
            after_called = False
            final_status = "success"
            try:
                self._emit_skill_discover(state)
                self._load_memory(state, value, memory_id)
                with self.middleware.execution_scope(execution):
                    self.middleware.before_agent(execution)
                    result = self.graph.invoke(state, graph_config)
                    after_called = True
                    result = self.middleware.after_agent(execution, result)
                    completed = self._completed(graph_config)
                    final_status = "success" if completed else "paused"
                    if completed:
                        self._update_memory(
                            result,
                            memory_id,
                            state["runtime_metadata"][_MEMORY_MESSAGE_START],
                        )
                    return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    with self.middleware.execution_scope(execution):
                        self.middleware.after_agent(execution, None)
                self._emit_agent_end(started, status="error", error=exc)
                if isinstance(exc, AgentError):
                    raise
                raise SessionError("Agent invocation failed", cause=exc) from exc
            else:
                raise AssertionError("unreachable")
            finally:
                if execution.error is None:
                    self._emit_agent_end(started, status=final_status)

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        graph_config, thread_id, public_id = self._config(config, session_id)
        trace_id, agent_span_id, parent_span_id = self._trace_values()
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
            "session_id": public_id,
            "trace_id": trace_id,
            "agent_span_id": agent_span_id,
            "parent_span_id": parent_span_id,
            _MEMORY_MESSAGE_START: self._message_count(
                await self.graph.aget_state(graph_config)
            ),
        }
        with self.observer.trace(
            agent_name=self.definition.name,
            session_id=public_id,
            trace_id=trace_id,
            span_id=agent_span_id,
            parent_span_id=parent_span_id,
        ):
            started = time.perf_counter()
            self.observer.emit(
                EventType.AGENT_START,
                status="started",
                metadata={"resumed": False},
            )
            execution = AgentExecution(
                self.definition.name,
                state,
                public_id,
                dict(context or {}),
                self._execution_metadata(
                    thread_id, trace_id, agent_span_id, parent_span_id
                ),
                debug=self.observer,
            )
            after_called = False
            final_status = "success"
            try:
                self._emit_skill_discover(state)
                self._load_memory(state, value, memory_id)
                with self.middleware.execution_scope(execution):
                    await self.middleware.abefore_agent(execution)
                    result = await self.graph.ainvoke(state, graph_config)
                    after_called = True
                    result = await self.middleware.aafter_agent(execution, result)
                    completed = self._completed(graph_config)
                    final_status = "success" if completed else "paused"
                    if completed:
                        await self._aupdate_memory(
                            result,
                            memory_id,
                            state["runtime_metadata"][_MEMORY_MESSAGE_START],
                        )
                    return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    with self.middleware.execution_scope(execution):
                        await self.middleware.aafter_agent(execution, None)
                self._emit_agent_end(started, status="error", error=exc)
                if isinstance(exc, AgentError):
                    raise
                raise SessionError("Async agent invocation failed", cause=exc) from exc
            else:
                raise AssertionError("unreachable")
            finally:
                if execution.error is None:
                    self._emit_agent_end(started, status=final_status)

    def stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Yield stable application events rather than LangGraph implementation chunks."""
        self._require_stream_session(session_id)
        buffer = BufferedEventSink()
        trace_values = self._trace_values()

        def iterator() -> Iterator[StreamEvent]:
            try:
                for raw in self._raw_stream(
                    value,
                    config,
                    session_id=session_id,
                    context=context,
                    memory_id=memory_id,
                    _extra_sinks=(buffer,),
                    _trace_values=trace_values,
                    **kwargs,
                ):
                    yield from self._application_events(buffer.drain())
                    yield from self._text_events(raw, session_id, trace_values)
            except Exception:
                yield from self._application_events(buffer.drain())
                raise
            yield from self._application_events(buffer.drain())
            graph_config, _, public_id = self._config(config, session_id)
            if self._completed(graph_config):
                state = dict(self.graph.get_state(graph_config).values)
                yield self._final_stream_event(state, public_id, trace_values)

        return iterator()

    def raw_stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Advanced API exposing the underlying LangGraph stream unchanged."""
        self._require_stream_session(session_id)
        return self._raw_stream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

    def _raw_stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None,
        context: Mapping[str, Any] | None,
        memory_id: str | None,
        _extra_sinks: Sequence[EventSink] = (),
        _trace_values: tuple[str, str, str | None] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        trace_id, agent_span_id, parent_span_id = _trace_values or self._trace_values()
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
            "session_id": public_id,
            "trace_id": trace_id,
            "agent_span_id": agent_span_id,
            "parent_span_id": parent_span_id,
            _MEMORY_MESSAGE_START: self._message_count(
                self.graph.get_state(graph_config)
            ),
        }

        def iterator() -> Iterator[Any]:
            with self.observer.trace(
                agent_name=self.definition.name,
                session_id=public_id,
                trace_id=trace_id,
                span_id=agent_span_id,
                parent_span_id=parent_span_id,
                extra_sinks=_extra_sinks,
            ):
                started = time.perf_counter()
                self.observer.emit(
                    EventType.AGENT_START,
                    status="started",
                    metadata={"resumed": False, "stream": True},
                )
                execution = AgentExecution(
                    self.definition.name,
                    state,
                    public_id,
                    dict(context or {}),
                    self._execution_metadata(
                        thread_id, trace_id, agent_span_id, parent_span_id
                    ),
                    debug=self.observer,
                )
                after_called = False
                final_status = "success"
                try:
                    self._emit_skill_discover(state)
                    self._load_memory(state, value, memory_id)
                    with self.middleware.execution_scope(execution):
                        self.middleware.before_agent(execution)
                        yield from self.graph.stream(state, graph_config, **kwargs)
                        final = dict(self.graph.get_state(graph_config).values)
                        after_called = True
                        final = self.middleware.after_agent(execution, final)
                        completed = self._completed(graph_config)
                        final_status = "success" if completed else "paused"
                        if completed:
                            self._update_memory(
                                final,
                                memory_id,
                                state["runtime_metadata"][_MEMORY_MESSAGE_START],
                            )
                except Exception as exc:
                    execution.error = exc
                    if not after_called:
                        with self.middleware.execution_scope(execution):
                            self.middleware.after_agent(execution, None)
                    self._emit_agent_end(started, status="error", error=exc)
                    if isinstance(exc, AgentError):
                        raise
                    raise SessionError("Agent stream failed", cause=exc) from exc
                finally:
                    if execution.error is None:
                        self._emit_agent_end(started, status=final_status)

        return iterator()

    def astream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self._require_stream_session(session_id)
        buffer = BufferedEventSink()
        trace_values = self._trace_values()

        async def iterator() -> AsyncIterator[StreamEvent]:
            try:
                async for raw in self._araw_stream(
                    value,
                    config,
                    session_id=session_id,
                    context=context,
                    memory_id=memory_id,
                    _extra_sinks=(buffer,),
                    _trace_values=trace_values,
                    **kwargs,
                ):
                    for event in self._application_events(buffer.drain()):
                        yield event
                    for event in self._text_events(raw, session_id, trace_values):
                        yield event
            except Exception:
                for event in self._application_events(buffer.drain()):
                    yield event
                raise
            for event in self._application_events(buffer.drain()):
                yield event
            graph_config, _, public_id = self._config(config, session_id)
            snapshot = await self.graph.aget_state(graph_config)
            if not tuple(snapshot.next):
                yield self._final_stream_event(
                    dict(snapshot.values), public_id, trace_values
                )

        return iterator()

    def araw_stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async advanced API exposing LangGraph chunks unchanged."""
        self._require_stream_session(session_id)
        return self._araw_stream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

    def _araw_stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None,
        context: Mapping[str, Any] | None,
        memory_id: str | None,
        _extra_sinks: Sequence[EventSink] = (),
        _trace_values: tuple[str, str, str | None] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        trace_id, agent_span_id, parent_span_id = _trace_values or self._trace_values()
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
            "session_id": public_id,
            "trace_id": trace_id,
            "agent_span_id": agent_span_id,
            "parent_span_id": parent_span_id,
        }

        async def iterator() -> AsyncIterator[Any]:
            state["runtime_metadata"][_MEMORY_MESSAGE_START] = self._message_count(
                await self.graph.aget_state(graph_config)
            )
            with self.observer.trace(
                agent_name=self.definition.name,
                session_id=public_id,
                trace_id=trace_id,
                span_id=agent_span_id,
                parent_span_id=parent_span_id,
                extra_sinks=_extra_sinks,
            ):
                started = time.perf_counter()
                self.observer.emit(
                    EventType.AGENT_START,
                    status="started",
                    metadata={"resumed": False, "stream": True},
                )
                execution = AgentExecution(
                    self.definition.name,
                    state,
                    public_id,
                    dict(context or {}),
                    self._execution_metadata(
                        thread_id, trace_id, agent_span_id, parent_span_id
                    ),
                    debug=self.observer,
                )
                after_called = False
                final_status = "success"
                try:
                    self._emit_skill_discover(state)
                    self._load_memory(state, value, memory_id)
                    with self.middleware.execution_scope(execution):
                        await self.middleware.abefore_agent(execution)
                        async for raw in self.graph.astream(
                            state, graph_config, **kwargs
                        ):
                            yield raw
                        final = dict((await self.graph.aget_state(graph_config)).values)
                        after_called = True
                        final = await self.middleware.aafter_agent(execution, final)
                        snapshot = await self.graph.aget_state(graph_config)
                        completed = not tuple(snapshot.next)
                        final_status = "success" if completed else "paused"
                        if completed:
                            await self._aupdate_memory(
                                final,
                                memory_id,
                                state["runtime_metadata"][_MEMORY_MESSAGE_START],
                            )
                except Exception as exc:
                    execution.error = exc
                    if not after_called:
                        with self.middleware.execution_scope(execution):
                            await self.middleware.aafter_agent(execution, None)
                    self._emit_agent_end(started, status="error", error=exc)
                    if isinstance(exc, AgentError):
                        raise
                    raise SessionError("Async agent stream failed", cause=exc) from exc
                finally:
                    if execution.error is None:
                        self._emit_agent_end(started, status=final_status)

        return iterator()

    @staticmethod
    def _application_events(
        events: Sequence[RuntimeEvent],
    ) -> Iterator[StreamEvent]:
        mapping = {
            EventType.TOOL_START.value: StreamEventType.TOOL_START.value,
            EventType.TOOL_END.value: StreamEventType.TOOL_END.value,
            EventType.SUBAGENT_START.value: StreamEventType.SUBAGENT_START.value,
            EventType.SUBAGENT_END.value: StreamEventType.SUBAGENT_END.value,
            EventType.HITL_PAUSE.value: StreamEventType.APPROVAL_REQUIRED.value,
            EventType.PLAN_CREATE.value: StreamEventType.PLAN_UPDATE.value,
            EventType.PLAN_STEP.value: StreamEventType.PLAN_UPDATE.value,
            EventType.PLAN_REPLAN.value: StreamEventType.PLAN_UPDATE.value,
            EventType.PLAN_COMPLETE.value: StreamEventType.PLAN_UPDATE.value,
            EventType.PLAN_FAIL.value: StreamEventType.PLAN_UPDATE.value,
            EventType.AGENT_ERROR.value: StreamEventType.ERROR.value,
            EventType.MODEL_ERROR.value: StreamEventType.ERROR.value,
            EventType.TOOL_ERROR.value: StreamEventType.ERROR.value,
            EventType.SUBAGENT_ERROR.value: StreamEventType.ERROR.value,
        }
        for event in events:
            target = mapping.get(event.event_type)
            if target is None:
                continue
            data = dict(event.metadata)
            data["status"] = event.status
            if event.duration_ms is not None:
                data["duration_ms"] = event.duration_ms
            if event.error is not None:
                data["error"] = event.error
            if target == StreamEventType.PLAN_UPDATE.value:
                data.setdefault("update_type", event.event_type.removeprefix("plan."))
            yield StreamEvent(
                target,
                data,
                timestamp=event.timestamp,
                session_id=event.session_id,
                trace_id=event.trace_id,
                span_id=event.span_id,
                parent_span_id=event.parent_span_id,
            )

    @staticmethod
    def _text_events(
        raw: Any,
        session_id: str | None,
        trace_values: tuple[str, str, str | None],
    ) -> Iterator[StreamEvent]:
        messages: list[AIMessage] = []

        def collect(value: Any) -> None:
            if isinstance(value, AIMessage):
                messages.append(value)
            elif isinstance(value, Mapping):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(raw)
        trace_id, span_id, parent_span_id = trace_values
        for message in messages:
            content = message.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, Mapping) and block.get("type") == "text"
                )
            else:
                text = ""
            if text:
                yield StreamEvent(
                    StreamEventType.TEXT_DELTA.value,
                    {"delta": text, "text": text},
                    session_id=session_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                )

    @staticmethod
    def _final_stream_event(
        state: Mapping[str, Any],
        session_id: str,
        trace_values: tuple[str, str, str | None],
    ) -> StreamEvent:
        structured = state.get("structured_response")
        output = structured
        if "structured_response" not in state:
            output_message = next(
                (
                    message
                    for message in reversed(state.get("messages", []))
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            output = output_message.content if output_message is not None else None
        trace_id, span_id, parent_span_id = trace_values
        return StreamEvent(
            StreamEventType.FINAL.value,
            {
                "output": output,
                "structured_output": structured,
                "status": "completed",
            },
            session_id=session_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

    @staticmethod
    def _memory_query(value: str | dict[str, Any]) -> str:
        if isinstance(value, str):
            return value
        return " ".join(
            str(getattr(item, "content", item)) for item in value["messages"][-3:]
        )

    def _load_memory(
        self, state: dict[str, Any], value: Any, memory_id: str | None
    ) -> None:
        if memory_id is None:
            return
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        if self.memory is None:
            raise ValueError(
                "memory_id was provided but long-term memory is not configured"
            )
        started = time.perf_counter()
        try:
            loaded = self.memory.load(
                self.definition.name, memory_id.strip(), self._memory_query(value)
            )
        except Exception as exc:
            self.observer.emit(
                EventType.MEMORY_LOAD,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata={"memory_id": memory_id},
                error=exc,
            )
            raise
        state["long_term_memories"] = loaded
        self.observer.emit(
            EventType.MEMORY_LOAD,
            status="success",
            duration_ms=(time.perf_counter() - started) * 1000,
            metadata={"memory_id": memory_id, "count": len(loaded)},
        )

    def _update_memory(
        self,
        result: Mapping[str, Any],
        memory_id: str | None,
        message_start: int = 0,
    ) -> None:
        if self.memory and memory_id:
            started = time.perf_counter()
            try:
                self.memory.update(
                    self.definition.name,
                    memory_id.strip(),
                    list(result.get("messages", []))[message_start:],
                )
            except Exception as exc:
                self.observer.emit(
                    EventType.MEMORY_UPDATE,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    metadata={"memory_id": memory_id},
                    error=exc,
                )
                raise
            self.observer.emit(
                EventType.MEMORY_UPDATE,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata={"memory_id": memory_id},
            )

    async def _aupdate_memory(
        self,
        result: Mapping[str, Any],
        memory_id: str | None,
        message_start: int = 0,
    ) -> None:
        if self.memory and memory_id:
            started = time.perf_counter()
            try:
                await self.memory.aupdate(
                    self.definition.name,
                    memory_id.strip(),
                    list(result.get("messages", []))[message_start:],
                )
            except Exception as exc:
                self.observer.emit(
                    EventType.MEMORY_UPDATE,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    metadata={"memory_id": memory_id},
                    error=exc,
                )
                raise
            self.observer.emit(
                EventType.MEMORY_UPDATE,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata={"memory_id": memory_id},
            )

    def _completed(self, config: RunnableConfig) -> bool:
        """A checkpoint with a pending node is paused (including HITL), not complete."""
        return not tuple(self.graph.get_state(config).next)

    @staticmethod
    def _snapshot_interrupts(snapshot: Any) -> list[Any]:
        return [
            interrupt.value
            for task in getattr(snapshot, "tasks", ())
            for interrupt in getattr(task, "interrupts", ())
        ]

    def pending_interrupts(self, *, session_id: str) -> list[Any]:
        config, _, _ = self._config(None, session_id)
        return self._snapshot_interrupts(self.graph.get_state(config))

    async def apending_interrupts(self, *, session_id: str) -> list[Any]:
        config, _, _ = self._config(None, session_id)
        return self._snapshot_interrupts(await self.graph.aget_state(config))

    def is_paused(self, *, session_id: str) -> bool:
        return bool(self.pending_interrupts(session_id=session_id))

    async def ais_paused(self, *, session_id: str) -> bool:
        return bool(await self.apending_interrupts(session_id=session_id))

    @staticmethod
    def _require_stream_session(session_id: str | None) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise SessionError(
                "stream and astream require an explicit non-empty session_id"
            )

    def _subagent_calls_for_session(
        self, *, session_id: str
    ) -> tuple[tuple[str, str], ...]:
        """Read durable SubAgent tool-call relationships from the Main checkpoint."""
        try:
            config, _, _ = self._config(None, session_id)
            return self._subagent_calls(self.graph.get_state(config).values)
        except Exception as exc:
            raise PersistenceError(
                "Unable to inspect persistent SubAgent sessions", cause=exc
            ) from exc

    async def _asubagent_calls_for_session(
        self, *, session_id: str
    ) -> tuple[tuple[str, str], ...]:
        try:
            config, _, _ = self._config(None, session_id)
            snapshot = await self.graph.aget_state(config)
            return self._subagent_calls(snapshot.values)
        except Exception as exc:
            raise PersistenceError(
                "Unable to inspect persistent SubAgent sessions asynchronously",
                cause=exc,
            ) from exc

    def _child_session_id(
        self, *, session_id: str, subagent_name: str, tool_call_id: str
    ) -> str:
        _, parent_thread_id, _ = self._config(None, session_id)
        return _derive_child_session_id(parent_thread_id, subagent_name, tool_call_id)

    def _subagent_calls(self, state: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        subagent_names = set(self.definition.subagent_names)
        calls: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for message in state.get("messages", []):
            for call in getattr(message, "tool_calls", ()):
                name = call.get("name")
                call_id = call.get("id")
                relationship = (name, call_id)
                if (
                    name in subagent_names
                    and isinstance(call_id, str)
                    and call_id
                    and relationship not in seen
                ):
                    seen.add(relationship)
                    calls.append(relationship)
        return tuple(calls)

    def clear_session(self, *, session_id: str) -> None:
        try:
            _, thread_id, _ = self._config(None, session_id)
            result = self.checkpointer.delete_thread(thread_id)
            if inspect.isawaitable(result):
                raise TypeError("Checkpointer only supports asynchronous deletion")
        except Exception as exc:
            raise PersistenceError(
                "Unable to clear persistent session", cause=exc
            ) from exc

    async def aclear_session(self, *, session_id: str) -> None:
        try:
            _, thread_id, _ = self._config(None, session_id)
            try:
                await self.checkpointer.adelete_thread(thread_id)
            except NotImplementedError:
                await asyncio.to_thread(self.checkpointer.delete_thread, thread_id)
        except Exception as exc:
            raise PersistenceError(
                "Unable to clear persistent session asynchronously", cause=exc
            ) from exc

    def completed_approval_count(self, *, session_id: str) -> int:
        """Count approval-gated calls already completed in a child session."""
        config, _, _ = self._config(None, session_id)
        return self._completed_approval_count(self.graph.get_state(config).values)

    async def acompleted_approval_count(self, *, session_id: str) -> int:
        config, _, _ = self._config(None, session_id)
        snapshot = await self.graph.aget_state(config)
        return self._completed_approval_count(snapshot.values)

    def _completed_approval_count(self, state: Mapping[str, Any]) -> int:
        approval_tools = {
            tool.name
            for tool in self.definition.tools
            if (tool.metadata or {}).get("harness_approval")
        }
        return sum(
            isinstance(message, ToolMessage) and message.name in approval_tools
            for message in state.get("messages", [])
        )

    @staticmethod
    def _decision(decision: str | Mapping[str, Any]) -> dict[str, Any]:
        value = {"decision": decision} if isinstance(decision, str) else dict(decision)
        action = value.get("decision", value.get("action"))
        if action not in {"approve", "reject", "edit"}:
            raise ValueError("decision must be approve, reject, or edit")
        if action == "edit" and not isinstance(value.get("args"), Mapping):
            raise ValueError("edit decision requires an args mapping")
        value["decision"] = action
        return value

    def _emit_hitl_resume(self, decision: Mapping[str, Any]) -> None:
        action = str(decision["decision"])
        metadata = {"decision": action}
        if action == "edit":
            metadata["arguments"] = decision.get("args", {})
        self.observer.emit(EventType.HITL_RESUME, status="resumed", metadata=metadata)
        action_types = {
            "approve": EventType.HITL_APPROVE,
            "reject": EventType.HITL_REJECT,
            "edit": EventType.HITL_EDIT,
        }
        self.observer.emit(action_types[action], status=action, metadata=metadata)

    def resume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        config, thread_id, public_id = self._config(None, session_id)
        saved = dict(self.graph.get_state(config).values)
        metadata = dict(saved.get("runtime_metadata", {}))
        memory_id = metadata.get("memory_id")
        normalized = self._decision(decision)
        trace_id, agent_span_id, parent_span_id = self._trace_values(metadata)
        metadata.update(
            trace_id=trace_id,
            agent_span_id=agent_span_id,
            parent_span_id=parent_span_id,
            session_id=public_id,
        )
        saved["runtime_metadata"] = metadata
        with self.observer.trace(
            agent_name=self.definition.name,
            session_id=public_id,
            trace_id=trace_id,
            span_id=agent_span_id,
            parent_span_id=parent_span_id,
        ):
            started = time.perf_counter()
            self.observer.emit(
                EventType.AGENT_START,
                status="started",
                metadata={"resumed": True},
            )
            self._emit_hitl_resume(normalized)
            execution = AgentExecution(
                self.definition.name,
                saved,
                public_id,
                dict(metadata.get("business_context", {})),
                self._execution_metadata(
                    thread_id,
                    trace_id,
                    agent_span_id,
                    parent_span_id,
                    resume_pending=True,
                ),
                debug=self.observer,
            )
            after_called = False
            final_status = "success"
            try:
                with self.middleware.execution_scope(execution):
                    self.middleware.before_agent(execution)
                    result = self.graph.invoke(Command(resume=normalized), config)
                    after_called = True
                    result = self.middleware.after_agent(execution, result)
                    completed = self._completed(config)
                    final_status = "success" if completed else "paused"
                    if completed:
                        self._update_memory(
                            result,
                            memory_id,
                            int(metadata.get(_MEMORY_MESSAGE_START, 0)),
                        )
                    return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    with self.middleware.execution_scope(execution):
                        self.middleware.after_agent(execution, None)
                self._emit_agent_end(started, status="error", resumed=True, error=exc)
                if isinstance(exc, AgentError):
                    raise
                raise HITLError(
                    "Unable to resume interrupted agent", cause=exc
                ) from exc
            finally:
                if execution.error is None:
                    self._emit_agent_end(started, status=final_status, resumed=True)

    async def aresume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        config, thread_id, public_id = self._config(None, session_id)
        saved = dict((await self.graph.aget_state(config)).values)
        metadata = dict(saved.get("runtime_metadata", {}))
        memory_id = metadata.get("memory_id")
        normalized = self._decision(decision)
        trace_id, agent_span_id, parent_span_id = self._trace_values(metadata)
        metadata.update(
            trace_id=trace_id,
            agent_span_id=agent_span_id,
            parent_span_id=parent_span_id,
            session_id=public_id,
        )
        saved["runtime_metadata"] = metadata
        with self.observer.trace(
            agent_name=self.definition.name,
            session_id=public_id,
            trace_id=trace_id,
            span_id=agent_span_id,
            parent_span_id=parent_span_id,
        ):
            started = time.perf_counter()
            self.observer.emit(
                EventType.AGENT_START,
                status="started",
                metadata={"resumed": True},
            )
            self._emit_hitl_resume(normalized)
            execution = AgentExecution(
                self.definition.name,
                saved,
                public_id,
                dict(metadata.get("business_context", {})),
                self._execution_metadata(
                    thread_id,
                    trace_id,
                    agent_span_id,
                    parent_span_id,
                    resume_pending=True,
                ),
                debug=self.observer,
            )
            after_called = False
            final_status = "success"
            try:
                with self.middleware.execution_scope(execution):
                    await self.middleware.abefore_agent(execution)
                    result = await self.graph.ainvoke(
                        Command(resume=normalized), config
                    )
                    after_called = True
                    result = await self.middleware.aafter_agent(execution, result)
                    completed = self._completed(config)
                    final_status = "success" if completed else "paused"
                    if completed:
                        await self._aupdate_memory(
                            result,
                            memory_id,
                            int(metadata.get(_MEMORY_MESSAGE_START, 0)),
                        )
                    return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    with self.middleware.execution_scope(execution):
                        await self.middleware.aafter_agent(execution, None)
                self._emit_agent_end(started, status="error", resumed=True, error=exc)
                if isinstance(exc, AgentError):
                    raise
                raise HITLError(
                    "Unable to resume interrupted agent", cause=exc
                ) from exc
            finally:
                if execution.error is None:
                    self._emit_agent_end(started, status=final_status, resumed=True)
