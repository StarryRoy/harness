"""Stable runtime events, safe delivery, trace propagation, and basic metrics."""

from __future__ import annotations

import contextvars
import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Protocol, TextIO, runtime_checkable


class EventType(str, Enum):
    """Event names are application-owned and independent of graph node names."""

    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    AGENT_ERROR = "agent.error"
    MODEL_START = "model.start"
    MODEL_END = "model.end"
    MODEL_ERROR = "model.error"
    MODEL_RETRY = "model.retry"
    MODEL_FALLBACK = "model.fallback"
    TOOL_START = "tool.start"
    TOOL_END = "tool.end"
    TOOL_ERROR = "tool.error"
    TOOL_RETRY = "tool.retry"
    SUBAGENT_START = "subagent.start"
    SUBAGENT_END = "subagent.end"
    SUBAGENT_ERROR = "subagent.error"
    SKILL_DISCOVER = "skill.discover"
    SKILL_LOAD = "skill.load"
    SKILL_UNLOAD = "skill.unload"
    SKILL_SCRIPT = "skill.script"
    HITL_PAUSE = "hitl.pause"
    HITL_RESUME = "hitl.resume"
    HITL_APPROVE = "hitl.approve"
    HITL_REJECT = "hitl.reject"
    HITL_EDIT = "hitl.edit"
    PLAN_CREATE = "plan.create"
    PLAN_STEP = "plan.step"
    PLAN_REPLAN = "plan.replan"
    PLAN_COMPLETE = "plan.complete"
    PLAN_FAIL = "plan.fail"
    MEMORY_LOAD = "memory.load"
    MEMORY_UPDATE = "memory.update"
    CONTEXT_SUMMARIZE = "context.summarize"
    MIDDLEWARE_HOOK = "middleware.hook"
    MCP_TOOL = "mcp.tool"
    GUARDRAIL = "guardrail"


class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"
    APPROVAL_REQUIRED = "approval_required"
    PLAN_UPDATE = "plan_update"
    FINAL = "final"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str = ""
    session_id: str | None = None
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str | None = None
    status: str = "info"
    duration_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def type(self) -> str:
        return self.event_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Stable frontend event; data intentionally excludes graph state by default."""

    event_type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str | None = None

    @property
    def type(self) -> str:
        return self.event_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": dict(self.data),
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }

    to_dict = as_dict


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: RuntimeEvent) -> None: ...


PayloadDetail = Literal["none", "minimal", "standard", "full"]


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Controls event payload visibility; identifiers and timings are always kept."""

    payload_detail: PayloadDetail = "standard"
    max_payload_chars: int = 2_000
    max_collection_items: int = 50
    sensitive_fields: tuple[str, ...] = (
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    )

    def __post_init__(self) -> None:
        if self.payload_detail not in {"none", "minimal", "standard", "full"}:
            raise ValueError("payload_detail must be none, minimal, standard, or full")
        if self.payload_detail == "none":
            object.__setattr__(self, "payload_detail", "minimal")
        if self.max_payload_chars < 1 or self.max_collection_items < 1:
            raise ValueError("observability payload limits must be positive")
        object.__setattr__(
            self,
            "sensitive_fields",
            tuple(str(item).casefold() for item in self.sensitive_fields),
        )


_SECRET_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|credential|password|secret|token)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_PAYLOAD_KEYS = {
    "args",
    "arguments",
    "input",
    "output",
    "payload",
    "result",
    "task",
}
_MESSAGE_KEYS = {"message", "messages", "prompt", "response"}


def _redact_text(value: str, limit: int) -> str:
    value = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    if len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


def sanitize_payload(value: Any, config: ObservabilityConfig) -> Any:
    """Recursively redact secrets and impose deterministic payload bounds."""

    sensitive = set(config.sensitive_fields)

    def clean(item: Any, *, key: str | None = None, depth: int = 0) -> Any:
        normalized = (key or "").casefold().replace("-", "_")
        if normalized and any(
            normalized == field or normalized.endswith("_" + field)
            for field in sensitive
        ):
            return "[REDACTED]"
        if config.payload_detail == "minimal" and normalized in _PAYLOAD_KEYS:
            return "[omitted]"
        if config.payload_detail != "full" and normalized in _MESSAGE_KEYS:
            return "[omitted]"
        if depth >= 8:
            return "[depth limited]"
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            return _redact_text(item, config.max_payload_chars)
        if isinstance(item, bytes):
            return f"[bytes:{len(item)}]"
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            entries = list(item.items())[: config.max_collection_items]
            for child_key, child_value in entries:
                rendered_key = _redact_text(str(child_key), config.max_payload_chars)
                result[rendered_key] = clean(
                    child_value, key=str(child_key), depth=depth + 1
                )
            if len(item) > len(entries):
                result["__truncated_items__"] = len(item) - len(entries)
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            values = list(item)
            cleaned = [
                clean(child, depth=depth + 1)
                for child in values[: config.max_collection_items]
            ]
            if len(values) > len(cleaned):
                cleaned.append(f"[{len(values) - len(cleaned)} items truncated]")
            return cleaned
        return _redact_text(str(item), config.max_payload_chars)

    return clean(value)


class CompositeEventSink:
    """Fan out safely: one broken sink never prevents delivery to another."""

    def __init__(self, sinks: Sequence[EventSink]) -> None:
        self.sinks = tuple(sinks)

    def emit(self, event: RuntimeEvent) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:  # noqa: BLE001,S110 - sink isolation is the contract
                pass


class ConsoleEventSink:
    def __init__(
        self,
        *,
        format: Literal["text", "print", "json", "md"] = "text",
        stream: TextIO | None = None,
    ) -> None:
        if format not in {"text", "print", "json", "md"}:
            raise ValueError("format must be text, print, json, or md")
        self.format = "text" if format == "print" else format
        self.stream = stream or sys.stdout

    def emit(self, event: RuntimeEvent) -> None:
        if self.format == "json":
            output = json.dumps(event.as_dict(), default=str, ensure_ascii=False)
        else:
            name = (
                f"**{event.event_type}**" if self.format == "md" else event.event_type
            )
            fields = [
                f"status={event.status}",
                f"agent={event.agent_name}",
                f"trace_id={event.trace_id}",
                f"span_id={event.span_id}",
            ]
            if event.duration_ms is not None:
                fields.append(f"duration_ms={event.duration_ms:.3f}")
            if event.metadata:
                fields.append(f"metadata={dict(event.metadata)!r}")
            if event.error:
                fields.append(f"error={event.error!r}")
            output = f"{name} {' '.join(fields)}"
        print(output, file=self.stream)


class JsonEventSink(ConsoleEventSink):
    def __init__(self, *, stream: TextIO | None = None) -> None:
        super().__init__(format="json", stream=stream)


class BufferedEventSink:
    """Thread-safe transient sink used by stable streaming and test doubles."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._events.append(event)

    def drain(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
        return events

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    agent_calls: int = 0
    agent_duration_ms: float = 0.0
    model_calls: int = 0
    model_duration_ms: float = 0.0
    tool_calls: int = 0
    tool_duration_ms: float = 0.0
    subagent_calls: int = 0
    subagent_duration_ms: float = 0.0
    retries: int = 0
    fallbacks: int = 0
    hitl_pauses: int = 0
    skill_loads: int = 0
    context_summaries: int = 0
    memory_loads: int = 0
    memory_updates: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __getitem__(self, key: str) -> int | float:
        return getattr(self, key)

    def as_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class MetricsEventSink:
    """Small in-process accumulator; exporters can consume the same events."""

    def __init__(self) -> None:
        self._values: dict[str, int | float] = RuntimeMetrics().as_dict()
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            if event.event_type == EventType.AGENT_START.value:
                self._values["agent_calls"] += 1
            elif event.event_type == EventType.MODEL_START.value:
                self._values["model_calls"] += 1
            elif event.event_type == EventType.TOOL_START.value:
                self._values["tool_calls"] += 1
            elif event.event_type == EventType.SUBAGENT_START.value:
                self._values["subagent_calls"] += 1
            elif event.event_type in {
                EventType.MODEL_RETRY.value,
                EventType.TOOL_RETRY.value,
            }:
                self._values["retries"] += 1
            elif event.event_type == EventType.MODEL_FALLBACK.value:
                self._values["fallbacks"] += 1
            elif event.event_type == EventType.HITL_PAUSE.value:
                self._values["hitl_pauses"] += 1
            elif event.event_type == EventType.SKILL_LOAD.value:
                self._values["skill_loads"] += 1
            elif event.event_type == EventType.CONTEXT_SUMMARIZE.value:
                self._values["context_summaries"] += 1
            elif event.event_type == EventType.MEMORY_LOAD.value:
                self._values["memory_loads"] += 1
            elif event.event_type == EventType.MEMORY_UPDATE.value:
                self._values["memory_updates"] += 1

            duration_fields = {
                EventType.AGENT_END.value: "agent_duration_ms",
                EventType.AGENT_ERROR.value: "agent_duration_ms",
                EventType.MODEL_END.value: "model_duration_ms",
                EventType.MODEL_ERROR.value: "model_duration_ms",
                EventType.TOOL_END.value: "tool_duration_ms",
                EventType.TOOL_ERROR.value: "tool_duration_ms",
                EventType.SUBAGENT_END.value: "subagent_duration_ms",
                EventType.SUBAGENT_ERROR.value: "subagent_duration_ms",
            }
            target = duration_fields.get(event.event_type)
            if target and event.duration_ms is not None:
                self._values[target] += event.duration_ms

            usage = event.metadata.get("token_usage")
            if isinstance(usage, Mapping):
                aliases = {
                    "input_tokens": ("input_tokens", "prompt_tokens"),
                    "output_tokens": ("output_tokens", "completion_tokens"),
                    "total_tokens": ("total_tokens",),
                }
                for target_name, source_names in aliases.items():
                    for source_name in source_names:
                        value = usage.get(source_name)
                        if isinstance(value, (int, float)):
                            self._values[target_name] += int(value)
                            break

    def snapshot(self) -> RuntimeMetrics:
        with self._lock:
            return RuntimeMetrics(**self._values)

    def reset(self) -> None:
        with self._lock:
            self._values = RuntimeMetrics().as_dict()


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    agent_name: str
    session_id: str | None
    sinks: tuple[EventSink, ...]


_TRACE_CONTEXT: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "agent_harness_trace_context", default=None
)


def _unique_sinks(sinks: Sequence[EventSink]) -> tuple[EventSink, ...]:
    result: list[EventSink] = []
    seen: set[int] = set()
    for sink in sinks:
        if id(sink) not in seen:
            seen.add(id(sink))
            result.append(sink)
    return tuple(result)


class SpanHandle:
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata: dict[str, Any] = dict(metadata)
        self.status = "success"


class RuntimeObserver:
    """Creates safe events and propagates trace context across sync/async paths."""

    def __init__(
        self,
        sinks: Sequence[EventSink] = (),
        config: ObservabilityConfig | None = None,
    ) -> None:
        self.sinks = _unique_sinks(tuple(sinks))
        self.config = config or ObservabilityConfig()

    @staticmethod
    def current_context() -> TraceContext | None:
        return _TRACE_CONTEXT.get()

    @contextmanager
    def trace(
        self,
        *,
        agent_name: str,
        session_id: str | None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        extra_sinks: Sequence[EventSink] = (),
    ) -> Iterator[TraceContext]:
        parent = _TRACE_CONTEXT.get()
        resolved_trace = trace_id or (parent.trace_id if parent else uuid.uuid4().hex)
        resolved_parent = parent_span_id
        if parent is not None and parent.trace_id == resolved_trace:
            resolved_parent = parent.span_id
        sinks = _unique_sinks(
            (*self.sinks, *(parent.sinks if parent else ()), *extra_sinks)
        )
        context = TraceContext(
            resolved_trace,
            span_id or uuid.uuid4().hex,
            resolved_parent,
            agent_name,
            session_id,
            sinks,
        )
        token = _TRACE_CONTEXT.set(context)
        try:
            yield context
        finally:
            _TRACE_CONTEXT.reset(token)

    def emit(
        self,
        event_type: EventType | str | RuntimeEvent,
        *,
        status: str = "info",
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        error: Any = None,
        **details: Any,
    ) -> RuntimeEvent:
        if isinstance(event_type, RuntimeEvent):
            event = event_type
        else:
            context = _TRACE_CONTEXT.get()
            combined = {**dict(metadata or {}), **details}
            safe_metadata = sanitize_payload(combined, self.config)
            safe_error = (
                None
                if error is None
                else _redact_text(str(error), self.config.max_payload_chars)
            )
            event = RuntimeEvent(
                event_type=(
                    event_type.value
                    if isinstance(event_type, EventType)
                    else str(event_type)
                ),
                agent_name=context.agent_name if context else "",
                session_id=context.session_id if context else None,
                trace_id=context.trace_id if context else uuid.uuid4().hex,
                span_id=context.span_id if context else uuid.uuid4().hex,
                parent_span_id=context.parent_span_id if context else None,
                status=status,
                duration_ms=duration_ms,
                metadata=safe_metadata,
                error=safe_error,
            )
        context = _TRACE_CONTEXT.get()
        sinks = context.sinks if context else self.sinks
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:  # noqa: BLE001,S110 - observer failures cannot fail a run
                pass
        return event

    @contextmanager
    def span(self, kind: str, **metadata: Any) -> Iterator[SpanHandle]:
        parent = _TRACE_CONTEXT.get()
        if parent is None:
            with (
                self.trace(agent_name="", session_id=None) as generated,
                self._bound_span(kind, generated, metadata) as handle,
            ):
                yield handle
            return
        with self._bound_span(kind, parent, metadata) as handle:
            yield handle

    @contextmanager
    def _bound_span(
        self, kind: str, parent: TraceContext, metadata: Mapping[str, Any]
    ) -> Iterator[SpanHandle]:
        context = TraceContext(
            parent.trace_id,
            uuid.uuid4().hex,
            parent.span_id,
            parent.agent_name,
            parent.session_id,
            parent.sinks,
        )
        token = _TRACE_CONTEXT.set(context)
        handle = SpanHandle(metadata)
        started = time.perf_counter()
        self.emit(f"{kind}.start", status="started", metadata=handle.metadata)
        try:
            yield handle
        except BaseException as exc:
            duration = (time.perf_counter() - started) * 1000
            if type(exc).__name__ in {"GraphInterrupt", "GraphBubbleUp"}:
                handle.status = "paused"
                self.emit(
                    f"{kind}.end",
                    status="paused",
                    duration_ms=duration,
                    metadata=handle.metadata,
                )
            else:
                self.emit(
                    f"{kind}.error",
                    status="error",
                    duration_ms=duration,
                    metadata=handle.metadata,
                    error=exc,
                )
            raise
        else:
            event_suffix = "error" if handle.status == "error" else "end"
            self.emit(
                f"{kind}.{event_suffix}",
                status=handle.status,
                duration_ms=(time.perf_counter() - started) * 1000,
                metadata=handle.metadata,
                error=(
                    handle.metadata.get("error") if handle.status == "error" else None
                ),
            )
        finally:
            _TRACE_CONTEXT.reset(token)


def token_usage(response: Any) -> dict[str, int]:
    """Normalize common LangChain/provider token metadata without requiring it."""

    candidates = [
        getattr(response, "usage_metadata", None),
        getattr(response, "response_metadata", {}).get("token_usage")
        if isinstance(getattr(response, "response_metadata", None), Mapping)
        else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            result = {
                str(key): int(value)
                for key, value in candidate.items()
                if isinstance(value, (int, float))
            }
            if result:
                return result
    return {}
