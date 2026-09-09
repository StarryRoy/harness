"""Application-facing Agent object and its isolated tool adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from .definition import AgentDefinition
from .errors import SubAgentError
from .middleware import MiddlewarePipeline
from .observability import RuntimeMetrics, StreamEvent
from .runtime import AgentRuntime, _derive_child_session_id


@dataclass(frozen=True, slots=True)
class SubAgentResult:
    content: Any = None
    status: str = "success"
    metadata: Mapping[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "metadata": dict(self.metadata or {}),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AgentResult(Mapping[str, Any]):
    """Stable application result with opt-in access to the complete graph state."""

    output: Any
    status: Literal["completed", "paused", "error"]
    session_id: str
    structured_output: Any = None
    interrupts: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] | None = None
    state: Mapping[str, Any] | None = None

    def __getitem__(self, key: str) -> Any:
        return (self.state or {})[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.state or {})

    def __len__(self) -> int:
        return len(self.state or {})


class Agent:
    def __init__(
        self,
        definition: AgentDefinition,
        runtime: AgentRuntime,
        *,
        subagents: Mapping[str, Agent] | None = None,
    ):
        self.definition = definition
        self.runtime = runtime
        self._subagents = dict(subagents or {})

    def invoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> AgentResult:
        state = self.runtime.invoke(
            value, config, session_id=session_id, context=context, memory_id=memory_id
        )
        return self._result(state)

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> AgentResult:
        state = await self.runtime.ainvoke(
            value, config, session_id=session_id, context=context, memory_id=memory_id
        )
        return await self._aresult(state)

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
        return self.runtime.stream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

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
        return self.runtime.astream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

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
        return self.runtime.raw_stream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

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
        return self.runtime.araw_stream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

    @property
    def metrics(self) -> RuntimeMetrics:
        return self.runtime.metrics

    def resume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> AgentResult:
        """Resume a paused sensitive tool call with approve/reject/edit."""
        return self._result(
            self.runtime.resume(session_id=session_id, decision=decision)
        )

    async def aresume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> AgentResult:
        state = await self.runtime.aresume(session_id=session_id, decision=decision)
        return await self._aresult(state)

    def clear_session(self, session_id: str) -> None:
        calls = self.runtime._subagent_calls_for_session(session_id=session_id)
        for tool_name, call_id in calls:
            child = self._subagents[tool_name]
            child_session = self.runtime._child_session_id(
                session_id=session_id,
                subagent_name=child.definition.name,
                tool_call_id=call_id,
            )
            child.clear_session(child_session)
        self.runtime.clear_session(session_id=session_id)

    async def aclear_session(self, session_id: str) -> None:
        calls = await self.runtime._asubagent_calls_for_session(session_id=session_id)
        for tool_name, call_id in calls:
            child = self._subagents[tool_name]
            child_session = self.runtime._child_session_id(
                session_id=session_id,
                subagent_name=child.definition.name,
                tool_call_id=call_id,
            )
            await child.aclear_session(child_session)
        await self.runtime.aclear_session(session_id=session_id)

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> BaseTool:
        """Expose this agent as a task-only tool with isolated runtime state."""
        tool_name = name or self.definition.name
        tool_description = (
            description
            or self.definition.description
            or (f"Delegate a self-contained task to the {self.definition.name} agent.")
        )

        def child_scope() -> tuple[str | None, str | None]:
            try:
                parent = MiddlewarePipeline.current_execution()
            except RuntimeError:
                return None, None
            call_id = parent.metadata.get("current_tool_call_id")
            parent_scope = parent.metadata.get("thread_id") or parent.session_id
            if not call_id or not parent_scope:
                return None, None
            runtime_metadata = parent.input.get("runtime_metadata", {})
            memory_id = runtime_metadata.get("memory_id")
            if self.runtime.memory is None:
                memory_id = None
            return (
                _derive_child_session_id(
                    parent_scope, self.definition.name, str(call_id)
                ),
                memory_id,
            )

        def pause_payload(task: str, child_session: str) -> dict[str, Any]:
            return {
                "subagent": self.definition.name,
                "task": task,
                "interrupts": self.runtime.pending_interrupts(session_id=child_session),
            }

        async def apause_payload(task: str, child_session: str) -> dict[str, Any]:
            return {
                "subagent": self.definition.name,
                "task": task,
                "interrupts": await self.runtime.apending_interrupts(
                    session_id=child_session
                ),
            }

        max_child_interrupts = (
            self.definition.runtime_config.max_iterations
            * self.definition.runtime_config.call_limit
        )

        def replay_parent_interrupts(
            task: str, child_session: str, completed: int
        ) -> None:
            if completed > max_child_interrupts:
                raise RuntimeError("SubAgent exceeded bounded HITL resume limit")
            payload = pause_payload(task, child_session)
            for _ in range(completed):
                interrupt(payload)

        async def areplay_parent_interrupts(
            task: str, child_session: str, completed: int
        ) -> None:
            if completed > max_child_interrupts:
                raise RuntimeError("SubAgent exceeded bounded HITL resume limit")
            payload = await apause_payload(task, child_session)
            for _ in range(completed):
                interrupt(payload)

        def _run(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            try:
                child_session, memory_id = child_scope()
                paused_at_entry = bool(
                    child_session and self.runtime.is_paused(session_id=child_session)
                )
                if paused_at_entry:
                    completed = self.runtime.completed_approval_count(
                        session_id=child_session
                    )
                    replay_parent_interrupts(task, child_session, completed)
                    state = None
                else:
                    state = self.invoke(
                        task, session_id=child_session, memory_id=memory_id
                    )
                    completed = 0
                if child_session:
                    for _ in range(completed, max_child_interrupts):
                        if not self.runtime.is_paused(session_id=child_session):
                            break
                        decision = interrupt(pause_payload(task, child_session))
                        state = self.resume(
                            session_id=child_session,
                            decision=decision,
                        )
                    else:
                        if self.runtime.is_paused(session_id=child_session):
                            raise RuntimeError(
                                "SubAgent exceeded bounded HITL resume limit"
                            )
                if state is None:
                    raise RuntimeError("SubAgent did not reach a completed state")
                result = self._subagent_result(state)
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - errors become isolated tool results
                error = SubAgentError(
                    f"SubAgent '{self.definition.name}' invocation failed", cause=exc
                )
                result = SubAgentResult(
                    status="error",
                    metadata={
                        "agent": self.definition.name,
                        "error_type": type(error).__name__,
                    },
                    error=str(error),
                )
            return result.as_dict()

        async def _arun(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            try:
                child_session, memory_id = child_scope()
                paused_at_entry = bool(
                    child_session
                    and await self.runtime.ais_paused(session_id=child_session)
                )
                if paused_at_entry:
                    completed = await self.runtime.acompleted_approval_count(
                        session_id=child_session
                    )
                    await areplay_parent_interrupts(task, child_session, completed)
                    state = None
                else:
                    state = await self.ainvoke(
                        task, session_id=child_session, memory_id=memory_id
                    )
                    completed = 0
                if child_session:
                    for _ in range(completed, max_child_interrupts):
                        if not await self.runtime.ais_paused(session_id=child_session):
                            break
                        decision = interrupt(await apause_payload(task, child_session))
                        state = await self.aresume(
                            session_id=child_session,
                            decision=decision,
                        )
                    else:
                        if await self.runtime.ais_paused(session_id=child_session):
                            raise RuntimeError(
                                "SubAgent exceeded bounded HITL resume limit"
                            )
                if state is None:
                    raise RuntimeError("SubAgent did not reach a completed state")
                result = self._subagent_result(state)
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - errors become isolated tool results
                error = SubAgentError(
                    f"SubAgent '{self.definition.name}' invocation failed", cause=exc
                )
                result = SubAgentResult(
                    status="error",
                    metadata={
                        "agent": self.definition.name,
                        "error_type": type(error).__name__,
                    },
                    error=str(error),
                )
            return result.as_dict()

        def run(task: str) -> dict[str, Any]:
            parent_execution = None
            try:
                parent_execution = MiddlewarePipeline.current_execution()
                observer = parent_execution.debug or self.runtime.observer
            except RuntimeError:
                observer = self.runtime.observer
            with observer.span(
                "subagent", name=self.definition.name, task=task
            ) as span:
                result = _run(task)
                span.status = str(result["status"])
                span.metadata["result_status"] = result["status"]
                if result.get("error"):
                    span.metadata["error"] = result["error"]
                if parent_execution is not None:
                    parent_execution.metadata.pop("resume_pending", None)
                return result

        async def arun(task: str) -> dict[str, Any]:
            parent_execution = None
            try:
                parent_execution = MiddlewarePipeline.current_execution()
                observer = parent_execution.debug or self.runtime.observer
            except RuntimeError:
                observer = self.runtime.observer
            with observer.span(
                "subagent", name=self.definition.name, task=task
            ) as span:
                result = await _arun(task)
                span.status = str(result["status"])
                span.metadata["result_status"] = result["status"]
                if result.get("error"):
                    span.metadata["error"] = result["error"]
                if parent_execution is not None:
                    parent_execution.metadata.pop("resume_pending", None)
                return result

        tool = StructuredTool.from_function(
            run,
            coroutine=arun,
            name=tool_name,
            description=tool_description,
        )
        object.__setattr__(tool, "_harness_subagent", self)
        return tool

    def _result(self, state: Mapping[str, Any]) -> AgentResult:
        metadata = dict(state.get("runtime_metadata", {}))
        session_id = str(metadata.get("session_id", ""))
        interrupts = tuple(
            self.runtime.pending_interrupts(session_id=session_id) if session_id else ()
        )
        return self._build_result(state, session_id, interrupts)

    async def _aresult(self, state: Mapping[str, Any]) -> AgentResult:
        metadata = dict(state.get("runtime_metadata", {}))
        session_id = str(metadata.get("session_id", ""))
        interrupts = tuple(
            await self.runtime.apending_interrupts(session_id=session_id)
            if session_id
            else ()
        )
        return self._build_result(state, session_id, interrupts)

    def _build_result(
        self,
        state: Mapping[str, Any],
        session_id: str,
        interrupts: tuple[Any, ...],
    ) -> AgentResult:
        structured = state.get("structured_response")
        output = structured
        if "structured_response" not in state:
            messages = state.get("messages", [])
            final = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            output = final.content if final is not None else None
        plan_status = state.get("plan", {}).get("status")
        status: Literal["completed", "paused", "error"] = "completed"
        if interrupts:
            status = "paused"
        elif plan_status == "failed":
            status = "error"
        return AgentResult(
            output=output,
            status=status,
            session_id=session_id,
            structured_output=structured,
            interrupts=interrupts,
            metadata={
                "agent": self.definition.name,
                "iteration": state.get("iteration", 0),
                "plan_status": plan_status,
                "trace_id": state.get("runtime_metadata", {}).get("trace_id"),
            },
            state=state,
        )

    def _subagent_result(self, state: Mapping[str, Any]) -> SubAgentResult:
        if "structured_response" in state:
            content = state["structured_response"]
        else:
            messages = state.get("messages", [])
            final = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            content = final.content if final is not None else None
        return SubAgentResult(
            content=content,
            metadata={"agent": self.definition.name},
        )
