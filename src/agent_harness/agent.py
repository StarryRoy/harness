"""Application-facing Agent object and its isolated tool adapter."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from .definition import AgentDefinition
from .errors import SubAgentError
from .middleware import MiddlewarePipeline
from .runtime import AgentRuntime


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


class Agent:
    def __init__(self, definition: AgentDefinition, runtime: AgentRuntime):
        self.definition = definition
        self.runtime = runtime

    def invoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        return self.runtime.invoke(
            value, config, session_id=session_id, context=context, memory_id=memory_id
        )

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.runtime.ainvoke(
            value, config, session_id=session_id, context=context, memory_id=memory_id
        )

    def stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
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
    ) -> AsyncIterator[Any]:
        return self.runtime.astream(
            value,
            config,
            session_id=session_id,
            context=context,
            memory_id=memory_id,
            **kwargs,
        )

    def resume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Resume a paused sensitive tool call with approve/reject/edit."""
        return self.runtime.resume(session_id=session_id, decision=decision)

    async def aresume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        return await self.runtime.aresume(session_id=session_id, decision=decision)

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
            digest = hashlib.sha256(
                f"{parent_scope}:{self.definition.name}:{call_id}".encode()
            ).hexdigest()
            runtime_metadata = parent.input.get("runtime_metadata", {})
            memory_id = runtime_metadata.get("memory_id")
            if self.runtime.memory is None:
                memory_id = None
            return f"parent-call-{digest}", memory_id

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

        def run(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            self.runtime.debug.emit(
                "SUBAGENT CALL", name=self.definition.name, task=task
            )
            try:
                child_session, memory_id = child_scope()
                if child_session and self.runtime.is_paused(session_id=child_session):
                    decision = interrupt(pause_payload(task, child_session))
                    state = self.resume(
                        session_id=child_session,
                        decision=decision,
                    )
                else:
                    state = self.invoke(
                        task, session_id=child_session, memory_id=memory_id
                    )
                    if child_session and self.runtime.is_paused(
                        session_id=child_session
                    ):
                        decision = interrupt(pause_payload(task, child_session))
                        state = self.resume(
                            session_id=child_session,
                            decision=decision,
                        )
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
            self.runtime.debug.emit(
                "SUBAGENT RESULT", name=self.definition.name, status=result.status
            )
            return result.as_dict()

        async def arun(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            self.runtime.debug.emit(
                "SUBAGENT CALL", name=self.definition.name, task=task
            )
            try:
                child_session, memory_id = child_scope()
                if child_session and await self.runtime.ais_paused(
                    session_id=child_session
                ):
                    decision = interrupt(await apause_payload(task, child_session))
                    state = await self.aresume(
                        session_id=child_session,
                        decision=decision,
                    )
                else:
                    state = await self.ainvoke(
                        task, session_id=child_session, memory_id=memory_id
                    )
                    if child_session and await self.runtime.ais_paused(
                        session_id=child_session
                    ):
                        decision = interrupt(await apause_payload(task, child_session))
                        state = await self.aresume(
                            session_id=child_session,
                            decision=decision,
                        )
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
            self.runtime.debug.emit(
                "SUBAGENT RESULT", name=self.definition.name, status=result.status
            )
            return result.as_dict()

        return StructuredTool.from_function(
            run,
            coroutine=arun,
            name=tool_name,
            description=tool_description,
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
