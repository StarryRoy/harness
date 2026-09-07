"""Application-facing Agent object and its isolated tool adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from .definition import AgentDefinition
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
    ) -> dict[str, Any]:
        return self.runtime.invoke(
            value, config, session_id=session_id, context=context
        )

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.runtime.ainvoke(
            value, config, session_id=session_id, context=context
        )

    def stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return self.runtime.stream(
            value, config, session_id=session_id, context=context, **kwargs
        )

    def astream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        return self.runtime.astream(
            value, config, session_id=session_id, context=context, **kwargs
        )

    def as_tool(
        self, *, name: str | None = None, description: str | None = None
    ) -> BaseTool:
        """Expose this agent as a task-only tool with isolated runtime state."""
        tool_name = name or self.definition.name
        tool_description = description or self.definition.description or (
            f"Delegate a self-contained task to the {self.definition.name} agent."
        )

        def run(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            self.runtime.debug.emit("SUBAGENT CALL", name=self.definition.name, task=task)
            try:
                state = self.invoke(task)
                result = self._subagent_result(state)
            except Exception as exc:  # noqa: BLE001 - errors become isolated tool results
                result = SubAgentResult(
                    status="error",
                    metadata={"agent": self.definition.name},
                    error=str(exc),
                )
            self.runtime.debug.emit(
                "SUBAGENT RESULT", name=self.definition.name, status=result.status
            )
            return result.as_dict()

        async def arun(task: str) -> dict[str, Any]:
            """Delegate one explicit task and return only its final business result."""
            self.runtime.debug.emit("SUBAGENT CALL", name=self.definition.name, task=task)
            try:
                state = await self.ainvoke(task)
                result = self._subagent_result(state)
            except Exception as exc:  # noqa: BLE001 - errors become isolated tool results
                result = SubAgentResult(
                    status="error",
                    metadata={"agent": self.definition.name},
                    error=str(exc),
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
                (message for message in reversed(messages) if isinstance(message, AIMessage)),
                None,
            )
            content = final.content if final is not None else None
        return SubAgentResult(
            content=content,
            metadata={"agent": self.definition.name},
        )
