"""Runtime facade providing sessions around a strategy-owned LangGraph."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .context import AgentContextManager
from .debug import DebugHandler
from .definition import AgentDefinition
from .middleware import AgentExecution, MiddlewarePipeline
from .skills import SkillRegistry
from .state import HARNESS_STATE_FIELDS
from .strategy import AgentStrategy


class AgentRuntime:
    def __init__(
        self,
        definition: AgentDefinition,
        strategy: AgentStrategy,
        skills: SkillRegistry,
        checkpointer: Any,
        *,
        session_namespace: str | None = None,
    ) -> None:
        self.definition = definition
        self.debug = DebugHandler(
            definition.runtime_config.debug, definition.runtime_config.debug_format
        )
        self.skills = skills
        self.context = AgentContextManager(
            definition.instructions, skills, definition.runtime_config.context_policy
        )
        self.middleware = MiddlewarePipeline(definition.middleware, self.debug)
        if session_namespace is not None and (
            not isinstance(session_namespace, str) or not session_namespace.strip()
        ):
            raise ValueError("session_namespace must be a non-empty string")
        self._namespace = (
            session_namespace.strip() if session_namespace is not None else definition.name
        )
        self.graph = strategy.build_graph(
            definition,
            skills,
            self.debug,
            context=self.context,
            middleware=self.middleware,
            checkpointer=checkpointer,
        )

    def _input(self, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, str):
            state: dict[str, Any] = {"messages": [HumanMessage(content=value)]}
        elif isinstance(value, dict) and "messages" in value:
            forbidden = sorted((set(value) & HARNESS_STATE_FIELDS) - {"messages"})
            if forbidden:
                raise ValueError(
                    "Input cannot set Harness-managed state fields: " + ", ".join(forbidden)
                )
            state = dict(value)
        else:
            raise TypeError("input must be a string or a state mapping containing 'messages'")
        state["iteration"] = 0
        state["runtime_metadata"] = {}
        state["available_skills"] = self.context.discover(value)
        return state

    def _config(
        self, config: RunnableConfig | None, session_id: str | None
    ) -> tuple[RunnableConfig, str, str | None]:
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a non-empty string")
        public_id = session_id.strip() if session_id else None
        scope = public_id or f"ephemeral-{uuid.uuid4().hex}"
        thread_id = hashlib.sha256(f"{self._namespace}:{scope}".encode()).hexdigest()
        merged: dict[str, Any] = dict(config or {})
        configurable = dict(merged.get("configurable", {}))
        configurable["thread_id"] = thread_id
        merged["configurable"] = configurable
        merged.setdefault("recursion_limit", self.definition.runtime_config.max_iterations * 3 + 8)
        return merged, thread_id, public_id

    def invoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        execution = AgentExecution(
            self.definition.name, state, public_id, dict(context or {}), {"thread_id": thread_id}
        )
        self.debug.emit("SESSION", session_id=public_id or "ephemeral")
        self.debug.emit("AGENT START", name=self.definition.name)
        with self.middleware.execution_scope(execution):
            try:
                self.middleware.before_agent(execution)
                result = self.graph.invoke(state, graph_config)
                return self.middleware.after_agent(execution, result)
            except Exception as exc:
                execution.error = exc
                self.debug.emit("ERROR", error=str(exc))
                self.middleware.after_agent(execution, None)
                raise
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        execution = AgentExecution(
            self.definition.name, state, public_id, dict(context or {}), {"thread_id": thread_id}
        )
        self.debug.emit("SESSION", session_id=public_id or "ephemeral")
        self.debug.emit("AGENT START", name=self.definition.name)
        with self.middleware.execution_scope(execution):
            try:
                await self.middleware.abefore_agent(execution)
                result = await self.graph.ainvoke(state, graph_config)
                return await self.middleware.aafter_agent(execution, result)
            except Exception as exc:
                execution.error = exc
                self.debug.emit("ERROR", error=str(exc))
                await self.middleware.aafter_agent(execution, None)
                raise
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

    def stream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        execution = AgentExecution(
            self.definition.name, state, public_id, dict(context or {}), {"thread_id": thread_id}
        )

        def iterator() -> Iterator[Any]:
            result: Any = None
            self.debug.emit("SESSION", session_id=public_id or "ephemeral")
            self.debug.emit("AGENT START", name=self.definition.name)
            with self.middleware.execution_scope(execution):
                try:
                    self.middleware.before_agent(execution)
                    for result in self.graph.stream(state, graph_config, **kwargs):
                        yield result
                    self.middleware.after_agent(execution, result)
                except Exception as exc:
                    execution.error = exc
                    self.debug.emit("ERROR", error=str(exc))
                    self.middleware.after_agent(execution, None)
                    raise
                finally:
                    self.debug.emit("AGENT END", name=self.definition.name)

        return iterator()

    def astream(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        state = self._input(value)
        graph_config, thread_id, public_id = self._config(config, session_id)
        execution = AgentExecution(
            self.definition.name, state, public_id, dict(context or {}), {"thread_id": thread_id}
        )

        async def iterator() -> AsyncIterator[Any]:
            result: Any = None
            self.debug.emit("SESSION", session_id=public_id or "ephemeral")
            self.debug.emit("AGENT START", name=self.definition.name)
            with self.middleware.execution_scope(execution):
                try:
                    await self.middleware.abefore_agent(execution)
                    async for result in self.graph.astream(state, graph_config, **kwargs):
                        yield result
                    await self.middleware.aafter_agent(execution, result)
                except Exception as exc:
                    execution.error = exc
                    self.debug.emit("ERROR", error=str(exc))
                    await self.middleware.aafter_agent(execution, None)
                    raise
                finally:
                    self.debug.emit("AGENT END", name=self.definition.name)

        return iterator()
