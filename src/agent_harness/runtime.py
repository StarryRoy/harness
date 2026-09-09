"""Runtime facade providing sessions around a strategy-owned LangGraph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
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
from .persistence import validate_checkpointer
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
        memory: LongTermMemory | None = None,
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

    def invoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
        }
        self._load_memory(state, value, memory_id)
        graph_config, thread_id, public_id = self._config(config, session_id)
        state["runtime_metadata"]["session_id"] = public_id
        execution = AgentExecution(
            self.definition.name,
            state,
            public_id,
            dict(context or {}),
            {"thread_id": thread_id},
            debug=self.debug,
        )
        self.debug.emit("SESSION", session_id=public_id)
        self.debug.emit("AGENT START", name=self.definition.name)
        with self.middleware.execution_scope(execution):
            after_called = False
            try:
                self.middleware.before_agent(execution)
                result = self.graph.invoke(state, graph_config)
                after_called = True
                result = self.middleware.after_agent(execution, result)
                if self._completed(graph_config):
                    self._update_memory(result, memory_id)
                return result
            except Exception as exc:
                execution.error = exc
                self.debug.emit("ERROR", error=str(exc))
                if not after_called:
                    self.middleware.after_agent(execution, None)
                if isinstance(exc, AgentError):
                    raise
                raise SessionError("Agent invocation failed", cause=exc) from exc
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

    async def ainvoke(
        self,
        value: str | dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        session_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
        }
        self._load_memory(state, value, memory_id)
        graph_config, thread_id, public_id = self._config(config, session_id)
        state["runtime_metadata"]["session_id"] = public_id
        execution = AgentExecution(
            self.definition.name,
            state,
            public_id,
            dict(context or {}),
            {"thread_id": thread_id},
            debug=self.debug,
        )
        self.debug.emit("SESSION", session_id=public_id)
        self.debug.emit("AGENT START", name=self.definition.name)
        with self.middleware.execution_scope(execution):
            after_called = False
            try:
                await self.middleware.abefore_agent(execution)
                result = await self.graph.ainvoke(state, graph_config)
                after_called = True
                result = await self.middleware.aafter_agent(execution, result)
                if self._completed(graph_config):
                    await self._aupdate_memory(result, memory_id)
                return result
            except Exception as exc:
                execution.error = exc
                self.debug.emit("ERROR", error=str(exc))
                if not after_called:
                    await self.middleware.aafter_agent(execution, None)
                if isinstance(exc, AgentError):
                    raise
                raise SessionError("Async agent invocation failed", cause=exc) from exc
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

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
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
        }
        self._load_memory(state, value, memory_id)
        graph_config, thread_id, public_id = self._config(config, session_id)
        state["runtime_metadata"]["session_id"] = public_id
        execution = AgentExecution(
            self.definition.name,
            state,
            public_id,
            dict(context or {}),
            {"thread_id": thread_id},
            debug=self.debug,
        )

        def iterator() -> Iterator[Any]:
            self.debug.emit("SESSION", session_id=public_id)
            self.debug.emit("AGENT START", name=self.definition.name)
            with self.middleware.execution_scope(execution):
                after_called = False
                try:
                    self.middleware.before_agent(execution)
                    yield from self.graph.stream(state, graph_config, **kwargs)
                    final = dict(self.graph.get_state(graph_config).values)
                    after_called = True
                    final = self.middleware.after_agent(execution, final)
                    if self._completed(graph_config):
                        self._update_memory(final, memory_id)
                except Exception as exc:
                    execution.error = exc
                    self.debug.emit("ERROR", error=str(exc))
                    if not after_called:
                        self.middleware.after_agent(execution, None)
                    if isinstance(exc, AgentError):
                        raise
                    raise SessionError("Agent stream failed", cause=exc) from exc
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
        memory_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        state = self._input(value)
        state["runtime_metadata"] = {
            "memory_id": memory_id,
            "business_context": dict(context or {}),
        }
        self._load_memory(state, value, memory_id)
        graph_config, thread_id, public_id = self._config(config, session_id)
        state["runtime_metadata"]["session_id"] = public_id
        execution = AgentExecution(
            self.definition.name,
            state,
            public_id,
            dict(context or {}),
            {"thread_id": thread_id},
            debug=self.debug,
        )

        async def iterator() -> AsyncIterator[Any]:
            result: Any = None
            self.debug.emit("SESSION", session_id=public_id)
            self.debug.emit("AGENT START", name=self.definition.name)
            with self.middleware.execution_scope(execution):
                after_called = False
                try:
                    await self.middleware.abefore_agent(execution)
                    async for result in self.graph.astream(
                        state, graph_config, **kwargs
                    ):
                        yield result
                    final = dict((await self.graph.aget_state(graph_config)).values)
                    after_called = True
                    final = await self.middleware.aafter_agent(execution, final)
                    if self._completed(graph_config):
                        await self._aupdate_memory(final, memory_id)
                except Exception as exc:
                    execution.error = exc
                    self.debug.emit("ERROR", error=str(exc))
                    if not after_called:
                        await self.middleware.aafter_agent(execution, None)
                    if isinstance(exc, AgentError):
                        raise
                    raise SessionError("Async agent stream failed", cause=exc) from exc
                finally:
                    self.debug.emit("AGENT END", name=self.definition.name)

        return iterator()

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
        loaded = self.memory.load(
            self.definition.name, memory_id.strip(), self._memory_query(value)
        )
        state["long_term_memories"] = loaded
        self.debug.emit("MEMORY LOAD", memory_id=memory_id, count=len(loaded))

    def _update_memory(self, result: Mapping[str, Any], memory_id: str | None) -> None:
        if self.memory and memory_id:
            self.memory.update(
                self.definition.name,
                memory_id.strip(),
                list(result.get("messages", [])),
            )
            self.debug.emit("MEMORY UPDATE", memory_id=memory_id)

    async def _aupdate_memory(
        self, result: Mapping[str, Any], memory_id: str | None
    ) -> None:
        if self.memory and memory_id:
            await self.memory.aupdate(
                self.definition.name,
                memory_id.strip(),
                list(result.get("messages", [])),
            )
            self.debug.emit("MEMORY UPDATE", memory_id=memory_id)

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

    def resume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        config, thread_id, public_id = self._config(None, session_id)
        saved = dict(self.graph.get_state(config).values)
        metadata = dict(saved.get("runtime_metadata", {}))
        memory_id = metadata.get("memory_id")
        execution = AgentExecution(
            self.definition.name,
            saved,
            public_id,
            dict(metadata.get("business_context", {})),
            {"thread_id": thread_id},
            debug=self.debug,
        )
        self.debug.emit("HITL RESUME", decision=decision)
        with self.middleware.execution_scope(execution):
            after_called = False
            try:
                self.middleware.before_agent(execution)
                result = self.graph.invoke(
                    Command(resume=self._decision(decision)), config
                )
                after_called = True
                result = self.middleware.after_agent(execution, result)
                if self._completed(config):
                    self._update_memory(result, memory_id)
                return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    self.middleware.after_agent(execution, None)
                if isinstance(exc, AgentError):
                    raise
                raise HITLError(
                    "Unable to resume interrupted agent", cause=exc
                ) from exc

    async def aresume(
        self, *, session_id: str, decision: str | Mapping[str, Any]
    ) -> dict[str, Any]:
        config, thread_id, public_id = self._config(None, session_id)
        saved = dict((await self.graph.aget_state(config)).values)
        metadata = dict(saved.get("runtime_metadata", {}))
        memory_id = metadata.get("memory_id")
        execution = AgentExecution(
            self.definition.name,
            saved,
            public_id,
            dict(metadata.get("business_context", {})),
            {"thread_id": thread_id},
            debug=self.debug,
        )
        self.debug.emit("HITL RESUME", decision=decision)
        with self.middleware.execution_scope(execution):
            after_called = False
            try:
                await self.middleware.abefore_agent(execution)
                result = await self.graph.ainvoke(
                    Command(resume=self._decision(decision)), config
                )
                after_called = True
                result = await self.middleware.aafter_agent(execution, result)
                if self._completed(config):
                    await self._aupdate_memory(result, memory_id)
                return result
            except Exception as exc:
                execution.error = exc
                if not after_called:
                    await self.middleware.aafter_agent(execution, None)
                if isinstance(exc, AgentError):
                    raise
                raise HITLError(
                    "Unable to resume interrupted agent", cause=exc
                ) from exc
