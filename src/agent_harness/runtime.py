"""Runtime facade around a strategy-owned compiled graph."""

from typing import Any, AsyncIterator, Iterator

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .debug import DebugHandler
from .definition import AgentDefinition
from .skills import SkillRegistry
from .strategy import AgentStrategy


class AgentRuntime:
    def __init__(self, definition: AgentDefinition, strategy: AgentStrategy, skills: SkillRegistry):
        self.definition = definition
        self.debug = DebugHandler(definition.runtime_config.debug)
        self.skills = skills
        self.graph = strategy.build_graph(definition, skills, self.debug)

    def _input(self, value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, str):
            value = {"messages": [HumanMessage(content=value)]}
        elif not isinstance(value, dict) or "messages" not in value:
            raise TypeError("input must be a string or a state mapping containing 'messages'")
        state = dict(value)
        state.setdefault("iteration", 0)
        state.setdefault("runtime_metadata", {})
        state.setdefault("available_skills", list(self.skills.summaries()))
        state.setdefault("loaded_skills", [])
        state.setdefault("active_skill", None)
        return state

    def invoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        self.debug.emit("AGENT START", name=self.definition.name)
        try:
            return self.graph.invoke(self._input(value), config)
        finally:
            self.debug.emit("AGENT END", name=self.definition.name)

    async def ainvoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        self.debug.emit("AGENT START", name=self.definition.name)
        try:
            return await self.graph.ainvoke(self._input(value), config)
        finally:
            self.debug.emit("AGENT END", name=self.definition.name)

    def stream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        return self.graph.stream(self._input(value), config, **kwargs)

    def astream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        return self.graph.astream(self._input(value), config, **kwargs)
