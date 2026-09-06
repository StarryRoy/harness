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
        # Runtime-owned fields must always begin from a trusted state. In particular,
        # loaded_skills can only be changed by the internal load_skill tool path.
        state["iteration"] = 0
        state.setdefault("runtime_metadata", {})
        state["available_skills"] = list(self.skills.summaries())
        state["loaded_skills"] = []
        state["active_skill"] = None
        return state

    def invoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        self.debug.emit("AGENT START", name=self.definition.name)
        try:
            return self.graph.invoke(self._input(value), config)
        except Exception as exc:
            self.debug.emit("ERROR", error=str(exc))
            raise
        finally:
            self.debug.emit("AGENT END", name=self.definition.name)

    async def ainvoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        self.debug.emit("AGENT START", name=self.definition.name)
        try:
            return await self.graph.ainvoke(self._input(value), config)
        except Exception as exc:
            self.debug.emit("ERROR", error=str(exc))
            raise
        finally:
            self.debug.emit("AGENT END", name=self.definition.name)

    def stream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        def iterator() -> Iterator[Any]:
            self.debug.emit("AGENT START", name=self.definition.name)
            try:
                yield from self.graph.stream(self._input(value), config, **kwargs)
            except Exception as exc:
                self.debug.emit("ERROR", error=str(exc))
                raise
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

        return iterator()

    def astream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        async def iterator() -> AsyncIterator[Any]:
            self.debug.emit("AGENT START", name=self.definition.name)
            try:
                async for item in self.graph.astream(self._input(value), config, **kwargs):
                    yield item
            except Exception as exc:
                self.debug.emit("ERROR", error=str(exc))
                raise
            finally:
                self.debug.emit("AGENT END", name=self.definition.name)

        return iterator()
