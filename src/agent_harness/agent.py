"""Application-facing Agent object."""

from typing import Any, AsyncIterator, Iterator

from langchain_core.runnables import RunnableConfig

from .definition import AgentDefinition
from .runtime import AgentRuntime


class Agent:
    def __init__(self, definition: AgentDefinition, runtime: AgentRuntime):
        self.definition = definition
        self.runtime = runtime

    def invoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return self.runtime.invoke(value, config)

    async def ainvoke(self, value: str | dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return await self.runtime.ainvoke(value, config)

    def stream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        return self.runtime.stream(value, config, **kwargs)

    def astream(self, value: str | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        return self.runtime.astream(value, config, **kwargs)
