"""Cross-session memory backed by LangGraph Store and managed by LangMem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.store.base import BaseStore

from .errors import MemoryError


@dataclass(slots=True)
class MemoryConfig:
    instructions: str | None = None
    schema: type | None = None
    model: Any | None = None
    namespace: str = "memory"
    search_limit: int = 6


class LongTermMemory:
    """Thin integration boundary; extraction/update semantics remain in LangMem."""

    def __init__(self, store: BaseStore, config: MemoryConfig, default_model: Any) -> None:
        self.store, self.config = store, config
        try:
            from langmem import create_memory_store_manager
        except ImportError as exc:
            raise MemoryError(
                "Long-term memory requires the optional 'langmem' package", cause=exc
            ) from exc
        kwargs: dict[str, Any] = {"model": config.model or default_model}
        if config.instructions is not None:
            kwargs["instructions"] = config.instructions
        if config.schema is not None:
            kwargs["schemas"] = [config.schema]
        self.manager = create_memory_store_manager(**kwargs)

    def namespace(self, agent: str, memory_id: str) -> tuple[str, ...]:
        return (self.config.namespace, agent, memory_id)

    def load(self, agent: str, memory_id: str, query: str) -> list[dict[str, Any]]:
        items = self.store.search(
            self.namespace(agent, memory_id), query=query, limit=self.config.search_limit
        )
        return [dict(item.value) for item in items]

    def update(self, agent: str, memory_id: str, messages: list[Any]) -> Any:
        namespace = self.namespace(agent, memory_id)
        config = {"configurable": {"langgraph_store": self.store, "namespace": namespace}}
        # LangMem owns extraction, deduplication, consolidation and update decisions.
        return self.manager.invoke({"messages": messages}, config=config)

    async def aupdate(self, agent: str, memory_id: str, messages: list[Any]) -> Any:
        namespace = self.namespace(agent, memory_id)
        config = {"configurable": {"langgraph_store": self.store, "namespace": namespace}}
        return await self.manager.ainvoke({"messages": messages}, config=config)
