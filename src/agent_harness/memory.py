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

    def __post_init__(self) -> None:
        if isinstance(self.search_limit, bool) or not isinstance(
            self.search_limit, int
        ):
            raise TypeError("search_limit must be an integer")
        if self.search_limit < 1:
            raise ValueError("search_limit must be at least 1")


class LongTermMemory:
    """Thin integration boundary; extraction/update semantics remain in LangMem."""

    def __init__(
        self, store: BaseStore, config: MemoryConfig, default_model: Any
    ) -> None:
        self.store, self.config = store, config
        try:
            from langmem import create_memory_store_manager
        except ImportError as exc:
            raise MemoryError(
                "Long-term memory requires the optional 'langmem' package", cause=exc
            ) from exc
        kwargs: dict[str, Any] = {
            "namespace": (config.namespace, "{agent_name}", "{memory_id}"),
            "store": self.store,
            "query_limit": config.search_limit,
        }
        if config.instructions is not None:
            kwargs["instructions"] = config.instructions
        if config.schema is not None:
            kwargs["schemas"] = [config.schema]
        try:
            self.manager = create_memory_store_manager(
                config.model or default_model, **kwargs
            )
        except Exception as exc:
            raise MemoryError(
                "Unable to configure LangMem memory manager", cause=exc
            ) from exc

    @staticmethod
    def _runnable_config(agent: str, memory_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"agent_name": agent, "memory_id": memory_id}}

    def load(self, agent: str, memory_id: str, query: str) -> list[dict[str, Any]]:
        try:
            items = self.manager.search(
                query=query,
                limit=self.config.search_limit,
                config=self._runnable_config(agent, memory_id),
            )
            if any(getattr(item, "score", None) is None for item in items):
                raise MemoryError(
                    "Long-term memory Store returned unranked results for a semantic "
                    "query; verify that its embedding index is configured and that "
                    "existing records have been indexed"
                )
            return [
                dict(getattr(item, "value", item))
                for item in items[: self.config.search_limit]
            ]
        except MemoryError:
            raise
        except Exception as exc:
            raise MemoryError("Unable to search long-term memory", cause=exc) from exc

    def update(self, agent: str, memory_id: str, messages: list[Any]) -> Any:
        config = self._runnable_config(agent, memory_id)
        try:
            # LangMem owns extraction, deduplication, consolidation and updates.
            return self.manager.invoke({"messages": messages}, config=config)
        except Exception as exc:
            raise MemoryError("Unable to update long-term memory", cause=exc) from exc

    async def aupdate(self, agent: str, memory_id: str, messages: list[Any]) -> Any:
        config = self._runnable_config(agent, memory_id)
        try:
            return await self.manager.ainvoke({"messages": messages}, config=config)
        except Exception as exc:
            raise MemoryError("Unable to update long-term memory", cause=exc) from exc
