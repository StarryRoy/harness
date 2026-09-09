"""Persistent checkpoint/store configuration without a database vendor lock-in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import PersistenceError


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    checkpointer: Any | None = None
    store: Any | None = None


_UNSET = object()
_DEFAULTS = PersistenceConfig()


def _inherits_known_memory_type(value: Any, names: set[tuple[str, str]]) -> bool:
    return any((item.__module__, item.__name__) in names for item in type(value).mro())


def validate_checkpointer(checkpointer: Any) -> Any:
    if checkpointer is None:
        raise PersistenceError("A persistent LangGraph checkpointer is required")
    if _inherits_known_memory_type(
        checkpointer,
        {
            ("langgraph.checkpoint.memory", "InMemorySaver"),
            ("langgraph.checkpoint.memory", "MemorySaver"),
        },
    ):
        raise PersistenceError("In-memory LangGraph checkpointers are not supported")
    required = ("get_tuple", "put", "put_writes", "delete_thread")
    missing = [
        name for name in required if not callable(getattr(checkpointer, name, None))
    ]
    if missing:
        raise PersistenceError(
            "Persistent checkpointer is missing required methods: " + ", ".join(missing)
        )
    return checkpointer


def validate_store(store: Any) -> Any:
    if store is None:
        raise PersistenceError("A persistent LangGraph Store is required for memory")
    if _inherits_known_memory_type(
        store, {("langgraph.store.memory", "InMemoryStore")}
    ):
        raise PersistenceError("In-memory LangGraph stores are not supported")
    required = ("search", "put")
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if missing:
        raise PersistenceError(
            "Persistent Store is missing required methods: " + ", ".join(missing)
        )
    return store


def configure_default_persistence(
    *, checkpointer: Any = _UNSET, store: Any = _UNSET
) -> PersistenceConfig:
    """Set process-local persistent defaults; pass ``None`` to clear one."""
    global _DEFAULTS
    if checkpointer is _UNSET and store is _UNSET:
        raise ValueError("Provide checkpointer and/or store")
    resolved_checkpointer = _DEFAULTS.checkpointer
    resolved_store = _DEFAULTS.store
    if checkpointer is not _UNSET:
        resolved_checkpointer = (
            None if checkpointer is None else validate_checkpointer(checkpointer)
        )
    if store is not _UNSET:
        resolved_store = None if store is None else validate_store(store)
    _DEFAULTS = PersistenceConfig(resolved_checkpointer, resolved_store)
    return _DEFAULTS


def default_persistence() -> PersistenceConfig:
    return _DEFAULTS
