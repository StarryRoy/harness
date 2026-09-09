"""Backward-compatible debug sink built on the stable event contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .observability import ConsoleEventSink, RuntimeEvent


@dataclass(slots=True)
class DebugHandler:
    """A lightweight EventSink enabled by ``RuntimeConfig.debug``.

    Runtime code sends structured events.  The legacy string form remains
    accepted for direct callers while they migrate.
    """

    enabled: bool = False
    format: str = "print"
    _sink: ConsoleEventSink = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.format not in {"print", "json", "md"}:
            raise ValueError("debug format must be 'print', 'json', or 'md'")
        self._sink = ConsoleEventSink(
            format="text" if self.format == "print" else self.format
        )

    def emit(self, event: RuntimeEvent | str, **details: Any) -> None:
        if not self.enabled:
            return
        if not isinstance(event, RuntimeEvent):
            event = RuntimeEvent(str(event), metadata=details)
        self._sink.emit(event)
