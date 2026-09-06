"""Deliberately minimal, replaceable debug output."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DebugHandler:
    enabled: bool = False

    def emit(self, event: str, **details: Any) -> None:
        if not self.enabled:
            return
        suffix = " ".join(f"{key}={value!r}" for key, value in details.items())
        print(f"{event}{' ' + suffix if suffix else ''}")
