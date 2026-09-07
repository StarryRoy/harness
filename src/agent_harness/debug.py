"""Deliberately minimal, replaceable debug output."""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DebugHandler:
    enabled: bool = False
    format: str = "print"

    def __post_init__(self) -> None:
        if self.format not in {"print", "json", "md"}:
            raise ValueError("debug format must be 'print', 'json', or 'md'")

    def emit(self, event: str, **details: Any) -> None:
        if not self.enabled:
            return
        if self.format == "json":
            print(json.dumps({"event": event, **details}, default=str, ensure_ascii=False))
            return
        suffix = " ".join(f"{key}={value!r}" for key, value in details.items())
        prefix = f"**{event}**" if self.format == "md" else event
        print(f"{prefix}{' ' + suffix if suffix else ''}")
