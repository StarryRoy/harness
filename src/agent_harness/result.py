"""Stable application result types."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentResult(Mapping[str, Any]):
    """Application-facing result with opt-in access to the complete graph state."""

    output: Any
    status: str
    session_id: str
    structured_response: Any = None
    hitl: tuple[Any, ...] = ()
    plan: Mapping[str, Any] | None = None
    state: Mapping[str, Any] = field(default_factory=dict, repr=False)

    # Mapping compatibility makes migration from the former raw-state return gradual.
    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.state)

    def __len__(self) -> int:
        return len(self.state)

    @property
    def completed(self) -> bool:
        return self.status == "completed"
