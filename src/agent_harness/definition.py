"""Immutable-ish configuration objects used by the runtime."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .skills import Skill


@dataclass(slots=True)
class RuntimeConfig:
    """Small set of execution controls intentionally owned by the harness."""

    max_iterations: int = 12
    debug: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


@dataclass(slots=True)
class AgentDefinition:
    name: str
    description: str
    model: BaseChatModel
    instructions: str
    tools: Sequence[BaseTool] = field(default_factory=tuple)
    skills: Sequence[Skill | str | Path] = field(default_factory=tuple)
    response_format: Any | None = None
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty")
        if not isinstance(self.model, BaseChatModel):
            raise TypeError("model must be a langchain BaseChatModel")
