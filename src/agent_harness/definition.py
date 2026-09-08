"""Immutable-ish configuration objects used by the runtime."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .context import ContextPolicy
from .middleware import AgentMiddleware
from .skills import Skill


@dataclass(slots=True)
class RuntimeConfig:
    """Small set of execution controls intentionally owned by the harness."""

    max_iterations: int = 12
    debug: bool = False
    debug_format: str = "print"
    retry_attempts: int = 2
    timeout_seconds: float = 60.0
    call_limit: int = 48
    script_timeout_seconds: float = 30.0
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.retry_attempts < 1 or self.call_limit < 1:
            raise ValueError("retry_attempts and call_limit must be at least 1")
        if self.timeout_seconds <= 0 or self.script_timeout_seconds <= 0:
            raise ValueError("timeout values must be positive")
        if self.debug_format not in {"print", "json", "md"}:
            raise ValueError("debug_format must be 'print', 'json', or 'md'")


@dataclass(slots=True)
class AgentDefinition:
    name: str
    description: str
    model: BaseChatModel
    instructions: str
    tools: Sequence[BaseTool] = field(default_factory=tuple)
    skills: Sequence[Skill | str | Path] = field(default_factory=tuple)
    response_format: Any | None = None
    state_schema: type | None = None
    middleware: Sequence[AgentMiddleware] = field(default_factory=tuple)
    subagent_names: tuple[str, ...] = ()
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    store: Any | None = None
    memory: Any | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty")
        if not isinstance(self.model, BaseChatModel):
            raise TypeError("model must be a langchain BaseChatModel")
