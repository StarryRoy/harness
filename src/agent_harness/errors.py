"""Stable, deliberately small public exception hierarchy."""

from __future__ import annotations


class AgentError(Exception):
    """Base error raised at an application boundary.

    The original exception is retained both through Python exception chaining and
    through ``cause`` for callers which cannot inspect ``__cause__``.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ModelError(AgentError):
    pass


class ToolError(AgentError):
    pass


class SkillError(AgentError):
    pass


class SubAgentError(AgentError):
    pass


class SessionError(AgentError):
    pass


class PersistenceError(AgentError):
    pass


class MemoryError(AgentError):
    pass


class StrategyError(AgentError):
    pass


class MiddlewareError(AgentError):
    pass


class HITLError(AgentError):
    pass


class MCPError(AgentError):
    pass
