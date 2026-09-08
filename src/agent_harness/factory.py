"""Convenience factory for constructing a complete isolated agent."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .agent import Agent
from .definition import AgentDefinition, RuntimeConfig
from .middleware import AgentMiddleware, default_middleware
from .enterprise import GuardrailMiddleware, ModelFallbackMiddleware
from .memory import LongTermMemory, MemoryConfig
from .runtime import AgentRuntime
from .skills import Skill, SkillLoader, SkillRegistry
from .strategy import AgentStrategy, ReActStrategy

_DEFAULT_MODEL: BaseChatModel | None = None


def configure_default_model(model: BaseChatModel) -> None:
    """Set the process-local model used when create_agent omits ``model``."""
    global _DEFAULT_MODEL
    if not isinstance(model, BaseChatModel):
        raise TypeError("model must be a langchain BaseChatModel")
    _DEFAULT_MODEL = model


def _resolve_model(model: BaseChatModel | str | None) -> BaseChatModel:
    if model is None and _DEFAULT_MODEL is not None:
        return _DEFAULT_MODEL
    requested = model or os.getenv("AGENT_HARNESS_MODEL")
    if isinstance(requested, BaseChatModel):
        return requested
    if isinstance(requested, str) and requested.strip():
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise RuntimeError(
                "String/default model initialization needs the optional 'langchain' package and "
                "the selected provider integration; otherwise pass a BaseChatModel instance"
            ) from exc
        resolved = init_chat_model(requested.strip())
        if not isinstance(resolved, BaseChatModel):
            raise TypeError("Initialized model is not a langchain BaseChatModel")
        return resolved
    raise ValueError(
        "No model configured. Pass model=..., call configure_default_model(...), or set "
        "AGENT_HARNESS_MODEL with the optional LangChain provider installed."
    )


def create_agent(
    *,
    name: str,
    instructions: str,
    description: str = "",
    model: BaseChatModel | str | None = None,
    tools: Sequence[BaseTool] | None = None,
    skills: Sequence[Skill | str | Path] | None = None,
    subagents: Sequence[Agent] | None = None,
    response_format: Any | None = None,
    state_schema: type | None = None,
    runtime_config: RuntimeConfig | None = None,
    strategy: AgentStrategy | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    middleware_mode: str = "extend",
    checkpointer: Any | None = None,
    session_namespace: str | None = None,
    store: Any | None = None,
    memory: MemoryConfig | bool | None = None,
    memory_schema: type | None = None,
    guardrail: GuardrailMiddleware | None = None,
    fallback: Sequence[BaseChatModel] | None = None,
) -> Agent:
    """Validate configuration, compile the graph, and return an Agent."""
    config = runtime_config or RuntimeConfig()
    if middleware_mode not in {"replace", "extend"}:
        raise ValueError("middleware_mode must be 'replace' or 'extend'")
    defaults = default_middleware(
        retry_attempts=config.retry_attempts,
        timeout_seconds=config.timeout_seconds,
        call_limit=config.call_limit,
    )
    # The first middleware is the outermost wrapper.  Fallback must therefore
    # surround retry so it only runs after the primary model exhausts retries.
    fallback_middleware = (ModelFallbackMiddleware(fallback),) if fallback else ()
    additions = tuple(item for item in (guardrail,) if item is not None)
    if middleware is None:
        resolved_middleware = (*fallback_middleware, *defaults, *additions)
    elif middleware_mode == "extend":
        resolved_middleware = (*fallback_middleware, *defaults, *middleware, *additions)
    else:
        resolved_middleware = (*fallback_middleware, *middleware, *additions)

    subagents = tuple(subagents or ())
    duplicate_subagents = {
        agent.definition.name
        for agent in subagents
        if sum(item.definition.name == agent.definition.name for item in subagents) > 1
    }
    if duplicate_subagents:
        raise ValueError(f"Duplicate subagent names: {', '.join(sorted(duplicate_subagents))}")
    tools = (*tuple(tools or ()), *(agent.as_tool() for agent in subagents))
    duplicate_tools = {tool.name for tool in tools if sum(t.name == tool.name for t in tools) > 1}
    if duplicate_tools:
        raise ValueError(f"Duplicate tool names: {', '.join(sorted(duplicate_tools))}")

    loader = SkillLoader()
    registry = SkillRegistry()
    resolved_skills = []
    for item in skills or ():
        skill = item if isinstance(item, Skill) else loader.load(item)
        registry.register(skill)
        resolved_skills.append(skill)
    registry.validate()

    resolved_model = _resolve_model(model)
    memory_config: MemoryConfig | None
    if memory is True:
        memory_config = MemoryConfig(schema=memory_schema)
    elif isinstance(memory, MemoryConfig):
        memory_config = memory
        if memory_schema is not None:
            memory_config.schema = memory_schema
    elif memory is None:
        memory_config = None
    else:
        raise TypeError("memory must be MemoryConfig, True, or None")
    resolved_store = store if store is not None else (InMemoryStore() if memory_config else None)
    memory_runtime = (
        LongTermMemory(resolved_store, memory_config, resolved_model) if memory_config else None
    )

    definition = AgentDefinition(
        name=name,
        description=description,
        model=resolved_model,
        instructions=instructions,
        tools=tools,
        skills=tuple(resolved_skills),
        response_format=response_format,
        state_schema=state_schema,
        middleware=resolved_middleware,
        subagent_names=tuple(agent.definition.name for agent in subagents),
        runtime_config=config,
        store=resolved_store,
        memory=memory_config,
    )
    runtime = AgentRuntime(
        definition,
        strategy or ReActStrategy(),
        registry,
        checkpointer if checkpointer is not None else MemorySaver(),
        session_namespace=session_namespace,
        memory=memory_runtime,
    )
    return Agent(definition, runtime)
