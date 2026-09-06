"""Convenience factory for constructing a complete agent."""

from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .agent import Agent
from .definition import AgentDefinition, RuntimeConfig
from .runtime import AgentRuntime
from .skills import Skill, SkillLoader, SkillRegistry
from .strategy import AgentStrategy, ReActStrategy


def create_agent(
    *,
    name: str,
    description: str,
    model: BaseChatModel,
    instructions: str,
    tools: Sequence[BaseTool] | None = None,
    skills: Sequence[Skill | str | Path] | None = None,
    response_format: Any | None = None,
    runtime_config: RuntimeConfig | None = None,
    strategy: AgentStrategy | None = None,
) -> Agent:
    """Validate configuration, compile the graph, and return an Agent."""
    tools = tuple(tools or ())
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

    definition = AgentDefinition(
        name=name,
        description=description,
        model=model,
        instructions=instructions,
        tools=tools,
        skills=tuple(resolved_skills),
        response_format=response_format,
        runtime_config=runtime_config or RuntimeConfig(),
    )
    runtime = AgentRuntime(definition, strategy or ReActStrategy(), registry)
    return Agent(definition, runtime)
