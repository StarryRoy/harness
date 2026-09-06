"""Shared LangGraph state schema."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration: int
    runtime_metadata: dict[str, Any]
    available_skills: list[dict[str, str]]
    loaded_skills: list[str]
    active_skill: str | None
    structured_response: Any
