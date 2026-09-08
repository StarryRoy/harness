"""Harness-owned state plus safe composition with an application schema."""

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
    skill_state: dict[str, dict[str, Any]]
    session_turn: int
    summary: str
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]
    structured_response: Any
    plan: dict[str, Any]
    step_failed: bool
    pending_tool_calls: list[dict[str, Any]]
    tool_call_index: int
    long_term_memories: list[dict[str, Any]]


HARNESS_STATE_FIELDS = frozenset(AgentState.__annotations__)


def compose_state_schema(business_schema: type | None) -> type:
    if business_schema is None:
        return AgentState
    fields = getattr(business_schema, "__annotations__", None)
    if not isinstance(fields, dict):
        raise TypeError("state_schema must be a TypedDict or annotated class")
    conflicts = sorted(HARNESS_STATE_FIELDS.intersection(fields))
    if conflicts:
        raise ValueError(
            "Business state_schema cannot override Harness fields: "
            + ", ".join(conflicts)
        )
    combined = dict(AgentState.__annotations__)
    combined.update(fields)
    return TypedDict(f"Harness{business_schema.__name__}", combined, total=False)
