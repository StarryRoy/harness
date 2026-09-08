"""Centralized model-context assembly and simple session compaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from .skills import SkillRegistry


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    summary_threshold: int = 40
    summary_keep_recent: int = 12
    max_tool_results: int = 8
    max_tool_result_chars: int = 8_000
    max_skill_candidates: int = 8
    skill_retention_turns: int = 2

    def __post_init__(self) -> None:
        if self.summary_threshold < 2:
            raise ValueError("summary_threshold must be at least 2")
        if not 1 <= self.summary_keep_recent < self.summary_threshold:
            raise ValueError("summary_keep_recent must be between 1 and summary_threshold - 1")
        if self.max_tool_results < 0 or self.max_tool_result_chars < 1:
            raise ValueError("tool context limits are invalid")
        if self.max_skill_candidates < 1 or self.skill_retention_turns < 0:
            raise ValueError("skill context limits are invalid")


class AgentContextManager:
    def __init__(
        self,
        instructions: str,
        skills: SkillRegistry,
        policy: ContextPolicy,
    ) -> None:
        self.instructions = instructions
        self.skills = skills
        self.policy = policy

    def discover(self, value: str | Mapping[str, Any]) -> list[dict[str, str]]:
        if isinstance(value, str):
            query = value
        else:
            messages = value.get("messages", [])
            query = " ".join(str(getattr(item, "content", item)) for item in messages[-3:])
        return list(self.skills.summaries(query=query, limit=self.policy.max_skill_candidates))

    def prepare_turn(self, state: Mapping[str, Any]) -> dict[str, Any]:
        turn = int(state.get("session_turn", 0)) + 1
        skill_state = dict(state.get("skill_state", {}))
        loaded = list(state.get("loaded_skills", []))
        retained_set: set[str] = set()
        for identifier in loaded:
            details = skill_state.get(identifier, {})
            last_used = int(details.get("last_used", turn))
            if turn - last_used <= self.policy.skill_retention_turns:
                retained_set.add(identifier)
        # A retained skill keeps its dependency context alive as well.
        for identifier in tuple(retained_set):
            retained_set.update(self.skills.dependency_identifiers(identifier))
        retained = [identifier for identifier in loaded if identifier in retained_set]
        skill_state = {
            identifier: details
            for identifier, details in skill_state.items()
            if identifier in retained_set
        }
        return {
            "session_turn": turn,
            "loaded_skills": retained,
            "active_skill": None,
            "skill_state": skill_state,
        }

    def build_messages(
        self,
        state: Mapping[str, Any],
        business_context: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        sections = [self.instructions.strip()]
        summary = state.get("summary")
        if summary:
            sections.append(f"Session summary of older messages:\n{summary}")
        memories = state.get("long_term_memories", [])
        if memories:
            rendered = "\n".join(f"- {item}" for item in memories)
            sections.append("Relevant cross-session memories:\n" + rendered)
        available = state.get("available_skills", [])
        if available:
            catalog = "\n".join(f"- {item['name']}: {item['description']}" for item in available)
            sections.append(
                "Candidate skills are shown as summaries only. Call load_skill before using one:\n"
                + catalog
            )
        for identifier in state.get("loaded_skills", []):
            sections.append(self.skills.get(identifier).context())
        if business_context:
            rendered = "\n".join(f"- {key}: {value}" for key, value in business_context.items())
            sections.append("Business runtime context:\n" + rendered)
        messages = self._bounded_tool_results(list(state.get("messages", [])))
        return [SystemMessage(content="\n\n".join(filter(None, sections))), *messages]

    def _bounded_tool_results(self, messages: list[Any]) -> list[Any]:
        tool_positions = [i for i, item in enumerate(messages) if isinstance(item, ToolMessage)]
        retained = set(tool_positions[-self.policy.max_tool_results :]) if self.policy.max_tool_results else set()
        bounded: list[Any] = []
        for index, item in enumerate(messages):
            if not isinstance(item, ToolMessage):
                bounded.append(item)
                continue
            content = str(item.content)
            if index not in retained:
                content = "[older tool result omitted from model context]"
            elif len(content) > self.policy.max_tool_result_chars:
                content = content[: self.policy.max_tool_result_chars] + "\n[tool result truncated]"
            bounded.append(
                ToolMessage(
                    content=content,
                    tool_call_id=item.tool_call_id,
                    name=item.name,
                    id=item.id,
                    status=getattr(item, "status", "success"),
                )
            )
        return bounded

    def needs_summary(self, state: Mapping[str, Any]) -> bool:
        return len(state.get("messages", [])) >= self.policy.summary_threshold

    def split_for_summary(self, state: Mapping[str, Any]) -> tuple[list[BaseMessage], list[BaseMessage]]:
        messages = list(state.get("messages", []))
        split = max(0, len(messages) - self.policy.summary_keep_recent)
        # Do not retain an orphaned tool response without its AI tool call.
        while split > 0 and split < len(messages) and isinstance(messages[split], ToolMessage):
            split -= 1
        return messages[:split], messages[split:]

    @staticmethod
    def summary_prompt(previous: str | None, messages: list[BaseMessage]) -> list[Any]:
        transcript = "\n".join(
            f"{getattr(message, 'type', type(message).__name__)}: {message.content}"
            for message in messages
        )
        prior = f"Existing summary:\n{previous}\n\n" if previous else ""
        return [
            SystemMessage(
                content=(
                    "Create a concise factual session summary. Preserve user preferences, decisions, "
                    "unresolved work, and facts needed in later turns. Do not add facts."
                )
            ),
            SystemMessage(content=prior + "Messages to summarize:\n" + transcript),
        ]
