"""Centralized model-context assembly around LangMem compaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from .skills import SkillRegistry

_DEFAULT_SUMMARY_CONTEXT_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    summary_token_threshold: int | None = None
    summary_keep_recent: int = 12
    max_tool_results: int = 8
    max_tool_result_chars: int = 8_000
    max_skill_candidates: int = 8
    skill_retention_turns: int = 2
    summary_context_fraction: float = _DEFAULT_SUMMARY_CONTEXT_FRACTION
    model_max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            self.summary_token_threshold is not None
            and self.summary_token_threshold < 256
        ):
            raise ValueError("summary_token_threshold must be at least 256")
        if self.summary_keep_recent < 1:
            raise ValueError("summary_keep_recent must be at least 1")
        if self.max_tool_results < 0 or self.max_tool_result_chars < 1:
            raise ValueError("tool context limits are invalid")
        if self.max_skill_candidates < 1 or self.skill_retention_turns < 0:
            raise ValueError("skill context limits are invalid")
        if not 0 < self.summary_context_fraction < 1:
            raise ValueError("summary_context_fraction must be between 0 and 1")
        if (
            self.model_max_input_tokens is not None
            and self.model_max_input_tokens < 256
        ):
            raise ValueError("model_max_input_tokens must be at least 256")

    def resolve_summary_token_threshold(self, model: Any | None = None) -> int:
        """Resolve an explicit threshold or derive one from model metadata."""
        profile = getattr(model, "profile", None)
        profiled_context_window = (
            profile.get("max_input_tokens") if isinstance(profile, Mapping) else None
        )
        context_window = self.model_max_input_tokens or profiled_context_window
        if (
            not isinstance(context_window, int)
            or isinstance(context_window, bool)
            or context_window < 256
        ):
            raise ValueError(
                "The model profile does not provide a valid max_input_tokens; "
                "set ContextPolicy(model_max_input_tokens=...) explicitly"
            )
        if self.summary_token_threshold is not None:
            return self.summary_token_threshold
        return max(256, int(context_window * self.summary_context_fraction))


class AgentContextManager:
    def __init__(
        self,
        instructions: str,
        skills: SkillRegistry,
        policy: ContextPolicy,
        *,
        model: Any | None = None,
    ) -> None:
        self.instructions = instructions
        self.skills = skills
        self.policy = policy
        self.summary_token_threshold = policy.resolve_summary_token_threshold(model)

    def discover(self, value: str | Mapping[str, Any]) -> list[dict[str, str]]:
        if isinstance(value, str):
            query = value
        else:
            messages = value.get("messages", [])
            query = " ".join(
                str(getattr(item, "content", item)) for item in messages[-3:]
            )
        return list(
            self.skills.summaries(query=query, limit=self.policy.max_skill_candidates)
        )

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
        memories = state.get("long_term_memories", [])
        if memories:
            rendered = "\n".join(f"- {item}" for item in memories)
            sections.append("Relevant cross-session memories:\n" + rendered)
        available = state.get("available_skills", [])
        if available:
            catalog = "\n".join(
                f"- {item['name']}: {item['description']}" for item in available
            )
            sections.append(
                "Candidate skills are shown as summaries only. Call load_skill before using one:\n"
                + catalog
            )
        for identifier in state.get("loaded_skills", []):
            sections.append(self.skills.get(identifier).context())
        if business_context:
            rendered = "\n".join(
                f"- {key}: {value}" for key, value in business_context.items()
            )
            sections.append("Business runtime context:\n" + rendered)
        messages = self._bounded_tool_results(self._current_messages(state))
        return [SystemMessage(content="\n\n".join(filter(None, sections))), *messages]

    @staticmethod
    def _current_messages(state: Mapping[str, Any]) -> list[Any]:
        """Return the compacted history followed by everything added since it.

        ``summarized_messages`` is a snapshot produced by LangMem, while
        ``messages`` keeps growing through the graph message reducer.  Treating
        the snapshot as the whole history would hide later model/tool messages.
        """
        history = list(state.get("messages", []))
        compacted = list(state.get("summarized_messages", []))
        if not compacted:
            return history

        history_by_id = {
            message.id: index
            for index, message in enumerate(history)
            if getattr(message, "id", None) is not None
        }
        represented = [
            history_by_id[message.id]
            for message in compacted
            if getattr(message, "id", None) in history_by_id
        ]
        if represented:
            return [*compacted, *history[max(represented) + 1 :]]

        # Messages normally have reducer-assigned IDs.  Equality is a safe
        # fallback for direct ContextManager use with manually-created messages.
        for compacted_message in reversed(compacted):
            for index in range(len(history) - 1, -1, -1):
                if (
                    compacted_message is history[index]
                    or compacted_message == history[index]
                ):
                    return [*compacted, *history[index + 1 :]]

        running_summary = dict(state.get("context", {})).get("running_summary")
        last_id = getattr(running_summary, "last_summarized_message_id", None)
        if last_id in history_by_id:
            return [*compacted, *history[history_by_id[last_id] + 1 :]]
        return compacted

    def _bounded_tool_results(self, messages: list[Any]) -> list[Any]:
        tool_positions = [
            i for i, item in enumerate(messages) if isinstance(item, ToolMessage)
        ]
        retained = (
            set(tool_positions[-self.policy.max_tool_results :])
            if self.policy.max_tool_results
            else set()
        )
        bounded: list[Any] = []
        for index, item in enumerate(messages):
            if not isinstance(item, ToolMessage):
                bounded.append(item)
                continue
            content = str(item.content)
            if index not in retained:
                content = "[older tool result omitted from model context]"
            elif len(content) > self.policy.max_tool_result_chars:
                content = (
                    content[: self.policy.max_tool_result_chars]
                    + "\n[tool result truncated]"
                )
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
        _, _, prompt_tokens = self._context_token_counts(state)
        return prompt_tokens >= self.summary_token_threshold

    def summary_history_token_threshold(self, state: Mapping[str, Any]) -> int:
        """Return the history budget after fixed prompt context is accounted for."""
        _, history_tokens, prompt_tokens = self._context_token_counts(state)
        fixed_prompt_tokens = max(0, prompt_tokens - history_tokens)
        return max(1, self.summary_token_threshold - fixed_prompt_tokens)

    def _context_token_counts(
        self, state: Mapping[str, Any]
    ) -> tuple[list[Any], int, int]:
        runtime_metadata = state.get("runtime_metadata", {})
        business_context = (
            runtime_metadata.get("business_context")
            if isinstance(runtime_metadata, Mapping)
            and isinstance(runtime_metadata.get("business_context"), Mapping)
            else None
        )
        prompt = self.build_messages(state, business_context)
        messages = prompt[1:]
        return (
            messages,
            count_tokens_approximately(messages),
            count_tokens_approximately(prompt),
        )
