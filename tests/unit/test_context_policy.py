from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent_harness import AgentContextManager, ContextPolicy, SkillRegistry


def _manager(policy: ContextPolicy, model=None) -> AgentContextManager:
    return AgentContextManager(
        "System instructions", SkillRegistry(), policy, model=model
    )


def test_default_summary_threshold_uses_sixty_percent_of_model_context_window():
    policy = ContextPolicy()
    model = SimpleNamespace(profile={"max_input_tokens": 128_000})

    manager = _manager(policy, model)

    assert policy.summary_token_threshold is None
    assert manager.summary_token_threshold == 76_800


def test_missing_model_profile_requires_explicit_max_input_tokens():
    with pytest.raises(ValueError, match="model_max_input_tokens"):
        _manager(ContextPolicy(), SimpleNamespace())

    assert (
        _manager(
            ContextPolicy(model_max_input_tokens=100_000),
            SimpleNamespace(),
        ).summary_token_threshold
        == 60_000
    )


def test_explicit_summary_threshold_still_requires_model_context_size():
    with pytest.raises(ValueError, match="model_max_input_tokens"):
        _manager(ContextPolicy(summary_token_threshold=48_000), SimpleNamespace())

    assert (
        _manager(
            ContextPolicy(summary_token_threshold=48_000),
            SimpleNamespace(profile={"max_input_tokens": 128_000}),
        ).summary_token_threshold
        == 48_000
    )


def test_summary_trigger_depends_only_on_prompt_tokens():
    manager = _manager(ContextPolicy(model_max_input_tokens=100_000))
    many_small_messages = [HumanMessage(content="ok") for _ in range(400)]
    large_message = HumanMessage(content="x" * 250_000)

    assert count_tokens_approximately(many_small_messages) < 60_000
    assert not manager.needs_summary({"messages": many_small_messages})
    assert manager.needs_summary({"messages": [large_message]})
