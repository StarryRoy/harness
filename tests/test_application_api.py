from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_harness import (
    AgentResult,
    LexicalSkillSelector,
    Skill,
    SkillMetadata,
    SkillRegistry,
    create_agent,
)


def test_result_exposes_generated_session_and_supports_session_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "AGENT_HARNESS_CHECKPOINT_PATH", str(tmp_path / "checkpoints.pkl")
    )
    agent = create_agent(
        name="application-result",
        instructions="Answer.",
        model=FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
    )

    result = agent.invoke("work")

    assert isinstance(result, AgentResult)
    assert result.output == "done"
    assert result.status == "completed"
    assert result.session_id
    assert result["messages"][-1].content == "done"
    agent.clear_session(session_id=result.session_id)
    assert not agent.runtime.graph.get_state(
        agent.runtime._config(None, result.session_id)[0]
    ).values


def test_registry_accepts_replaceable_skill_selector():
    class ReverseSelector(LexicalSkillSelector):
        def select(self, query, skills, limit):
            selected = tuple(reversed(skills))
            return selected[:limit] if limit is not None else selected

    registry = SkillRegistry(ReverseSelector())
    registry.register(Skill(SkillMetadata("first", "First"), "one"))
    registry.register(Skill(SkillMetadata("second", "Second"), "two"))

    assert [item["name"] for item in registry.summaries(query="anything")] == [
        "second",
        "first",
    ]
