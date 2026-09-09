"""Reusable deterministic test doubles for Harness-level behavior tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import Field

from .observability import BufferedEventSink, RuntimeEvent


class DeterministicFakeModel(FakeMessagesListChatModel):
    """Scripted chat model supporting tools, structured output, sync, and async."""

    bound_tool_names: list[str] = Field(default_factory=list)
    seen_messages: list[list[Any]] = Field(default_factory=list)
    structured_outputs: list[Any] = Field(default_factory=list)
    structured_index: int = 0

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        self.bound_tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def run(_value: Any, config: Any = None) -> Any:
            if self.structured_index >= len(self.structured_outputs):
                raise AssertionError("missing deterministic structured output")
            result = self.structured_outputs[self.structured_index]
            self.structured_index += 1
            return result

        return RunnableLambda(run)

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@dataclass(slots=True)
class FakeTool:
    """Stateful fake that can return a value or fail a fixed number of calls."""

    name: str
    result: Any = None
    failures: int = 0
    description: str = "Deterministic fake tool."
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    def as_tool(self) -> StructuredTool:
        def run(**arguments: Any) -> Any:
            self.calls.append(dict(arguments))
            if len(self.calls) <= self.failures:
                raise RuntimeError(f"{self.name} scripted failure")
            return self.result(arguments) if callable(self.result) else self.result

        return StructuredTool.from_function(
            run, name=self.name, description=self.description
        )


class FakeEventSink(BufferedEventSink):
    """Recording sink with convenience filtering for contract assertions."""

    def by_type(self, event_type: str) -> tuple[RuntimeEvent, ...]:
        return tuple(event for event in self.events if event.event_type == event_type)


@dataclass(slots=True)
class FakeSubAgent:
    """Task-only subagent double for testing delegation boundaries."""

    name: str
    result: Any = "fake subagent result"
    calls: list[str] = field(default_factory=list)

    def as_tool(self) -> StructuredTool:
        def run(task: str) -> dict[str, Any]:
            self.calls.append(task)
            value = self.result(task) if callable(self.result) else self.result
            return {
                "content": value,
                "status": "success",
                "metadata": {"agent": self.name},
                "error": None,
            }

        return StructuredTool.from_function(
            run, name=self.name, description="Deterministic fake subagent."
        )


def fake_tool(
    name: str,
    result: Any = None,
    *,
    failures: int = 0,
    description: str = "Deterministic fake tool.",
) -> tuple[StructuredTool, FakeTool]:
    fake = FakeTool(name, result, failures, description)
    return fake.as_tool(), fake


def sync_async_cases(
    sync_call: Callable[[], Any], async_call: Callable[[], Any]
) -> tuple[Callable[[], Any], Callable[[], Any]]:
    """Keep paired sync/async cases explicit for pytest parametrization."""

    return sync_call, async_call
