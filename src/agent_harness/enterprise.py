"""Small enterprise policies built on the existing middleware/tool runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .middleware import AgentExecution, AgentMiddleware, ModelRequest, ToolRequest
from .observability import EventType

GuardrailAction = Literal["pass", "reject", "modify"]


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    action: GuardrailAction = "pass"
    value: Any = None
    reason: str | None = None


Guardrail = Callable[[Any], GuardrailResult]


class GuardrailMiddleware(AgentMiddleware):
    """Apply optional input, tool and output policies without a rule engine."""

    def __init__(
        self,
        *,
        input: Guardrail | None = None,
        tool: Guardrail | None = None,
        output: Guardrail | None = None,
    ) -> None:
        self.input_guardrail, self.tool_guardrail, self.output_guardrail = (
            input,
            tool,
            output,
        )

    @staticmethod
    def _apply(result: GuardrailResult, original: Any) -> Any:
        if result.action == "reject":
            raise ValueError(result.reason or "Rejected by guardrail")
        return result.value if result.action == "modify" else original

    @staticmethod
    def _evaluate(
        execution: AgentExecution,
        stage: str,
        guardrail: Guardrail,
        subject: Any,
        original: Any,
    ) -> Any:
        result = guardrail(subject)
        if not isinstance(result, GuardrailResult):
            raise TypeError("guardrail must return GuardrailResult")
        if execution.debug is not None:
            details = {
                "stage": stage,
                "action": result.action,
                "reason": result.reason,
            }
            if callable(getattr(execution.debug, "span", None)):
                execution.debug.emit(
                    EventType.GUARDRAIL,
                    status=result.action,
                    metadata={"stage": stage, "reason": result.reason},
                )
            else:
                execution.debug.emit("GUARDRAIL", **details)
        return GuardrailMiddleware._apply(result, original)

    def before_agent(self, execution: AgentExecution) -> None:
        if self.input_guardrail:
            updated = self._evaluate(
                execution,
                "input",
                self.input_guardrail,
                execution.input,
                execution.input,
            )
            if updated is not execution.input:
                execution.input.clear()
                execution.input.update(updated)

    def wrap_tool_call(self, request: ToolRequest, call_next: Any) -> Any:
        if self.tool_guardrail:
            request.arguments = self._evaluate(
                request.execution,
                "tool",
                self.tool_guardrail,
                request,
                request.arguments,
            )
        return call_next(request)

    def after_agent(self, execution: AgentExecution, result: Any) -> Any:
        return (
            self._evaluate(execution, "output", self.output_guardrail, result, result)
            if self.output_guardrail
            else result
        )


class ModelFallbackMiddleware(AgentMiddleware):
    """Try configured fallback chat models after the primary middleware call fails."""

    def __init__(self, models: Sequence[BaseChatModel]) -> None:
        self.models = tuple(models)

    def wrap_model_call(self, request: ModelRequest, call_next: Any) -> Any:
        try:
            return call_next(request)
        except Exception as primary:
            last = primary
            for index, model in enumerate(self.models, start=1):
                if request.execution.debug is not None:
                    details = {
                        "index": index,
                        "model": type(model).__name__,
                        "primary_error": str(primary),
                    }
                    if callable(getattr(request.execution.debug, "span", None)):
                        request.execution.debug.emit(
                            EventType.MODEL_FALLBACK,
                            status="fallback",
                            metadata=details,
                        )
                    else:
                        request.execution.debug.emit("MODEL FALLBACK", **details)
                try:
                    return call_next(request.with_model(model))
                except Exception as exc:  # noqa: BLE001 - continue through configured fallbacks
                    last = exc
            raise last from primary

    async def awrap_model_call(self, request: ModelRequest, call_next: Any) -> Any:
        try:
            return await call_next(request)
        except Exception as primary:
            last = primary
            for index, model in enumerate(self.models, start=1):
                if request.execution.debug is not None:
                    details = {
                        "index": index,
                        "model": type(model).__name__,
                        "primary_error": str(primary),
                    }
                    if callable(getattr(request.execution.debug, "span", None)):
                        request.execution.debug.emit(
                            EventType.MODEL_FALLBACK,
                            status="fallback",
                            metadata=details,
                        )
                    else:
                        request.execution.debug.emit("MODEL FALLBACK", **details)
                try:
                    return await call_next(request.with_model(model))
                except Exception as exc:  # noqa: BLE001 - continue through configured fallbacks
                    last = exc
            raise last from primary


def require_approval(tool: BaseTool, *, message: str | None = None) -> BaseTool:
    """Mark an existing LangChain tool as sensitive (no wrapper runtime)."""
    tool.metadata = {**(tool.metadata or {}), "harness_approval": True}
    if message:
        tool.metadata["harness_approval_message"] = message
    return tool
