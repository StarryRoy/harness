"""Replaceable graph-building strategy and the Phase 2 ReAct implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from jsonschema import validate as validate_json_schema
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import TypeAdapter

from .context import AgentContextManager
from .debug import DebugHandler
from .definition import AgentDefinition
from .errors import AgentError, ModelError, ToolError
from .middleware import MiddlewarePipeline, ModelRequest, ToolRequest
from .observability import EventType
from .skills import SkillError, SkillRegistry, SkillScriptRunner
from .state import compose_state_schema


@dataclass(frozen=True, slots=True)
class _StructuredSchema:
    original: Any
    schema: dict[str, Any]
    parameters: dict[str, Any]
    wrapped: bool

    def unwrap(self, value: Any) -> Any:
        if self.wrapped and isinstance(value, Mapping) and "value" in value:
            return value["value"]
        return value


class AgentStrategy(ABC):
    """The narrow extension point between runtime and execution strategy."""

    @abstractmethod
    def build_graph(
        self,
        definition: AgentDefinition,
        skills: SkillRegistry,
        debug: DebugHandler,
        *,
        context: AgentContextManager,
        middleware: MiddlewarePipeline,
        checkpointer: Any = None,
    ) -> Any:
        """Return a compiled LangGraph runnable."""


class _ExecutionStrategySupport:
    """Shared graph execution plumbing; it is not a public strategy."""

    def _build_graph(
        self,
        definition: AgentDefinition,
        skills: SkillRegistry,
        debug: DebugHandler,
        *,
        context: AgentContextManager,
        middleware: MiddlewarePipeline,
        checkpointer: Any = None,
    ) -> Any:
        planning = bool(getattr(self, "_planning", False))
        business_tools = {tool.name: tool for tool in definition.tools}
        internal_tools = self._skill_tools(
            skills,
            set(business_tools),
            SkillScriptRunner(
                definition.runtime_config.script_timeout_seconds,
                env_allowlist=definition.runtime_config.script_env_allowlist,
            ),
            debug,
        )
        overlap = set(business_tools).intersection(tool.name for tool in internal_tools)
        if overlap:
            raise ValueError(
                "Tool names are reserved by the Harness: " + ", ".join(sorted(overlap))
            )
        all_tools: list[BaseTool] = [*definition.tools, *internal_tools]
        executable = {tool.name: tool for tool in all_tools}
        structured_schema = (
            self._normalize_structured_schema(definition.response_format)
            if definition.response_format is not None
            else None
        )
        response_tool = (
            self._structured_response_tool(structured_schema)
            if structured_schema is not None and all_tools and not planning
            else None
        )
        if response_tool is not None and response_tool["function"]["name"] in executable:
            raise ValueError(
                f"Tool name '{response_tool['function']['name']}' is reserved by the Harness"
            )
        agent_tools: list[Any] = [
            *all_tools,
            *([response_tool] if response_tool is not None else []),
        ]
        model = (
            definition.model.bind_tools(
                agent_tools,
                **({"tool_choice": "auto"} if response_tool is not None else {}),
            )
            if agent_tools
            else definition.model
        )
        structured_model = (
            definition.model.with_structured_output(structured_schema.parameters)
            if structured_schema is not None and response_tool is None
            else None
        )

        def prompt(state: Mapping[str, Any]) -> list[Any]:
            execution = middleware.current_execution()
            messages = context.build_messages(state, execution.business_context)
            if response_tool is not None:
                messages.insert(
                    1,
                    SystemMessage(
                        content=(
                            "Continue using tools for as many rounds as needed. When the task is "
                            f"complete, call {response_tool['function']['name']} exactly once with "
                            "the final structured response. Do not return the final answer as text."
                        )
                    ),
                )
            if planning and state.get("plan"):
                plan = state["plan"]
                current = int(plan.get("current_step", 0))
                steps = plan.get("steps", [])
                directive = (
                    "Execute only the current plan step. Use tools when useful and give a concise "
                    "step result when complete."
                )
                if plan.get("status") in {"synthesizing", "partial", "failed"}:
                    directive = (
                        "Synthesize the completed plan results into the final answer."
                    )
                elif current < len(steps):
                    directive += f"\nCurrent step: {steps[current]['description']}"
                messages.insert(1, SystemMessage(content=directive))
            return messages

        def prepare_node(state: Mapping[str, Any]) -> dict[str, Any]:
            return {
                **context.prepare_turn(state),
                "structured_output_complete": False,
            }

        def prepare_route(
            state: Mapping[str, Any],
        ) -> Literal["summarize", "planner", "model"]:
            if context.needs_summary(state):
                return "summarize"
            return "planner" if planning else "model"

        def call_model(
            target: Any,
            state: Mapping[str, Any],
            config: RunnableConfig,
            messages: list[Any],
            purpose: str,
            response_format: Any | None = None,
            tools: tuple[Any, ...] = (),
            tool_choice: Any | None = None,
        ) -> Any:
            request = ModelRequest(
                middleware.current_execution(),
                state,
                messages,
                config,
                purpose=purpose,
                tools=tools,
                response_format=(
                    self._normalize_structured_schema(response_format).parameters
                    if response_format is not None
                    else None
                ),
                tool_choice=tool_choice,
            )
            try:
                return middleware.model(
                    request, lambda req: target.invoke(req.messages, req.config)
                )
            except AgentError:
                raise
            except Exception as exc:
                raise ModelError(f"Model call failed ({purpose})", cause=exc) from exc

        async def acall_model(
            target: Any,
            state: Mapping[str, Any],
            config: RunnableConfig,
            messages: list[Any],
            purpose: str,
            response_format: Any | None = None,
            tools: tuple[Any, ...] = (),
            tool_choice: Any | None = None,
        ) -> Any:
            request = ModelRequest(
                middleware.current_execution(),
                state,
                messages,
                config,
                purpose=purpose,
                tools=tools,
                response_format=(
                    self._normalize_structured_schema(response_format).parameters
                    if response_format is not None
                    else None
                ),
                tool_choice=tool_choice,
            )
            try:
                return await middleware.amodel(
                    request, lambda req: target.ainvoke(req.messages, req.config)
                )
            except AgentError:
                raise
            except Exception as exc:
                raise ModelError(
                    f"Async model call failed ({purpose})", cause=exc
                ) from exc

        def use_structured_model(state: Mapping[str, Any]) -> bool:
            if structured_model is None or state.get("structured_output_complete"):
                return False
            if planning:
                return state.get("plan", {}).get("status") in {
                    "synthesizing",
                    "completed",
                    "partial",
                    "failed",
                }
            return not all_tools

        def model_update(response: Any, iteration: int) -> dict[str, Any]:
            if response_tool is not None:
                parsed = self._parse_structured_response_tool(
                    response, response_tool, structured_schema
                )
                if parsed is not None:
                    return {
                        "structured_response": parsed,
                        "structured_output_complete": True,
                        "iteration": iteration,
                    }
                if not isinstance(response, AIMessage) or not response.tool_calls:
                    raise ModelError(
                        "ReAct model must call a business tool or the structured response tool"
                    )
            return {"messages": [response], "iteration": iteration}

        def model_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            iteration = int(state.get("iteration", 0)) + 1
            structured = use_structured_model(state)
            response = call_model(
                structured_model if structured else model,
                state,
                config,
                prompt(state),
                "agent",
                definition.response_format if structured else None,
                tuple(agent_tools) if not structured else (),
                "auto" if response_tool is not None and not structured else None,
            )
            if structured:
                return {
                    "structured_response": self._validate_structured_result(
                        response, structured_schema
                    ),
                    "structured_output_complete": True,
                    "iteration": iteration,
                }
            return model_update(response, iteration)

        async def async_model_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            iteration = int(state.get("iteration", 0)) + 1
            structured = use_structured_model(state)
            response = await acall_model(
                structured_model if structured else model,
                state,
                config,
                prompt(state),
                "agent",
                definition.response_format if structured else None,
                tuple(agent_tools) if not structured else (),
                "auto" if response_tool is not None and not structured else None,
            )
            if structured:
                return {
                    "structured_response": self._validate_structured_result(
                        response, structured_schema
                    ),
                    "structured_output_complete": True,
                    "iteration": iteration,
                }
            return model_update(response, iteration)

        try:
            from langchain_core.messages.utils import count_tokens_approximately
            from langmem.short_term import asummarize_messages, summarize_messages
        except ImportError as exc:
            from .errors import MemoryError

            raise MemoryError(
                "Short-term summarization requires the 'langmem' package", cause=exc
            ) from exc

        summary_max_tokens = context.summary_token_threshold + max(
            context.policy.summary_keep_recent * 256, 512
        )
        summary_max_summary_tokens = max(context.policy.summary_keep_recent * 64, 128)

        def summary_input(
            state: Mapping[str, Any],
        ) -> tuple[list[Any], dict[str, Any], Any, int]:
            messages = list(state.get("messages", []))
            summary_context = dict(state.get("context", {}))
            running_summary = summary_context.get("running_summary")
            threshold = context.summary_history_token_threshold(state)
            return messages, summary_context, running_summary, threshold

        def summary_update(
            result: Any, summary_context: dict[str, Any]
        ) -> dict[str, Any]:
            update = {"summarized_messages": result.messages}
            if result.running_summary is not None:
                update["summary"] = result.running_summary.summary
                update["context"] = {
                    **summary_context,
                    "running_summary": result.running_summary,
                }
            return update

        def summarize_node(state: Mapping[str, Any]) -> dict[str, Any]:
            messages, summary_context, running_summary, threshold = summary_input(state)
            with (
                debug.span("context.summarize", message_count=len(messages)),
                debug.span("model", purpose="summary"),
            ):
                result = summarize_messages(
                    messages,
                    running_summary=running_summary,
                    model=definition.model,
                    max_tokens=summary_max_tokens,
                    max_tokens_before_summary=threshold,
                    max_summary_tokens=summary_max_summary_tokens,
                    token_counter=count_tokens_approximately,
                )
            debug.emit(
                EventType.CONTEXT_SUMMARIZE,
                status="success",
                metadata={"message_count": len(messages)},
            )
            return summary_update(result, summary_context)

        async def async_summarize_node(
            state: Mapping[str, Any],
        ) -> dict[str, Any]:
            messages, summary_context, running_summary, threshold = summary_input(state)
            with (
                debug.span("context.summarize", message_count=len(messages)),
                debug.span("model", purpose="summary"),
            ):
                result = await asummarize_messages(
                    messages,
                    running_summary=running_summary,
                    model=definition.model,
                    max_tokens=summary_max_tokens,
                    max_tokens_before_summary=threshold,
                    max_summary_tokens=summary_max_summary_tokens,
                    token_counter=count_tokens_approximately,
                )
            debug.emit(
                EventType.CONTEXT_SUMMARIZE,
                status="success",
                metadata={"message_count": len(messages)},
            )
            return summary_update(result, summary_context)

        def route(
            state: Mapping[str, Any],
        ) -> Literal[
            "tools", "replan", "advance", "fail", "finalize", "done"
        ]:
            if state.get("structured_output_complete"):
                return "finalize" if planning else "done"
            message = state["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                if (
                    int(state.get("iteration", 0))
                    >= definition.runtime_config.max_iterations
                ):
                    raise RuntimeError(
                        f"Agent exceeded max_iterations={definition.runtime_config.max_iterations}"
                    )
                return "tools"
            if planning and state.get("plan", {}).get("status") not in {
                "synthesizing",
                "completed",
                "partial",
                "failed",
            }:
                plan = state["plan"]
                content = str(getattr(message, "content", "")).strip().lower()
                inadequate = (
                    bool(state.get("step_failed"))
                    or not content
                    or content.startswith("error:")
                )
                if inadequate and int(plan.get("replan_count", 0)) < self.max_replans:
                    return "replan"
                if inadequate:
                    return "fail"
                return "advance"
            if planning:
                return "finalize"
            return "done"

        def advance_node(state: Mapping[str, Any]) -> dict[str, Any]:
            plan = dict(state["plan"])
            steps = [dict(step) for step in plan["steps"]]
            index = int(plan["current_step"])
            message = state["messages"][-1]
            steps[index]["status"] = "completed"
            steps[index]["result"] = str(message.content)
            index += 1
            plan.update(steps=steps, current_step=index)
            if index >= len(steps):
                plan["status"] = "synthesizing"
            debug.emit(
                EventType.PLAN_STEP,
                status=plan["status"],
                metadata={"current_step": index, "plan": plan},
            )
            return {"plan": plan, "step_failed": False}

        def fail_step_node(state: Mapping[str, Any]) -> dict[str, Any]:
            plan = dict(state["plan"])
            steps = [dict(step) for step in plan["steps"]]
            index = int(plan["current_step"])
            message = state["messages"][-1]
            failure = str(getattr(message, "content", "")).strip()
            steps[index]["status"] = "failed"
            steps[index]["result"] = failure or "Step failed without a result"
            for step in steps[index + 1 :]:
                step["status"] = "skipped"
            plan.update(
                steps=steps,
                current_step=len(steps),
                status=(
                    "partial"
                    if any(step["status"] == "completed" for step in steps[:index])
                    else "failed"
                ),
            )
            debug.emit(
                EventType.PLAN_FAIL,
                status=plan["status"],
                metadata={"current_step": index, "plan": plan},
            )
            return {"plan": plan, "step_failed": False}

        def tool_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            return self._execute_tools(
                state,
                executable,
                set(business_tools),
                set(definition.subagent_names),
                skills,
                middleware,
                config,
                debug,
            )

        async def async_tool_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            return await self._aexecute_tools(
                state,
                executable,
                set(business_tools),
                set(definition.subagent_names),
                skills,
                middleware,
                config,
                debug,
            )

        graph = StateGraph(compose_state_schema(definition.state_schema))
        graph.add_node("prepare", prepare_node)
        graph.add_node(
            "summarize", RunnableLambda(summarize_node, async_summarize_node)
        )
        graph.add_node("model", RunnableLambda(model_node, async_model_node))
        graph.add_node("tools", RunnableLambda(tool_node, async_tool_node))
        if planning:

            def finalize_node(state: Mapping[str, Any]) -> dict[str, Any]:
                plan = dict(state["plan"])
                if plan.get("status") == "synthesizing":
                    plan["status"] = "completed"
                if plan.get("status") == "completed":
                    debug.emit(
                        EventType.PLAN_COMPLETE,
                        status="completed",
                        metadata={"plan": plan},
                    )
                return {"plan": plan}

            graph.add_node("finalize", finalize_node)
        if planning:
            planner = definition.model.with_structured_output(_PlanOutput)

            def plan_node(
                state: Mapping[str, Any], config: RunnableConfig
            ) -> dict[str, Any]:
                request = context.build_messages(
                    state, middleware.current_execution().business_context
                )
                request.insert(
                    1,
                    SystemMessage(
                        content=f"Create a plan of at most {self.max_steps} concrete steps."
                    ),
                )
                value = call_model(
                    planner, state, config, request, "planner", _PlanOutput
                )
                raw_steps = (
                    value.get("steps", [])
                    if isinstance(value, Mapping)
                    else value.steps
                )
                descriptions = [
                    s.get("description", str(s)) if isinstance(s, Mapping) else str(s)
                    for s in raw_steps
                ]
                if not descriptions or len(descriptions) > self.max_steps:
                    raise ValueError(f"Planner must return 1..{self.max_steps} steps")
                plan = {
                    "steps": [
                        {
                            "step_id": str(i + 1),
                            "description": item,
                            "status": "pending",
                            "result": None,
                        }
                        for i, item in enumerate(descriptions)
                    ],
                    "current_step": 0,
                    "status": "executing",
                    "replan_count": 0,
                }
                debug.emit(
                    EventType.PLAN_CREATE,
                    status="created",
                    metadata={"plan": plan},
                )
                return {"plan": plan}

            graph.add_node("planner", plan_node)
            graph.add_node("advance", advance_node)
            graph.add_edge("advance", "model")
            graph.add_node("fail", fail_step_node)
            graph.add_edge("fail", "model")

            def replan_node(
                state: Mapping[str, Any], config: RunnableConfig
            ) -> dict[str, Any]:
                plan = dict(state["plan"])
                completed = [
                    dict(step) for step in plan["steps"][: int(plan["current_step"])]
                ]
                remaining_limit = self.max_steps - len(completed)
                request = context.build_messages(
                    state, middleware.current_execution().business_context
                )
                request.insert(
                    1,
                    SystemMessage(
                        content=(
                            "Revise only the unfinished portion of the plan. Preserve completed work. "
                            f"Return 1..{remaining_limit} remaining concrete steps. "
                            f"Completed steps and results: {completed}. Failed step: "
                            f"{plan['steps'][int(plan['current_step'])]}."
                        )
                    ),
                )
                value = call_model(
                    planner, state, config, request, "replanner", _PlanOutput
                )
                raw = (
                    value.get("steps", [])
                    if isinstance(value, Mapping)
                    else value.steps
                )
                descriptions = [
                    item.get("description", str(item))
                    if isinstance(item, Mapping)
                    else str(item)
                    for item in raw
                ]
                if not descriptions or len(descriptions) > remaining_limit:
                    raise ValueError(
                        f"Re-planner must return 1..{remaining_limit} steps"
                    )
                remaining = [
                    {
                        "step_id": str(len(completed) + i + 1),
                        "description": item,
                        "status": "pending",
                        "result": None,
                    }
                    for i, item in enumerate(descriptions)
                ]
                plan.update(
                    steps=[*completed, *remaining],
                    current_step=len(completed),
                    status="executing",
                    replan_count=int(plan["replan_count"]) + 1,
                )
                debug.emit(
                    EventType.PLAN_REPLAN,
                    status="replanned",
                    metadata={
                        "plan": plan,
                        "replan_count": plan["replan_count"],
                    },
                )
                return {"plan": plan, "step_failed": False}

            graph.add_node("replan", replan_node)
            graph.add_edge("replan", "model")
        graph.add_edge(START, "prepare")
        if planning:
            graph.add_conditional_edges(
                "prepare",
                prepare_route,
                {"summarize": "summarize", "planner": "planner"},
            )
            graph.add_edge("summarize", "planner")
            graph.add_edge("planner", "model")
        else:
            graph.add_conditional_edges(
                "prepare",
                prepare_route,
                {"summarize": "summarize", "model": "model"},
            )
            graph.add_edge("summarize", "model")
        destinations: dict[str, Any] = {
            "tools": "tools",
            "done": END,
        }
        if planning:
            destinations["advance"] = "advance"
            destinations["replan"] = "replan"
            destinations["fail"] = "fail"
            destinations["finalize"] = "finalize"
        graph.add_conditional_edges("model", route, destinations)

        def tool_route(state: Mapping[str, Any]) -> Literal["tools", "model"]:
            return "tools" if state.get("pending_tool_calls") else "model"

        graph.add_conditional_edges(
            "tools", tool_route, {"tools": "tools", "model": "model"}
        )
        if planning:
            graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _response_format_schema(response_format: Any) -> dict[str, Any]:
        if response_format is dict:
            return {"type": "object", "additionalProperties": True}
        if isinstance(response_format, Mapping):
            if response_format.get("type") == "function":
                function = response_format.get("function", {})
                schema = function.get("parameters") if isinstance(function, Mapping) else None
            elif "function" in response_format:
                function = response_format["function"]
                schema = function.get("parameters") if isinstance(function, Mapping) else None
            elif "parameters" in response_format and (
                "name" in response_format or "description" in response_format
            ):
                schema = response_format["parameters"]
            else:
                schema = response_format
            if not isinstance(schema, Mapping):
                raise TypeError("response_format parameters must be a JSON Schema object")
            return dict(schema)
        if isinstance(response_format, type) and is_basemodel_subclass(response_format):
            schema_builder = getattr(response_format, "model_json_schema", None)
            schema = (
                schema_builder()
                if callable(schema_builder)
                else response_format.schema()
            )
            return dict(schema)
        try:
            return dict(TypeAdapter(response_format).json_schema())
        except (TypeError, ValueError, RuntimeError):
            converted = convert_to_openai_tool(response_format)
            return dict(converted["function"]["parameters"])

    @classmethod
    def _normalize_structured_schema(cls, response_format: Any) -> _StructuredSchema:
        schema = cls._response_format_schema(response_format)
        if schema.get("type") == "object":
            schema.setdefault("title", "agent_harness_structured_response")
            return _StructuredSchema(response_format, schema, schema, False)

        parameters: dict[str, Any] = {
            "title": "agent_harness_structured_response",
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
        }
        for key in ("$defs", "definitions"):
            if key in schema:
                parameters[key] = schema[key]
        return _StructuredSchema(response_format, schema, parameters, True)

    @staticmethod
    def _structured_response_tool(
        structured_schema: _StructuredSchema,
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "agent_harness_structured_response",
                "description": "Submit the final response matching the required schema.",
                "parameters": structured_schema.parameters,
            },
        }

    @staticmethod
    def _validate_structured_result(
        value: Any, structured_schema: _StructuredSchema
    ) -> Any:
        value = structured_schema.unwrap(value)
        response_format = structured_schema.original
        try:
            if isinstance(response_format, Mapping):
                validate_json_schema(value, structured_schema.schema)
                return value
            if isinstance(response_format, type) and is_basemodel_subclass(response_format):
                validator = getattr(response_format, "model_validate", None)
                return (
                    validator(value)
                    if callable(validator)
                    else response_format.parse_obj(value)
                )
            try:
                adapter = TypeAdapter(response_format)
            except (TypeError, ValueError, RuntimeError):
                validate_json_schema(value, structured_schema.schema)
                return value
            return adapter.validate_python(value)
        except Exception as exc:
            raise ModelError(
                "Structured response validation failed", cause=exc
            ) from exc

    @classmethod
    def _parse_structured_response_tool(
        cls,
        response: Any,
        response_tool: Mapping[str, Any],
        structured_schema: _StructuredSchema,
    ) -> Any | None:
        if not isinstance(response, AIMessage):
            return None
        name = response_tool["function"]["name"]
        matches = [call for call in response.tool_calls if call.get("name") == name]
        if not matches:
            return None
        if len(matches) != 1 or len(response.tool_calls) != 1:
            raise ModelError(
                "Structured response tool must be the only tool call in the final response"
            )
        arguments = matches[0].get("args", {})
        if not isinstance(arguments, Mapping):
            raise ModelError("Structured response tool arguments must be an object")
        return cls._validate_structured_result(dict(arguments), structured_schema)

    @staticmethod
    def _skill_tools(
        registry: SkillRegistry,
        tool_names: set[str],
        runner: SkillScriptRunner,
        debug: DebugHandler,
    ) -> list[BaseTool]:
        if not registry.list():
            return []

        def load_skill(name: str) -> str:
            """Load a candidate skill and all of its dependencies into agent context."""
            ordered = registry.load_order(name, tool_names)
            debug.emit(
                EventType.SKILL_LOAD,
                status="loaded",
                metadata={
                    "name": name,
                    "skills": [item.identifier for item in ordered],
                    "dependencies": [item.identifier for item in ordered[:-1]],
                },
            )
            return "Loaded skills: " + ", ".join(item.identifier for item in ordered)

        def unload_skill(name: str) -> str:
            """Unload a skill from the current agent context."""
            skill = registry.get(name)
            debug.emit(
                EventType.SKILL_UNLOAD,
                status="unloaded",
                metadata={"name": skill.identifier},
            )
            return f"Skill '{skill.identifier}' unloaded."

        def read_skill_reference(skill: str, name: str) -> str:
            """Read one named UTF-8 reference from an already loaded skill."""
            return registry.get(skill).read_reference(name)

        def read_skill_resource(skill: str, name: str) -> dict[str, Any]:
            """Read one named text or binary resource from an already loaded skill."""
            return registry.get(skill).read_resource(name)

        def read_skill_asset(skill: str, name: str) -> dict[str, Any]:
            """Read one named text or binary asset from an already loaded skill."""
            return registry.get(skill).read_asset(name)

        def run_skill_script(
            skill: str, script: str, arguments: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            """Run a declared skill script with JSON arguments."""
            target = registry.get(skill)
            debug.emit(
                EventType.SKILL_SCRIPT,
                status="started",
                metadata={
                    "skill": target.identifier,
                    "script": script,
                    "arguments": arguments or {},
                },
            )
            result = runner.run(target, script, arguments).as_dict()
            debug.emit(
                EventType.SKILL_SCRIPT,
                status=result["status"],
                metadata={
                    "skill": target.identifier,
                    "script": script,
                    "result": result,
                },
            )
            return result

        tools: list[BaseTool] = [
            StructuredTool.from_function(load_skill, name="load_skill"),
            StructuredTool.from_function(unload_skill, name="unload_skill"),
        ]
        if any(skill.references for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(
                    read_skill_reference, name="read_skill_reference"
                )
            )
        if any(skill.resources for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(
                    read_skill_resource, name="read_skill_resource"
                )
            )
        if any(skill.assets for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(read_skill_asset, name="read_skill_asset")
            )
        if any(skill.script_files for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(run_skill_script, name="run_skill_script")
            )
        return tools

    @staticmethod
    def _result(call: dict[str, Any], value: Any) -> ToolMessage:
        content = (
            value
            if isinstance(value, str)
            else json.dumps(value, default=str, ensure_ascii=False)
        )
        return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])

    @staticmethod
    def _require_loaded(
        call: Mapping[str, Any], loaded: list[str], registry: SkillRegistry
    ) -> None:
        if call["name"] not in {
            "read_skill_reference",
            "read_skill_resource",
            "read_skill_asset",
            "run_skill_script",
        }:
            return
        reference = call["args"].get("skill")
        identifier = registry.get(reference).identifier
        if identifier not in loaded:
            raise SkillError(
                f"Skill '{identifier}' must be loaded before using its assets"
            )

    @staticmethod
    def _skill_state_after(
        call: Mapping[str, Any],
        loaded: list[str],
        skill_state: dict[str, dict[str, Any]],
        registry: SkillRegistry,
        tool_names: set[str],
        turn: int,
    ) -> tuple[list[str], dict[str, dict[str, Any]], str | None]:
        active = loaded[-1] if loaded else None
        if call["name"] == "load_skill":
            ordered = registry.load_order(call["args"]["name"], tool_names)
            for skill in ordered:
                if skill.identifier not in loaded:
                    loaded.append(skill.identifier)
                details = dict(skill_state.get(skill.identifier, {}))
                details.setdefault("loaded_at", turn)
                details["last_used"] = turn
                skill_state[skill.identifier] = details
            active = ordered[-1].identifier
        elif call["name"] == "unload_skill":
            identifier = registry.get(call["args"]["name"]).identifier
            removed = {identifier, *registry.dependents_of(identifier, loaded)}
            loaded = [item for item in loaded if item not in removed]
            for item in removed:
                skill_state.pop(item, None)
            active = loaded[-1] if loaded else None
        elif call["name"] in {
            "read_skill_reference",
            "read_skill_resource",
            "read_skill_asset",
            "run_skill_script",
        }:
            identifier = registry.get(call["args"]["skill"]).identifier
            details = dict(skill_state.get(identifier, {}))
            details["last_used"] = turn
            skill_state[identifier] = details
            active = identifier
        return loaded, skill_state, active

    def _execute_tools(
        self,
        state: Mapping[str, Any],
        tools: dict[str, BaseTool],
        business_tool_names: set[str],
        subagent_names: set[str],
        registry: SkillRegistry,
        middleware: MiddlewarePipeline,
        config: RunnableConfig,
        debug: DebugHandler,
    ) -> dict[str, Any] | Command:
        calls = list(state.get("pending_tool_calls", []))
        index = int(state.get("tool_call_index", 0))
        if not calls:
            calls = list(state["messages"][-1].tool_calls)
            index = 0
        call = calls[index]
        loaded = list(state.get("loaded_skills", []))
        skill_state = dict(state.get("skill_state", {}))
        active = state.get("active_skill")
        any_failed = False
        if call["name"] not in tools:
            raise ToolError(f"Model requested unknown tool: {call['name']}")
        if "mcp" in type(tools[call["name"]]).__module__.lower() or (
            tools[call["name"]].metadata or {}
        ).get("mcp"):
            debug.emit(
                EventType.MCP_TOOL,
                status="started",
                metadata={"name": call["name"]},
            )
        succeeded = False
        execution = middleware.current_execution()
        previous_call_id = execution.metadata.get("current_tool_call_id")
        execution.metadata["current_tool_call_id"] = call["id"]
        try:
            self._require_loaded(call, loaded, registry)
            tool = tools[call["name"]]
            args = self._approved_arguments(tool, call, debug)
            request = ToolRequest(
                execution,
                tool,
                args,
                config,
                call["id"],
                state,
            )
            value = middleware.tool(
                request, lambda req: req.tool.invoke(req.arguments, req.config)
            )
            succeeded = True
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are observations
            any_failed = True
            value = f"Error: {exc}"
        finally:
            if previous_call_id is None:
                execution.metadata.pop("current_tool_call_id", None)
            else:
                execution.metadata["current_tool_call_id"] = previous_call_id
        if isinstance(value, Command):
            return self._advance_tool_command(value, calls, index + 1)
        if succeeded:
            loaded, skill_state, active = self._skill_state_after(
                call,
                loaded,
                skill_state,
                registry,
                business_tool_names,
                int(state.get("session_turn", 1)),
            )
        next_index = index + 1
        update = {
            "messages": [self._result(call, value)],
            "loaded_skills": loaded,
            "active_skill": active,
            "skill_state": skill_state,
            "step_failed": bool(state.get("step_failed")) or any_failed,
            "pending_tool_calls": calls if next_index < len(calls) else [],
            "tool_call_index": next_index if next_index < len(calls) else 0,
        }
        return update

    async def _aexecute_tools(
        self,
        state: Mapping[str, Any],
        tools: dict[str, BaseTool],
        business_tool_names: set[str],
        subagent_names: set[str],
        registry: SkillRegistry,
        middleware: MiddlewarePipeline,
        config: RunnableConfig,
        debug: DebugHandler,
    ) -> dict[str, Any] | Command:
        calls = list(state.get("pending_tool_calls", []))
        index = int(state.get("tool_call_index", 0))
        if not calls:
            calls = list(state["messages"][-1].tool_calls)
            index = 0
        call = calls[index]
        loaded = list(state.get("loaded_skills", []))
        skill_state = dict(state.get("skill_state", {}))
        active = state.get("active_skill")
        any_failed = False
        if call["name"] not in tools:
            raise ToolError(f"Model requested unknown tool: {call['name']}")
        if "mcp" in type(tools[call["name"]]).__module__.lower() or (
            tools[call["name"]].metadata or {}
        ).get("mcp"):
            debug.emit(
                EventType.MCP_TOOL,
                status="started",
                metadata={"name": call["name"]},
            )
        succeeded = False
        execution = middleware.current_execution()
        previous_call_id = execution.metadata.get("current_tool_call_id")
        execution.metadata["current_tool_call_id"] = call["id"]
        try:
            self._require_loaded(call, loaded, registry)
            tool = tools[call["name"]]
            args = self._approved_arguments(tool, call, debug)
            request = ToolRequest(
                execution,
                tool,
                args,
                config,
                call["id"],
                state,
            )
            value = await middleware.atool(
                request, lambda req: req.tool.ainvoke(req.arguments, req.config)
            )
            succeeded = True
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are observations
            any_failed = True
            value = f"Error: {exc}"
        finally:
            if previous_call_id is None:
                execution.metadata.pop("current_tool_call_id", None)
            else:
                execution.metadata["current_tool_call_id"] = previous_call_id
        if isinstance(value, Command):
            return self._advance_tool_command(value, calls, index + 1)
        if succeeded:
            loaded, skill_state, active = self._skill_state_after(
                call,
                loaded,
                skill_state,
                registry,
                business_tool_names,
                int(state.get("session_turn", 1)),
            )
        next_index = index + 1
        update = {
            "messages": [self._result(call, value)],
            "loaded_skills": loaded,
            "active_skill": active,
            "skill_state": skill_state,
            "step_failed": bool(state.get("step_failed")) or any_failed,
            "pending_tool_calls": calls if next_index < len(calls) else [],
            "tool_call_index": next_index if next_index < len(calls) else 0,
        }
        return update

    @staticmethod
    def _advance_tool_command(
        value: Command, calls: list[dict[str, Any]], next_index: int
    ) -> Command:
        command_update = (
            dict(value.update) if isinstance(value.update, Mapping) else {}
        )
        command_update.update(
            {
                "pending_tool_calls": calls if next_index < len(calls) else [],
                "tool_call_index": next_index if next_index < len(calls) else 0,
            }
        )
        return Command(
            graph=value.graph,
            update=command_update,
            resume=value.resume,
            goto=value.goto,
        )

    @staticmethod
    def _validated_arguments(
        tool: BaseTool, call: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate a model Tool Call without invoking the tool.

        ``BaseTool.run`` performs this same parsing immediately before calling
        the tool implementation.  Do it explicitly here so approval interrupts
        are only created for structurally valid calls.
        """
        raw_args = call.get("args")
        if not isinstance(raw_args, Mapping):
            raise TypeError("Tool call arguments must be an object")
        _, parsed_args = tool._to_args_and_kwargs(dict(raw_args), call.get("id"))
        if not isinstance(parsed_args, dict):
            raise TypeError("Tool call arguments must be an object")
        return parsed_args

    @staticmethod
    def _approved_arguments(
        tool: BaseTool, call: Mapping[str, Any], debug: DebugHandler
    ) -> dict[str, Any]:
        metadata = tool.metadata or {}
        if not metadata.get("harness_approval"):
            return dict(call["args"])
        validated_args = _ExecutionStrategySupport._validated_arguments(tool, call)
        payload = {
            "tool": tool.name,
            "args": dict(call["args"]),
            "message": metadata.get("harness_approval_message"),
        }
        execution = MiddlewarePipeline.current_execution()
        replayed = bool(execution.metadata.pop("resume_pending", False))
        if not replayed:
            debug.emit(EventType.HITL_PAUSE, status="paused", metadata=payload)
        decision = interrupt(payload)
        if isinstance(decision, str):
            decision = {"decision": decision}
        action = decision.get("decision", decision.get("action"))
        if action == "approve":
            return validated_args
        if action == "edit" and isinstance(decision.get("args"), Mapping):
            return dict(decision["args"])
        if action == "reject":
            raise ValueError(f"Tool '{tool.name}' was rejected by the user")
        raise ValueError("Invalid HITL resume decision")


class _PlanOutput(TypedDict):
    steps: list[dict[str, str]]


class ReActStrategy(_ExecutionStrategySupport, AgentStrategy):
    """Standard model -> tools -> model loop with centralized context."""

    _planning = False

    def build_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self._build_graph(*args, **kwargs)


class PlanExecuteStrategy(_ExecutionStrategySupport, AgentStrategy):
    """A bounded planner followed by the common execution runtime."""

    _planning = True

    def __init__(self, *, max_steps: int = 8, max_replans: int = 2) -> None:
        if max_steps < 1 or max_replans < 0:
            raise ValueError("max_steps must be positive and max_replans non-negative")
        self.max_steps, self.max_replans = max_steps, max_replans

    def build_graph(self, *args: Any, **kwargs: Any) -> Any:
        return self._build_graph(*args, **kwargs)
