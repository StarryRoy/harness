"""Replaceable graph-building strategy and the Phase 2 ReAct implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from .context import AgentContextManager
from .debug import DebugHandler
from .definition import AgentDefinition
from .errors import ModelError, StrategyError
from .middleware import MiddlewarePipeline, ModelRequest, ToolRequest
from .skills import SkillError, SkillRegistry, SkillScriptRunner
from .state import compose_state_schema


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


class _GraphStrategySupport:
    """Shared graph plumbing; concrete strategies only select their own flow."""

    planning = False

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
        planning = self.planning
        business_tools = {tool.name: tool for tool in definition.tools}
        internal_tools = self._skill_tools(
            skills,
            set(business_tools),
            SkillScriptRunner(definition.runtime_config.script_timeout_seconds),
            debug,
        )
        overlap = set(business_tools).intersection(tool.name for tool in internal_tools)
        if overlap:
            raise ValueError(
                "Tool names are reserved by the Harness: " + ", ".join(sorted(overlap))
            )
        all_tools: list[BaseTool] = [*definition.tools, *internal_tools]
        executable = {tool.name: tool for tool in all_tools}
        model = definition.model.bind_tools(all_tools) if all_tools else definition.model

        def prompt(state: Mapping[str, Any]) -> list[Any]:
            execution = middleware.current_execution()
            messages = context.build_messages(state, execution.business_context)
            if planning and state.get("plan"):
                plan = state["plan"]
                current = int(plan.get("current_step", 0))
                steps = plan.get("steps", [])
                directive = (
                    "Execute only the current plan step. Use tools when useful and give a concise "
                    "step result when complete. If the step cannot be completed or its result is "
                    "clearly insufficient, begin the result with 'REPLAN:'."
                )
                if plan.get("status") == "synthesizing":
                    directive = "Synthesize the completed plan results into the final answer."
                elif current < len(steps):
                    directive += f"\nCurrent step: {steps[current]}"
                messages.insert(1, SystemMessage(content=directive))
            return messages

        def prepare_node(state: Mapping[str, Any]) -> dict[str, Any]:
            return context.prepare_turn(state)

        def prepare_route(state: Mapping[str, Any]) -> Literal["summarize", "model"]:
            return "summarize" if context.needs_summary(state) else "model"

        def call_model(
            target: Any,
            state: Mapping[str, Any],
            config: RunnableConfig,
            messages: list[Any],
            purpose: str,
            model_adapter: Any = None,
        ) -> Any:
            request = ModelRequest(
                middleware.current_execution(), state, messages, config, purpose=purpose,
                model_adapter=model_adapter,
            )
            try:
                return middleware.model(
                    request, lambda req: target.invoke(req.messages, req.config)
                )
            except Exception as exc:
                if isinstance(exc, ModelError):
                    raise
                raise ModelError(f"Model call failed during {purpose}", cause=exc) from exc

        async def acall_model(
            target: Any,
            state: Mapping[str, Any],
            config: RunnableConfig,
            messages: list[Any],
            purpose: str,
            model_adapter: Any = None,
        ) -> Any:
            request = ModelRequest(
                middleware.current_execution(), state, messages, config, purpose=purpose,
                model_adapter=model_adapter,
            )
            try:
                return await middleware.amodel(
                    request, lambda req: target.ainvoke(req.messages, req.config)
                )
            except Exception as exc:
                if isinstance(exc, ModelError):
                    raise
                raise ModelError(f"Model call failed during {purpose}", cause=exc) from exc

        def model_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
            iteration = int(state.get("iteration", 0)) + 1
            debug.emit("MODEL CALL", iteration=iteration)
            adapter = (lambda item: item.bind_tools(all_tools)) if all_tools else (lambda item: item)
            response = call_model(model, state, config, prompt(state), "agent", adapter)
            return {"messages": [response], "iteration": iteration}

        async def async_model_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            iteration = int(state.get("iteration", 0)) + 1
            debug.emit("MODEL CALL", iteration=iteration)
            adapter = (lambda item: item.bind_tools(all_tools)) if all_tools else (lambda item: item)
            response = await acall_model(model, state, config, prompt(state), "agent", adapter)
            return {"messages": [response], "iteration": iteration}

        summarizer: Any = None

        def get_summarizer() -> Any:
            nonlocal summarizer
            if summarizer is None:
                try:
                    from langmem.short_term import SummarizationNode
                except ImportError as exc:
                    raise StrategyError(
                        "Short-term summarization requires the 'langmem' package", cause=exc
                    ) from exc
                policy = definition.runtime_config.context_policy
                retained_tokens = policy.summary_max_tokens or policy.summary_keep_recent * 512
                summarizer = SummarizationNode(
                    model=policy.summarization_model or definition.model,
                    max_tokens=retained_tokens,
                    max_tokens_before_summary=retained_tokens,
                    max_summary_tokens=policy.summary_max_summary_tokens,
                )
            return summarizer

        def summarize_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
            node = get_summarizer()
            if hasattr(node, "invoke"):
                return node.invoke(state, config)
            return node(state)

        async def async_summarize_node(
            state: Mapping[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            node = get_summarizer()
            if hasattr(node, "ainvoke"):
                return await node.ainvoke(state, config)
            return await asyncio.to_thread(node, state)

        def route(state: Mapping[str, Any]) -> Literal[
            "tools", "advance", "replan", "finalize", "format", "done"
        ]:
            message = state["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                if int(state.get("iteration", 0)) >= definition.runtime_config.max_iterations:
                    raise RuntimeError(
                        f"Agent exceeded max_iterations={definition.runtime_config.max_iterations}"
                    )
                return "tools"
            if planning and state.get("plan", {}).get("status") not in {"synthesizing", "completed"}:
                plan = state["plan"]
                content = str(getattr(message, "content", "")).lstrip().upper()
                failed_tool = any(
                    isinstance(item, ToolMessage) and getattr(item, "status", None) == "error"
                    for item in state.get("messages", [])[int(plan.get("step_message_start", 0)) :]
                )
                if (
                    (failed_tool or content.startswith("REPLAN:"))
                    and int(plan.get("replan_count", 0)) < self.max_replans
                ):
                    return "replan"
                return "advance"
            if planning:
                return "finalize"
            return "format" if definition.response_format is not None else "done"

        def advance_node(state: Mapping[str, Any]) -> dict[str, Any]:
            plan = dict(state["plan"])
            steps = [dict(step) for step in plan["steps"]]
            index = int(plan["current_step"])
            message = state["messages"][-1]
            steps[index]["status"] = "completed"
            steps[index]["result"] = str(message.content)
            index += 1
            plan.update(
                steps=steps,
                current_step=index,
                step_message_start=len(state.get("messages", [])),
            )
            if index >= len(steps):
                plan["status"] = "synthesizing"
            debug.emit("PLAN STEP", current_step=index, status=plan["status"])
            return {"plan": plan}

        def tool_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
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
        ) -> dict[str, Any]:
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
        graph.add_node("summarize", RunnableLambda(summarize_node, async_summarize_node))
        graph.add_node("model", RunnableLambda(model_node, async_model_node))
        graph.add_node("tools", RunnableLambda(tool_node, async_tool_node))
        if planning:
            def finalize_node(state: Mapping[str, Any]) -> dict[str, Any]:
                plan = dict(state["plan"])
                plan["status"] = "completed"
                return {"plan": plan}

            graph.add_node("finalize", finalize_node)
        if planning:
            planner = definition.model.with_structured_output(_PlanOutput)

            def plan_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
                request = [
                    *context.build_messages(state, middleware.current_execution().business_context),
                    SystemMessage(
                        content=f"Create a plan of at most {self.max_steps} concrete steps."
                    ),
                ]
                value = call_model(
                    planner, state, config, request, "planner",
                    lambda item: item.with_structured_output(_PlanOutput),
                )
                raw_steps = value.get("steps", []) if isinstance(value, Mapping) else value.steps
                descriptions = [s.get("description", str(s)) if isinstance(s, Mapping) else str(s) for s in raw_steps]
                if not descriptions or len(descriptions) > self.max_steps:
                    raise ValueError(f"Planner must return 1..{self.max_steps} steps")
                plan = {"steps": [{"step_id": str(i + 1), "description": item,
                                    "status": "pending", "result": None}
                                  for i, item in enumerate(descriptions)],
                        "current_step": 0, "status": "executing", "replan_count": 0,
                        "step_message_start": len(state.get("messages", []))}
                debug.emit("PLAN", plan=plan)
                return {"plan": plan}

            graph.add_node("planner", plan_node)
            graph.add_edge(START, "planner")
            graph.add_edge("planner", "prepare")
            graph.add_node("advance", advance_node)
            graph.add_edge("advance", "model")

            def replan_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
                plan = dict(state["plan"])
                completed = [dict(step) for step in plan["steps"] if step["status"] == "completed"]
                remaining_limit = self.max_steps - len(completed)
                if remaining_limit < 1:
                    return {"plan": {**plan, "status": "synthesizing"}}
                results = "\n".join(
                    f"- {step['description']}: {step['result']}" for step in completed
                ) or "- none"
                failure = str(state["messages"][-1].content)
                request = [
                    *context.build_messages(state, middleware.current_execution().business_context),
                    SystemMessage(content=(
                        f"Re-plan only the unfinished work using at most {remaining_limit} steps. "
                        f"Keep these completed results unchanged:\n{results}\n"
                        f"Latest failed or insufficient result:\n{failure}"
                    )),
                ]
                value = call_model(
                    planner, state, config, request, "replan",
                    lambda item: item.with_structured_output(_PlanOutput),
                )
                raw_steps = value.get("steps", []) if isinstance(value, Mapping) else value.steps
                descriptions = [
                    item.get("description", str(item)) if isinstance(item, Mapping) else str(item)
                    for item in raw_steps
                ][:remaining_limit]
                if not descriptions:
                    plan["status"] = "synthesizing"
                    return {"plan": plan}
                offset = len(completed)
                remaining = [
                    {"step_id": str(offset + index + 1), "description": description,
                     "status": "pending", "result": None}
                    for index, description in enumerate(descriptions)
                ]
                plan.update(
                    steps=[*completed, *remaining], current_step=offset,
                    status="executing", replan_count=int(plan["replan_count"]) + 1,
                    step_message_start=len(state.get("messages", [])),
                )
                debug.emit("REPLAN", plan=plan)
                return {"plan": plan}

            graph.add_node("replan", replan_node)
            graph.add_edge("replan", "model")
        else:
            graph.add_edge(START, "prepare")
        graph.add_conditional_edges(
            "prepare", prepare_route, {"summarize": "summarize", "model": "model"}
        )
        graph.add_edge("summarize", "model")
        destinations: dict[str, Any] = {"tools": "tools", "done": END}
        if planning:
            destinations["advance"] = "advance"
            destinations["replan"] = "replan"
            destinations["finalize"] = "finalize"
        if definition.response_format is not None:
            destinations["format"] = "format"
        graph.add_conditional_edges("model", route, destinations)
        graph.add_edge("tools", "model")
        if planning:
            graph.add_edge("finalize", "format" if definition.response_format is not None else END)

        if definition.response_format is not None:
            formatter = definition.model.with_structured_output(definition.response_format)

            def format_node(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
                result = call_model(
                    formatter, state, config, prompt(state), "format",
                    lambda item: item.with_structured_output(definition.response_format),
                )
                return {"structured_response": result}

            async def async_format_node(
                state: Mapping[str, Any], config: RunnableConfig
            ) -> dict[str, Any]:
                result = await acall_model(
                    formatter, state, config, prompt(state), "format",
                    lambda item: item.with_structured_output(definition.response_format),
                )
                return {"structured_response": result}

            graph.add_node("format", RunnableLambda(format_node, async_format_node))
            graph.add_edge("format", END)
        return graph.compile(checkpointer=checkpointer)


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
            debug.emit("SKILL LOAD", name=name, dependencies=[item.identifier for item in ordered[:-1]])
            return "Loaded skills: " + ", ".join(item.identifier for item in ordered)

        def unload_skill(name: str) -> str:
            """Unload a skill from the current agent context."""
            skill = registry.get(name)
            return f"Skill '{skill.identifier}' unloaded."

        def read_skill_reference(skill: str, name: str) -> str:
            """Read one named UTF-8 reference from an already loaded skill."""
            return registry.get(skill).read_reference(name)

        def read_skill_resource(skill: str, name: str) -> dict[str, Any]:
            """Read one named text or binary resource from an already loaded skill."""
            return registry.get(skill).read_resource(name)

        def run_skill_script(
            skill: str, script: str, arguments: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            """Run a declared skill script with JSON arguments."""
            target = registry.get(skill)
            debug.emit("SKILL SCRIPT", skill=target.identifier, script=script)
            return runner.run(target, script, arguments).as_dict()

        tools: list[BaseTool] = [
            StructuredTool.from_function(load_skill, name="load_skill"),
            StructuredTool.from_function(unload_skill, name="unload_skill"),
        ]
        if any(skill.references for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(read_skill_reference, name="read_skill_reference")
            )
        if any(skill.resources for skill in registry.list()):
            tools.append(
                StructuredTool.from_function(read_skill_resource, name="read_skill_resource")
            )
        if any(skill.script_files for skill in registry.list()):
            tools.append(StructuredTool.from_function(run_skill_script, name="run_skill_script"))
        return tools

    @staticmethod
    def _result(call: dict[str, Any], value: Any, *, status: str = "success") -> ToolMessage:
        content = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        return ToolMessage(
            content=content, tool_call_id=call["id"], name=call["name"], status=status
        )

    @staticmethod
    def _require_loaded(
        call: Mapping[str, Any], loaded: list[str], registry: SkillRegistry
    ) -> None:
        if call["name"] not in {
            "read_skill_reference",
            "read_skill_resource",
            "run_skill_script",
        }:
            return
        reference = call["args"].get("skill")
        identifier = registry.get(reference).identifier
        if identifier not in loaded:
            raise SkillError(f"Skill '{identifier}' must be loaded before using its assets")

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
    ) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        loaded = list(state.get("loaded_skills", []))
        skill_state = dict(state.get("skill_state", {}))
        active = state.get("active_skill")
        results = []
        for call in calls:
            if call["name"] not in tools:
                raise ValueError(f"Model requested unknown tool: {call['name']}")
            if call["name"] in subagent_names:
                debug.emit("SUBAGENT CALL", name=call["name"], task=call["args"].get("task"))
            if (tools[call["name"]].metadata or {}).get("harness_mcp"):
                debug.emit("MCP TOOL", name=call["name"])
            debug.emit("TOOL CALL", name=call["name"])
            succeeded = False
            try:
                self._require_loaded(call, loaded, registry)
                args = self._approved_arguments(tools[call["name"]], call, debug)
                request = ToolRequest(
                    middleware.current_execution(),
                    tools[call["name"]],
                    args,
                    config,
                    call["id"],
                )
                value = middleware.tool(
                    request, lambda req: req.tool.invoke(req.arguments, req.config)
                )
                succeeded = True
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - tool failures are observations
                debug.emit("ERROR", error=str(exc))
                value = f"Error: {exc}"
            if succeeded:
                loaded, skill_state, active = self._skill_state_after(
                    call,
                    loaded,
                    skill_state,
                    registry,
                    business_tool_names,
                    int(state.get("session_turn", 1)),
                )
            debug.emit("TOOL RESULT", name=call["name"], result=value)
            if call["name"] in subagent_names:
                debug.emit("SUBAGENT RESULT", name=call["name"], result=value)
            results.append(self._result(call, value, status="success" if succeeded else "error"))
        return {
            "messages": results,
            "loaded_skills": loaded,
            "active_skill": active,
            "skill_state": skill_state,
        }

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
    ) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        loaded = list(state.get("loaded_skills", []))
        skill_state = dict(state.get("skill_state", {}))
        active = state.get("active_skill")
        results = []
        for call in calls:
            if call["name"] not in tools:
                raise ValueError(f"Model requested unknown tool: {call['name']}")
            if call["name"] in subagent_names:
                debug.emit("SUBAGENT CALL", name=call["name"], task=call["args"].get("task"))
            if (tools[call["name"]].metadata or {}).get("harness_mcp"):
                debug.emit("MCP TOOL", name=call["name"])
            debug.emit("TOOL CALL", name=call["name"])
            succeeded = False
            try:
                self._require_loaded(call, loaded, registry)
                args = self._approved_arguments(tools[call["name"]], call, debug)
                request = ToolRequest(
                    middleware.current_execution(),
                    tools[call["name"]],
                    args,
                    config,
                    call["id"],
                )
                value = await middleware.atool(
                    request, lambda req: req.tool.ainvoke(req.arguments, req.config)
                )
                succeeded = True
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - tool failures are observations
                debug.emit("ERROR", error=str(exc))
                value = f"Error: {exc}"
            if succeeded:
                loaded, skill_state, active = self._skill_state_after(
                    call,
                    loaded,
                    skill_state,
                    registry,
                    business_tool_names,
                    int(state.get("session_turn", 1)),
                )
            debug.emit("TOOL RESULT", name=call["name"], result=value)
            if call["name"] in subagent_names:
                debug.emit("SUBAGENT RESULT", name=call["name"], result=value)
            results.append(self._result(call, value, status="success" if succeeded else "error"))
        return {
            "messages": results,
            "loaded_skills": loaded,
            "active_skill": active,
            "skill_state": skill_state,
        }

    @staticmethod
    def _approved_arguments(tool: BaseTool, call: Mapping[str, Any], debug: DebugHandler) -> dict[str, Any]:
        metadata = tool.metadata or {}
        if not metadata.get("harness_approval"):
            return dict(call["args"])
        payload = {
            "tool": tool.name,
            "args": dict(call["args"]),
            "message": metadata.get("harness_approval_message"),
        }
        debug.emit("HITL INTERRUPT", **payload)
        decision = interrupt(payload)
        if isinstance(decision, str):
            decision = {"decision": decision}
        action = decision.get("decision", decision.get("action"))
        if action == "approve":
            return dict(call["args"])
        if action == "edit" and isinstance(decision.get("args"), Mapping):
            return dict(decision["args"])
        if action == "reject":
            raise ValueError(f"Tool '{tool.name}' was rejected by the user")
        raise ValueError("Invalid HITL resume decision")


class _PlanOutput(TypedDict):
    steps: list[dict[str, str]]


class ReActStrategy(_GraphStrategySupport, AgentStrategy):
    """Standard model -> tools -> model loop with centralized context."""


class PlanExecuteStrategy(_GraphStrategySupport, AgentStrategy):
    """A bounded planner followed by the existing ReAct tool/runtime loop."""

    def __init__(self, *, max_steps: int = 8, max_replans: int = 2) -> None:
        if max_steps < 1 or max_replans < 0:
            raise ValueError("max_steps must be positive and max_replans non-negative")
        self.max_steps, self.max_replans = max_steps, max_replans

    planning = True
