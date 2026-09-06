"""Replaceable graph-building strategy and the Phase 1 ReAct implementation."""

import json
from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph

from .debug import DebugHandler
from .definition import AgentDefinition
from .skills import SkillError, SkillRegistry
from .state import AgentState


class AgentStrategy(ABC):
    """The narrow extension point between runtime and execution strategy."""

    @abstractmethod
    def build_graph(
        self,
        definition: AgentDefinition,
        skills: SkillRegistry,
        debug: DebugHandler,
    ) -> Any:
        """Return a compiled LangGraph runnable."""


class ReActStrategy(AgentStrategy):
    """Standard model -> tools -> model loop."""

    def build_graph(
        self,
        definition: AgentDefinition,
        skills: SkillRegistry,
        debug: DebugHandler,
    ) -> Any:
        tool_map = {tool.name: tool for tool in definition.tools}
        load_skill = self._skill_tool(skills, set(tool_map), debug)
        all_tools: list[BaseTool] = list(definition.tools)
        if skills.list():
            if load_skill.name in tool_map:
                raise ValueError(f"Tool name is reserved by the harness: {load_skill.name}")
            all_tools.append(load_skill)
        executable = {tool.name: tool for tool in all_tools}
        model = definition.model.bind_tools(all_tools) if all_tools else definition.model

        def prompt(state: AgentState) -> list[Any]:
            sections = [definition.instructions.strip()]
            available = state.get("available_skills", [])
            if available:
                catalog = "\n".join(f"- {item['name']}: {item['description']}" for item in available)
                sections.append(
                    "Available skills (only summaries are shown). Call load_skill before using one:\n"
                    + catalog
                )
            for name in state.get("loaded_skills", []):
                sections.append(skills.get(name).context())
            return [SystemMessage(content="\n\n".join(filter(None, sections))), *state["messages"]]

        def model_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            debug.emit("MODEL CALL", iteration=state.get("iteration", 0) + 1)
            response = model.invoke(prompt(state), config)
            return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

        async def async_model_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            debug.emit("MODEL CALL", iteration=state.get("iteration", 0) + 1)
            response = await model.ainvoke(prompt(state), config)
            return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

        def route(state: AgentState) -> Literal["tools", "format", "done"]:
            message = state["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                if state.get("iteration", 0) >= definition.runtime_config.max_iterations:
                    raise RuntimeError(
                        f"Agent exceeded max_iterations={definition.runtime_config.max_iterations}"
                    )
                return "tools"
            return "format" if definition.response_format is not None else "done"

        def tool_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            return self._execute_tools(state, executable, config, debug)

        async def async_tool_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            return await self._aexecute_tools(state, executable, config, debug)

        graph = StateGraph(AgentState)
        graph.add_node("model", RunnableLambda(model_node, async_model_node))
        graph.add_node("tools", RunnableLambda(tool_node, async_tool_node))
        graph.add_edge(START, "model")
        destinations: dict[str, Any] = {"tools": "tools", "done": END}
        if definition.response_format is not None:
            destinations["format"] = "format"
        graph.add_conditional_edges("model", route, destinations)
        graph.add_edge("tools", "model")

        if definition.response_format is not None:
            formatter = definition.model.with_structured_output(definition.response_format)

            def format_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
                return {"structured_response": formatter.invoke(prompt(state), config)}

            async def async_format_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
                return {"structured_response": await formatter.ainvoke(prompt(state), config)}

            graph.add_node("format", RunnableLambda(format_node, async_format_node))
            graph.add_edge("format", END)
        return graph.compile()

    @staticmethod
    def _skill_tool(
        registry: SkillRegistry, tool_names: set[str], debug: DebugHandler
    ) -> StructuredTool:
        def load_skill(name: str) -> str:
            """Load a named skill's complete instructions and references into context."""
            skill = registry.get(name)
            missing = sorted(set(skill.metadata.required_tools) - tool_names)
            if missing:
                raise SkillError(
                    f"Skill '{name}' requires unavailable tools: {', '.join(missing)}"
                )
            debug.emit("SKILL LOAD", name=name)
            return f"Skill '{name}' loaded. Its instructions are now available."

        return StructuredTool.from_function(load_skill, name="load_skill")

    @staticmethod
    def _result(call: dict[str, Any], value: Any) -> ToolMessage:
        content = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])

    def _execute_tools(
        self,
        state: AgentState,
        tools: dict[str, BaseTool],
        config: RunnableConfig,
        debug: DebugHandler,
    ) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        loaded = list(state.get("loaded_skills", []))
        results = []
        for call in calls:
            if call["name"] not in tools:
                raise ValueError(f"Model requested unknown tool: {call['name']}")
            debug.emit("TOOL CALL", name=call["name"])
            try:
                value = tools[call["name"]].invoke(call["args"], config)
                if call["name"] == "load_skill":
                    name = call["args"]["name"]
                    if name not in loaded:
                        loaded.append(name)
            except Exception as exc:
                debug.emit("ERROR", error=str(exc))
                value = f"Error: {exc}"
            debug.emit("TOOL RESULT", name=call["name"], result=value)
            results.append(self._result(call, value))
        return {"messages": results, "loaded_skills": loaded, "active_skill": loaded[-1] if loaded else None}

    async def _aexecute_tools(
        self,
        state: AgentState,
        tools: dict[str, BaseTool],
        config: RunnableConfig,
        debug: DebugHandler,
    ) -> dict[str, Any]:
        calls = state["messages"][-1].tool_calls
        loaded = list(state.get("loaded_skills", []))
        results = []
        for call in calls:
            if call["name"] not in tools:
                raise ValueError(f"Model requested unknown tool: {call['name']}")
            debug.emit("TOOL CALL", name=call["name"])
            try:
                value = await tools[call["name"]].ainvoke(call["args"], config)
                if call["name"] == "load_skill":
                    name = call["args"]["name"]
                    if name not in loaded:
                        loaded.append(name)
            except Exception as exc:
                debug.emit("ERROR", error=str(exc))
                value = f"Error: {exc}"
            debug.emit("TOOL RESULT", name=call["name"], result=value)
            results.append(self._result(call, value))
        return {"messages": results, "loaded_skills": loaded, "active_skill": loaded[-1] if loaded else None}
