# Agent Harness

A deliberately small Agent Harness built on LangGraph. It exposes an application-level
`Agent` instead of requiring callers to assemble graph nodes and edges.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from agent_harness import create_agent

agent = create_agent(
    name="demo_agent",
    description="Example agent",
    model=model,  # Any LangChain BaseChatModel
    instructions="You are a business assistant.",
    tools=[tool_a, tool_b],
    skills=["skills/road_noise_analysis"],
)

state = agent.invoke("Help me process this task")
print(state["messages"][-1].content)
```

The returned object also provides `ainvoke`, `stream`, and `astream`. Skills use a
`SKILL.md` with YAML frontmatter. Initially the model receives only each skill's name
and description; calling the internal `load_skill` tool adds its full instructions and
UTF-8 files from `references/` to subsequent model context.

```markdown
---
name: road_noise_analysis
description: Analyze and report road noise measurements
version: 1.0
tags: [acoustics]
required_tools: [measurement_lookup]
dependencies: []
---

Follow the measurement and reporting SOP...
```
