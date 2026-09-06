# Agent Harness

这是一个基于 LangGraph 的轻量级 Agent Harness。它对外提供应用层的
`Agent`

## 安装依赖

项目使用 `requirements.txt` 管理依赖，不使用 uv。建议在虚拟环境中执行：

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需以可编辑方式安装当前项目，可在安装 requirements 后执行：

```bash
python -m pip install -e . --no-deps
```

## 快速开始

```python
from agent_harness import create_agent

agent = create_agent(
    name="demo_agent",
    description="示例 Agent",
    model=model,  # 任意 LangChain BaseChatModel
    instructions="你是一名业务助手。",
    tools=[tool_a, tool_b],
    skills=["skills/road_noise_analysis"],
)

state = agent.invoke("请帮助我处理这个任务")
print(state["messages"][-1].content)
```

返回的对象还提供 `ainvoke`、`stream` 和 `astream` 方法。

## Skill

Skill 使用带 YAML frontmatter 的 `SKILL.md` 文件定义。模型首次只会收到每个
Skill 的名称和描述；模型调用内部的 `load_skill` 工具后，完整指令以及
`references/` 目录中的 UTF-8 文件才会加入后续模型上下文。

示例：

```markdown
---
name: road_noise_analysis
description: 分析并报告道路噪声测量结果
version: 1.0
tags: [声学]
required_tools: [measurement_lookup]
dependencies: []
---

请遵循测量和报告标准操作流程……
```

## 开发检查

```bash
python -m compileall -q src
python -m pytest
ruff check .
```
