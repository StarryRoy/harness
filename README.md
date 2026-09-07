# Agent Harness

基于 LangGraph 的轻量 Agent Runtime。Harness 管理 ReAct、会话、Checkpoint、
Context、Middleware、SubAgent 隔离及渐进式 Skill；应用层只需定义 Agent。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## 快速开始

```python
from agent_harness import create_agent

agent = create_agent(
    name="assistant",
    instructions="你是一名业务助手。",
    model=model,  # LangChain BaseChatModel
    tools=[tool_a, tool_b],
    skills=["skills/analysis"],
)

state = agent.invoke("处理这个任务", session_id="session-001")
print(state["messages"][-1].content)
```

`invoke` / `ainvoke` / `stream` / `astream` 均支持 `session_id`。同一 ID 通过
LangGraph Checkpointer 保持上下文，不同 ID 隔离；应用层不接触 `thread_id`。
默认 Session namespace 使用稳定的 Agent name，因此重新创建同名 Agent 后仍可从
持久化 Checkpointer 恢复。高级用户可用 `session_namespace="service-a"` 区分同名
Agent，并可向 `create_agent(checkpointer=...)` 传入其他 LangGraph Checkpointer。

模型也可通过 `configure_default_model(model)` 配置一次后省略。若安装了可选的
`langchain` 及相应 Provider 集成，也可传模型字符串或设置
`AGENT_HARNESS_MODEL`。

## SubAgent

```python
researcher = create_agent(
    name="researcher",
    description="检索并归纳事实",
    instructions="只返回任务结论和必要依据。",
    model=model,
    tools=[search],
)

main = create_agent(
    name="main",
    instructions="按需委派并汇总结果。",
    model=model,
    subagents=[researcher],
)
```

Harness 自动调用 `researcher.as_tool()`。工具输入只有 `task`，SubAgent 使用自己
独立的 Runtime、State、Session scope、Middleware、Checkpointer 和 SkillRegistry。
返回结构固定为 `content`、`status`、`metadata`、`error`，不会把完整执行轨迹交给
Main Agent。

## Middleware

未配置时默认启用 `CallLimitMiddleware`、`RetryMiddleware` 和
`TimeoutMiddleware`。生命周期为：

```text
before_agent -> before_model -> wrap_model_call -> after_model
             -> wrap_tool_call -> after_agent
```

Before 按注册顺序执行，Wrapper 的第一个注册项位于最外层，After 逆序执行。
传 `middleware=[...]` 默认按“内置 Middleware 在前、自定义 Middleware 在后”的
稳定顺序追加；只有显式传 `middleware_mode="replace"` 才完全替换默认项。
自定义中间件继承 `AgentMiddleware`，异步特殊逻辑可覆盖对应的 `a...` 方法。

`TimeoutMiddleware` 的异步路径使用可取消的 `asyncio.wait_for`。同步 Model/Tool
调用无法可靠取消，因此同步路径直接执行、不制造后台残留任务或由 timeout 引发的
重复调用；需要强制超时时应使用异步接口及底层客户端自身的 timeout。

默认超时、重试、调用上限和 Context 策略由 `RuntimeConfig` 配置：

```python
from agent_harness import ContextPolicy, RuntimeConfig

config = RuntimeConfig(
    max_iterations=12,
    retry_attempts=2,
    timeout_seconds=60,
    call_limit=48,
    context_policy=ContextPolicy(
        summary_threshold=40,
        summary_keep_recent=12,
        skill_retention_turns=2,
    ),
)
```

## 业务 State 与 Runtime Context

```python
from typing import TypedDict

class BusinessState(TypedDict, total=False):
    project_id: str
    approval_status: str

agent = create_agent(..., state_schema=BusinessState)
state = agent.invoke(
    {"messages": [message], "project_id": "P-001"},
    context={"tenant": "acme"},
)
```

业务字段与 Harness State 在建图时合并。`messages`、`iteration`、
`runtime_metadata`、`available_skills`、`loaded_skills`、`active_skill`、
`skill_state`、`session_turn`、`summary` 和 `structured_response` 是保留字段，业务
Schema 不能覆盖。`context` 是单次执行的业务 Runtime Context，不写入会话 State。

长会话达到阈值后由当前模型摘要旧消息，保存 `summary` 并保留最近消息。旧 Tool
Result 在模型上下文中会先被省略或截断，不会无限堆积。

## Advanced Skills

Skill 目录示例：

```text
analysis/
├── SKILL.md
├── references/
│   └── standard.md
├── resources/
│   └── report.json
└── scripts/
    └── calculate.py
```

```markdown
---
name: analysis
description: 分析测量结果
version: 2.0
tags: [analysis]
required_tools: [measurement_lookup]
dependencies: [common_rules@1.0]
scripts: [calculate.py]
---

按规范分析；需要时读取 reference/resource 或运行声明脚本。
```

支持同名不同版本（如 `analysis@1.0`、`analysis@2.0`）。不带版本引用在存在多个
版本时解析到最高版本。创建 Agent 时检查目录、Metadata、依赖缺失和循环依赖；
加载前检查 `required_tools` 并按拓扑顺序加载依赖。

模型起初只看到按当前任务动态筛选的 `name` 和 `description`。加载后只加入
`SKILL.md` 指令以及可用资产名称；`references/` 和 `resources/` 由内部工具按需
读取。Skill 在若干未使用会话轮次后自动卸载，也可调用 `unload_skill`。

声明脚本只能从该 Skill 的 `scripts/` 中执行，支持 `.py`、`.ps1`、`.sh`。统一
协议是从 stdin 读取 JSON 参数，并向 stdout 输出 JSON（普通文本也可作为结果）：

```python
import json
import sys

arguments = json.load(sys.stdin)
print(json.dumps({"result": arguments["value"] * 2}))
```

这只是受控路径和超时/输出限制，不是完整安全沙箱；Skill 来源仍应可信。

## Debug

`RuntimeConfig(debug=True, debug_format="print")` 可输出极简事件，包括
`SESSION`、`MIDDLEWARE`、`SUBAGENT CALL/RESULT`、`SKILL SCRIPT`、Model 和 Tool
调用。`debug_format` 支持 `print`、`json`、`md`。

## 开发检查

```powershell
python -m compileall -q src
ruff check src
```
