# Agent Harness

## Merge protection

The CI workflow runs on every pull request and on pushes to the default `master`
branch. Repository administrators must configure a GitHub ruleset (or branch
protection rule) for `master`, require pull requests, and require these stable
status-check names before merging:

- `Ruff`
- `pytest (Python 3.11)`
- `pytest (Python 3.12)`

Branch protection is repository-side configuration and cannot be enforced by a
workflow committed to this repository. The rule should also prevent bypassing
these requirements for the roles that merge normal changes.

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
from agent_harness import configure_default_persistence, create_agent

# 由应用启动层创建并管理官方 LangGraph 持久化实现。
configure_default_persistence(
    checkpointer=persistent_checkpointer,
    store=persistent_store,
)

agent = create_agent(
    name="assistant",
    instructions="你是一名业务助手。",
    model=model,  # LangChain BaseChatModel
    tools=[tool_a, tool_b],
    skills=["skills/analysis"],
)

result = agent.invoke("处理这个任务", session_id="session-001")
print(result.output)
```

`invoke`、`ainvoke`、`resume`、`aresume` 返回稳定的 `AgentResult`。普通应用读取
`output`、`status`、`session_id`、`structured_output` 和 `interrupts`；需要诊断或
高级编排时才读取完整的 `result.state`。状态为 `completed`、`paused` 或 `error`。
启用 Structured Output 时，格式化结果同时作为 `output` 和 `structured_output` 返回。

所有执行都使用公开 `session_id`。调用时未传 ID，Harness 会生成 `session-...` 并通过
`AgentResult.session_id` 返回；HITL 暂停后可以直接用该 ID 恢复，不存在应用层无法取得的
隐藏 ephemeral ID。`stream` / `astream` 返回稳定的 `StreamEvent`，事件类型包括
`text_delta`、`tool_start/end`、`subagent_start/end`、`approval_required`、
`plan_update`、`final` 和 `error`，无需理解 LangGraph chunk 或节点名称。两者必须显式传入
非空 `session_id`；该 ID 可直接用于 HITL 恢复及 Session 清理。需要原始 LangGraph
事件的高级集成可使用 `raw_stream` / `araw_stream`。同一 ID 通过
LangGraph Checkpointer 保持上下文，不同 ID 隔离；应用层不接触 `thread_id`。
默认 Session namespace 使用稳定的 Agent name，因此重新创建同名 Agent 后仍可从
持久化 Checkpointer 恢复。高级用户可用 `session_namespace="service-a"` 区分同名
Agent，并可向 `create_agent(checkpointer=...)` 显式传入持久化 Checkpointer。

模型也可通过 `configure_default_model(model)` 配置一次后省略。若安装了可选的
`langchain` 及相应 Provider 集成，也可传模型字符串或设置
`AGENT_HARNESS_MODEL`。

## 持久化与 Session 生命周期

Harness 不提供 RAM Checkpointer/Store fallback。每个 Agent 必须显式传入持久化
`checkpointer=`，或先调用 `configure_default_persistence()` 配置全局默认值；两者都没有
时，`create_agent()` 立即抛出 `PersistenceError`。已知内存 Checkpointer 和内存 Store
会被拒绝。

生产部署推荐使用 LangGraph 官方 PostgreSQL Checkpointer/Store，由应用启动层从外部配置
读取连接信息、初始化 schema，并注入 Harness；Harness 不保存或硬编码数据库连接地址。
本地开发可安装 `agent-harness[persistence-sqlite]` 并注入文件型 SQLite Checkpointer。
同步与异步调用应选择支持相应接口的官方实现。

```python
configure_default_persistence(
    checkpointer=postgres_checkpointer,
    store=postgres_store,
)

agent.clear_session("session-001")
await agent.aclear_session("session-002")
```

`clear_session` 内部将公开 Session ID 转换为稳定 thread ID，并根据 Main checkpoint 中
持久化的 SubAgent tool call 关系先清理派生 child checkpoint，再删除 Main checkpoint；
应用层无需接触 child session。后端失败统一包装为 `PersistenceError`。Checkpointer/Store
的连接生命周期仍由创建它们的应用负责。该入口可供后续 retention policy 调用，本期不包含
定时清理系统。

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

未配置时默认启用 `RetryMiddleware`、`CallLimitMiddleware` 和
`TimeoutMiddleware`。Retry 位于 CallLimit 外层，因此每一次实际 Model/Tool retry
attempt 都会单独计入调用上限。生命周期为：

```text
before_agent -> before_model -> wrap_model_call -> after_model
             -> wrap_tool_call -> after_agent
```

Before 按注册顺序执行，Wrapper 的第一个注册项位于最外层，After 逆序执行。
传 `middleware=[...]` 默认按“内置 Middleware 在前、自定义 Middleware 在后”的
稳定顺序追加；只有显式传 `middleware_mode="replace"` 才完全替换默认项。
自定义中间件继承 `AgentMiddleware`，异步特殊逻辑可覆盖对应的 `a...` 方法。

`timeout_seconds` 只对异步 Runtime 提供 Harness 级强制超时，底层使用可取消的
`asyncio.wait_for`。同步 Model/Tool 调用无法可靠取消，因此同步路径始终在线程内直接
执行；它不会启动后台工作线程，也不会在超时后重试仍在运行的副作用。同步调用需要
由 Provider/Tool 客户端配置自身 timeout；需要 Harness 强制超时时应使用异步接口。

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
        summary_token_threshold=6000,
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
result = agent.invoke(
    {"messages": [message], "project_id": "P-001"},
    context={"tenant": "acme"},
)
```

业务字段与 Harness State 在建图时合并。`messages`、`iteration`、
`runtime_metadata`、`available_skills`、`loaded_skills`、`active_skill`、
`skill_state`、`session_turn`、`summary`、`summarized_messages`、`context`、
`structured_response`、`plan`、`step_failed`、`pending_tool_calls`、
`tool_call_index` 和 `long_term_memories` 是完整的 Harness 保留字段，业务 Schema 不能
覆盖。调用参数 `context=` 是单次执行的业务 Runtime Context，不写入会话 State。

长会话达到消息或 token 阈值后，由 LangMem `summarize_messages` 压缩旧消息并保留
近期上下文；Harness 只提供默认阈值和接线。模型上下文始终由“压缩快照 + 快照后新增
消息”组成，因此摘要后的 Tool Call/Tool Result 不会丢失。旧 Tool Result 在模型
上下文中会先被省略或截断，不会无限堆积。PlanExecute 的入口顺序为
`prepare -> summarize（按需）-> planner`，Planner 会看到本轮最新输入和最新摘要。

## Plan & Execute、长期记忆与 HITL

默认仍为 `ReActStrategy`；复杂任务可传
`strategy=PlanExecuteStrategy(max_steps=8, max_replans=2)`。Planner 使用模型原生
`with_structured_output()` 生成有界计划，执行阶段继续复用相同 Tool、Skill、
SubAgent、Middleware、Session 与 Checkpoint runtime，结果 state 的 `plan` 包含
`steps`、`current_step`、`status` 和 `replan_count`。步骤失败或结果明显不足时只重排
未完成步骤；已完成步骤及结果保持不变，并受 `max_replans` / `max_steps` 限制。Step 状态
为 `pending/completed/failed/skipped`；Plan 状态为
`executing/synthesizing/completed/partial/failed`。重规划耗尽后，失败步骤不会被误标为
completed：无成功步骤时 Plan 为 failed，已有部分成果时为 partial。Planner/Replanner
控制指令使用 SystemMessage，不伪造 Assistant 响应。

通过 `memory=True` 启用基于持久化 LangGraph Store 和 LangMem manager 的跨会话记忆。
启用时必须显式传入 `store=persistent_store`，或配置全局 persistent Store；缺失时在创建
Agent 阶段抛出 `PersistenceError`。应用调用时提供稳定的 `memory_id`：

```python
agent = create_agent(..., memory=True, store=persistent_store)
agent.invoke("我喜欢简洁报告", session_id="A", memory_id="user-1")
agent.invoke("按我的偏好写", session_id="B", memory_id="user-1")
```

需要语义相似度检索时，应由调用方传入已配置 vector index/embedding 的持久化 LangGraph
`BaseStore`；Harness
不会选择或硬编码 embedding provider。Main 委派时会把稳定的 `memory_id` 传给已启用
Memory 的 SubAgent；长期记忆仍按 `(namespace, agent_name, memory_id)` 隔离，因此
SubAgent A/B 和 Main 不会读取彼此的记忆。

用 `require_approval(tool)` 标记敏感工具。图会通过 LangGraph `interrupt()` 暂停，
随后调用 `agent.resume(session_id="...", decision="approve")`；也支持 `reject`，或
`{"decision": "edit", "args": {...}}` 修改参数。每个 Tool Call 单独形成可检查点的
图步骤，因此一条 AIMessage 中较早完成的副作用工具不会在后续审批恢复时重放。
SubAgent 内的审批会通过稳定的 parent-session/subagent/tool-call 映射传播到 Main，
应用仍只需恢复 Main session。计划和步骤结果由 Checkpointer 原样保留。
`GuardrailMiddleware` 支持 input/tool/output 的 pass、reject、modify；
`fallback=[model_b, model_c]` 在主模型重试耗尽后依次降级，Fallback 调用继续经过其
下游的 Retry、CallLimit、Timeout 和自定义 model wrapper。

MCP 不使用专属 runtime。安装 `agent-harness[mcp]` 后，调用异步
`load_mcp_tools(server_config)`，并把得到的标准 `BaseTool` 列表传入 `tools=`，即可
自动获得现有 Middleware、HITL、Debug 和调用上限能力。

## 错误边界

Harness 在应用边界提供稳定的 `AgentError` 子类。Model、Session、Persistence、Memory、HITL、MCP
以及 Strategy 错误会保留原始异常为 `cause`；Middleware 的 before/after hook 错误包装为
`MiddlewareError`，未知 Tool 包装为 `ToolError`。普通 Tool 的可恢复执行失败仍作为
observation 写回模型，使 Agent 可以自行修正；SubAgent 为保持隔离不会向 Main Agent 抛出
内部异常，而是在结果中返回 `status="error"`、`error` 及
`metadata.error_type="SubAgentError"`。

HITL resume 会重新进入同一 Middleware agent 生命周期；无论普通调用还是 resume，
`after_agent` 每次执行最多调用一次，避免 hook 自身失败时重复产生副作用。

## Advanced Skills

Skill 目录示例：

```text
analysis/
├── SKILL.md
├── references/
│   └── standard.md
├── assets/
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

按规范分析；需要时读取 reference/asset 或运行声明脚本。
```

支持同名不同版本（如 `analysis@1.0`、`analysis@2.0`）。不带版本引用在存在多个
版本时解析到最高版本。创建 Agent 时检查目录、Metadata、依赖缺失和循环依赖；
加载前检查 `required_tools` 并按拓扑顺序加载依赖。

模型起初只看到按当前任务动态筛选的 `name` 和 `description`。默认
`LexicalSkillSelector` 保持原有字面/CJK 匹配行为；可通过
`create_agent(skill_selector=custom_selector)` 注入自定义 `SkillSelector`，Harness 不绑定
embedding provider。加载后只加入 `SKILL.md` 指令以及可用资产名称；`references/` 和
`assets/` 由内部工具按需读取。旧 `resources/` 目录继续兼容，但新 Skill 推荐使用
`references/assets/scripts`。Skill 在若干未使用会话轮次后自动卸载，也可调用
`unload_skill`。

声明脚本只能从该 Skill 的 `scripts/` 中执行，支持 `.py`、`.ps1`、`.sh`。统一
协议是从 stdin 读取 JSON 参数，并向 stdout 输出 JSON（普通文本也可作为结果）：

```python
import json
import sys

arguments = json.load(sys.stdin)
print(json.dumps({"result": arguments["value"] * 2}))
```

脚本默认只获得运行所需的少量平台环境变量，不继承完整 `os.environ`。需要的变量必须通过
`RuntimeConfig(script_env_allowlist=("APP_REGION",))` 显式允许；API Key、数据库密码和
Token 不会因父进程环境而自动暴露。该机制继续保留受控路径、timeout、stdout 大小限制及
JSON stdin/stdout 合约，但不是完整安全沙箱；Skill 来源仍应可信。

## Observability 与 Debug

`create_agent(event_sink=sink)` 或 `event_sinks=[sink_a, sink_b]` 可接入实现
`emit(RuntimeEvent)` 的日志、审计或遥测平台。每个事件都有稳定的 `event_type`、时间、
Agent/Session、trace/span/parent span、状态、耗时、元数据和错误字段；一次请求及其
SubAgent 共用 `trace_id`，HITL resume 继续原 trace。`agent.metrics` 提供 Agent、Model、
Tool、SubAgent、retry/fallback、HITL、Skill、summary、Memory 和 token usage 基础指标。

默认事件会脱敏 API Key、Token、密码等字段，限制集合和字符串大小，并省略完整 Model
messages。可通过 `RuntimeConfig(observability=ObservabilityConfig(...))` 调整 payload
详细度与上限。任何 EventSink 异常都会与主流程隔离。

`RuntimeConfig(debug=True, debug_format="print")` 只是上述 EventSink 的轻量控制台实现；
`debug_format` 支持 `print`、`json`、`md`，Runtime 本身不再散落直接 `print`。

## 开发检查

```powershell
python -m compileall -q src
ruff check src tests
pytest -q
```

CI 在 Pull Request 上运行 Ruff，并在 Python 3.11、3.12 执行核心测试。测试默认
使用临时 SQLite Checkpointer/Store，不把内存存储当作运行时持久化实现。
