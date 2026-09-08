# Phase 3 Prompt：Plan & Execute + HITL / Guardrail / MCP / Structured Output

你现在要在 **已经完成 Phase 1 和 Phase 2 的 Agent Harness 项目** 上继续开发最终 Phase 3。

请先阅读现有代码并复用已经完成的：

```text
Agent
create_agent()
AgentDefinition
AgentRuntime
AgentStrategy
ReActStrategy
Tool Integration
Skills Core + Skills Advanced
SubAgent / Agent.as_tool()
Middleware
Session / Checkpoint
Context Manager
Basic Summarization
```

**不要推翻已有架构**

本期目标：

1. 增加第二种 Agent 运行模式：**Plan & Execute**。
2. 在现有 Harness 上补齐公司内部实际常用的高级能力：Structured Output、HITL、Guardrail、Model Fallback、MCP、统一 Error Model。
3. 仍然保持“够用、专业、不过度设计”。

---

# 一、总体原则

1. LangGraph 继续作为底层 Runtime。
2. ReAct 继续保留并作为默认 Strategy。
3. Plan & Execute 必须通过 Phase 1 已定义的可替换 `AgentStrategy` 接口接入。
4. Plan & Execute 必须复用现有 Tool、Skill、SubAgent、Middleware、Session、Context，不得重新创建第二套生态。
7. Testing / Eval 平台仍然不开发。
8. Observability 仍然只需要极简 Debug，不建设正式平台。
9. 不追求 C 端大规模 SaaS 能力。

---

# 二、Agent Strategy API 正式开放

Phase 1 已存在：

```text
AgentStrategy
└── ReActStrategy
```

本期新增：

```text
PlanExecuteStrategy
```

应用层应该能够选择 Strategy，例如：

```python
agent = create_agent(
    name="complex_agent",
    model=model,
    instructions="...",
    tools=[...],
    skills=[...],
    subagents=[...],
    strategy=PlanExecuteStrategy(...),
)
```

默认不传时：

```text
ReActStrategy
```

如果现有 API 更适合支持字符串：

```python
strategy="plan_execute"
```

也可以提供快捷方式，但内部必须对应真正的 Strategy 对象，不要大量 if/else 写死在 Runtime。

---

# 三、Plan & Execute 核心

Plan & Execute 用于复杂、多步骤、跨 Tool / Skill / SubAgent 的任务。

核心流程：

```text
User Goal
↓
Planner
↓
Plan
↓
Executor
↓
Step Result
↓
继续下一 Step / Re-plan
↓
Final Synthesis
```

---

## 1. Planner

负责将用户目标拆为有限步骤。

Plan 数据结构至少包含：

```text
plan_id（可选）
steps
current_step
status
```

每个 Step 至少包含：

```text
step_id
description
status
result
```

可选加入：

```text
assigned_capability
metadata
```

但不要把 Plan 设计成复杂 DSL。

---

## 2. Plan State

PlanExecuteStrategy 维护自己的 Strategy-specific State。

至少包括：

```text
plan
current_step
completed_steps
step_results
replan_count
```

不要污染所有 Agent 必需的基础 State；Strategy 特有字段应该保持边界清晰。

---

## 3. Executor

Executor 按 Step 执行。

每个 Step 可以使用现有：

```text
Tool
Skill
SubAgent
```

重要：

```text
Plan & Execute 不允许重新开发 Tool Runtime
Plan & Execute 不允许重新开发 Skill Runtime
Plan & Execute 不允许重新开发 SubAgent Runtime
```

应通过已经存在的 Harness 能力完成执行。

---

## 4. Re-plan

支持简单 Re-plan：

```text
Step 执行结果不足 / 失败
↓
根据当前结果调整剩余计划
```

必须提供限制：

```text
max_steps
max_replans
```

避免无限规划。

本期不需要复杂 Reflection / Self-Critic。

---

## 5. Final Synthesis

全部必要 Step 完成后：

```text
Step Results
↓
Final Synthesis
↓
Final Answer
```

最终输出仍走 Agent 统一输出接口。

---

## 6. Plan & Execute 与 Session / Context

必须继续兼容：

```text
Session
Checkpoint
Context Manager
Loaded Skills
SubAgent Context Isolation
Middleware
```

不要让 Plan & Execute 成为一个旁路系统。

---

## 7. Plan & Execute 与 HITL

本期 HITL 完成后，Plan Step 中如果触发需要审批的 Tool，应正常暂停并 Resume，而不是破坏 Plan State。




# 五、HITL

本期实现实用版本 Human-in-the-Loop。

主要场景：

```text
高风险 Tool
↓
暂停
↓
用户 Approve / Reject / Edit
↓
Resume
```

底层直接使用 LangGraph：

```text
interrupt()
Checkpointer
Command(resume=...)
```

不要重新实现 Pause/Resume/Persistence。

---

## 1. Tool Approval

需要提供清晰配置方式。

可以采用 Tool Metadata / Harness 配置，例如概念上：

```python
@tool
# + approval metadata
```

或者：

```python
create_agent(
    tool_approval={...}
)
```

请结合现有 Tool 结构选择最简单、清晰的方式。

不要为了语法漂亮重写整个 Tool 系统。

---

## 2. Approval Result

至少支持：

```text
approve
reject
edit
```

Edit 允许修改 Tool 参数后继续。

---

## 3. Resume

Session / Checkpoint 必须保证暂停后可以恢复。

ReAct 与 Plan & Execute 都必须可用。

---

# 六、Guardrail

本期只做轻量、实用的 Guardrail。

支持三个位置：

```text
Input Guardrail
Tool Guardrail
Output Guardrail
```

建议使用 Middleware / Hook 实现，避免污染 Agent Runtime 核心。

至少允许：

```text
pass
reject
modify（如实现简单）
```

不建设复杂策略配置平台。

---

# 七、Model Fallback

通过 Middleware 实现：

```text
Primary Model
↓ error / retry exhausted
Fallback Model
```

配置应简单，例如：

```python
ModelFallbackMiddleware([...])
```

或者结合现有 Middleware 风格。

不要构建复杂 Model Router 平台。

---

# 八、MCP

本期支持 MCP Tool 接入。

目标：

```text
MCP Server
↓
发现/获取 Tools
↓
转换为 Harness 兼容 Tool
↓
Agent.tools
```

关键原则：

1. MCP Tool 最终进入现有 Tool 调用链。
2. 不创建 MCP 专属第二套 Agent Runtime。
3. Tool Call、Middleware、HITL、Debug 应尽可能继续生效。
4. 连接方式优先复用成熟 MCP / LangChain 能力，不自研 MCP 协议栈。
5. 只做够业务使用的基础 MCP 接入，不开发 MCP 管理平台。

---

# 九、统一 Error Model

当前项目经过三期后模块较多，需要统一对外异常。

至少定义：

```text
AgentError
ModelError
ToolError
SkillError
SubAgentError
SessionError
StrategyError
MiddlewareError
```

如需增加：

```text
HITLError
MCPError
```

可以加入。

原则：

1. 保留原始 exception 作为 cause。
2. 给应用层清楚的错误类型和 message。
3. 不要把大量底层 LangGraph / LangChain exception 原样泄漏到业务 API。
4. 不要过度定义几十种细粒度异常。

---

# 十、Context / Memory 的最终范围

本项目当前只需要：

```text
Session short-term memory
Checkpoint
Basic Summarization
Context Manager
Skill Context
SubAgent Result Context
Plan State
```

本期**不开发 Long-Term Memory**。

不开发：

```text
User Profile Memory
Cross-session semantic memory
复杂 Memory Store
自动 Memory Extraction
```

以后有明确业务需求再做。

---

# 十一、Debug / Observability

继续保持极低开发量。

只需让 Debug 能看到新增关键节点：

```text
STRATEGY
PLAN CREATED
PLAN STEP
REPLAN
HITL INTERRUPT
HITL RESUME
GUARDRAIL
MCP TOOL
ERROR
```

仍然使用：

```text
print
JSON
MD
```

即可。

不要开发：

```text
OpenTelemetry
LangSmith Adapter
Trace UI
Dashboard
Metrics Platform
Token/Cost Platform
```

代码只需保留以后替换 DebugHandler 的空间。

---

# 十二、Testing / Evaluation

**不开发。**

不要创建：

```text
Agent Eval Framework
Dataset Manager
LLM Judge
Trajectory Evaluation
Skill Benchmark
SubAgent Routing Benchmark
Regression Platform
```

如果为了验证代码需要临时 demo / example，可以创建最少量可运行示例，但不要把它扩展成测试工程。

---

# 十三、Examples / Demo

本期结束时建议保留极少量运行示例用于人工验证，例如：

```text
examples/
├── react_agent.py
├── plan_execute_agent.py
├── multi_agent.py
└── hitl_agent.py
```

示例只用于人工运行验证，不建设测试框架。

---

# 十四、本期明确不做

```text
Reflection Strategy
Tree Search
复杂 Planning DSL
复杂并行 Planner
Long-Term Memory
Skill Marketplace
Skill Governance Platform
正式 Observability
Testing / Eval Framework
复杂权限中心
复杂 Audit Platform
复杂 Rate Limit Platform
大型 MCP 管理平台
C 端千万级高并发设计
```

---

# 十五、Phase 3 验收标准

必须人工真实跑通以下能力。

## 场景 A：ReAct 未被破坏

```text
现有 ReAct Agent 继续正常工作
```

---

## 场景 B：Plan & Execute

```text
User 复杂任务
↓
Planner
↓
生成多个 Steps
↓
Tool / Skill / SubAgent 执行
↓
必要时 Re-plan
↓
Final Synthesis
```

---

## 场景 C：两种 Strategy 统一 API

```python
react_agent = create_agent(...)

plan_agent = create_agent(
    ...,
    strategy=PlanExecuteStrategy(...),
)
```

两者继续共用：

```text
Tools
Skills
SubAgents
Middleware
Session
Context
Structured Output
```

---

## 场景 D：HITL

```text
Agent
→ 高风险 Tool
→ interrupt
→ approve/reject/edit
→ resume
→ Final
```

且 Plan & Execute 中的 Tool 也可以暂停恢复。

---

## 场景 E：MCP

```text
MCP Server
→ MCP Tool
→ Agent
→ Tool Call
→ Result
```

MCP Tool 进入统一 Tool Runtime。

---

## 场景 F：Structured Output

ReAct 和 Plan & Execute 都能按照 `response_format` 返回结构化结果。

---

## 场景 G：Guardrail / Fallback

基础 Guardrail 和 Model Fallback 可以通过统一 Middleware/Hook 正常工作。

---

# 十六、最终项目应达到的能力

三期完成后，Harness 应支持：

```text
Agent
create_agent()
AgentDefinition
AgentRuntime
AgentStrategy
ReActStrategy
PlanExecuteStrategy

Tool Integration

Skills Core
Progressive Disclosure
Skill Dependencies
References
Resources
Scripts
Version
Validation
Dynamic Discovery
Skill Context Lifecycle

SubAgent
Agent.as_tool()
Main Agent + SubAgent

Middleware
Retry
Timeout
CallLimit
Model Fallback

Session
Checkpoint
Context Manager
Basic Summarization

Structured Output
HITL
Guardrail
MCP

极简 Debug
统一 Error Model
```
