Phase 3：Plan & Execute + HITL + Memory + Enterprise Features

基于当前 "codex" 分支已完成的 Phase 1、Phase 2 继续开发。

先阅读现有代码，不要推翻现有架构，不要改用 LangChain "create_agent()"。

核心原则：

- LangGraph 继续作为底层 Runtime。
- 默认开箱即用，高级能力允许覆盖。
- 能直接复用 LangChain / LangGraph / LangMem 的能力，不自行重复实现。
- Main Agent 负责全局统筹；SubAgent 只负责自己的领域任务。
- SubAgent 之间默认不直接通信，统一通过 Main Agent 协作。
- 不开发正式测试平台、Eval、Observability 平台。

---

1. Plan & Execute

在现有：

AgentStrategy
├── ReActStrategy

基础上新增：

PlanExecuteStrategy

默认仍使用 ReAct。

高级用户可以：

create_agent(
    ...,
    strategy=PlanExecuteStrategy(...)
)

Plan & Execute 流程：

User Goal
→ Planner
→ Plan
→ Execute Step
→ Tool / Skill / SubAgent
→ Step Result
→ 下一 Step / Re-plan
→ Final Synthesis

Plan 至少包含：

steps
current_step
status
replan_count

Step 至少包含：

step_id
description
status
result

提供：

max_steps
max_replans

不要开发复杂 Planning DSL、Reflection、Tree Search。

PlanExecuteStrategy 必须复用现有：

Tool Runtime
Skill Runtime
SubAgent as Tool
Middleware
Session
Context
Memory
Checkpoint

不得重新实现第二套运行体系。

SubAgent 仍只能作为能力被 Main Agent / 当前 Agent 调用，不增加 Peer-to-Peer、Handoff、Agent 网络。

---

2. Structured Output

不开发自己的 Structured Output Framework。

应用层继续使用：

create_agent(
    ...,
    response_format=MySchema
)

内部直接使用 LangChain：

model.with_structured_output(response_format)

要求：

- 支持 Pydantic / TypedDict / JSON Schema 等 LangChain 原生支持格式。
- ReAct 与 PlanExecute 都使用同一机制。
- Planner 如果需要结构化 Plan，也直接使用 "with_structured_output()"。
- 不自行实现 JSON Parser、Schema Retry Framework、Provider Structured Output Adapter。
- Provider 兼容问题交给 LangChain Model 层处理。

---

3. Short-Term Memory

现有：

session_id
→ thread_id
→ LangGraph Checkpointer

继续作为短期记忆基础。

应用层只使用：

agent.invoke(
    "...",
    session_id="session-001"
)

不要暴露 "thread_id"。

短期记忆压缩

Phase 2 当前已有 Basic Summarization。

本期将压缩实现收敛到成熟组件，不要继续维护自研摘要算法。

优先直接复用：

LangMem short_term
summarize_messages
SummarizationNode

或与当前架构兼容的 LangChain 官方 Summarization 能力。

要求：

达到 token / message 阈值
→ 压缩旧消息
→ 保留摘要
→ 保留最近消息

Context Manager 可以继续负责 Context 组装和触发策略，但不要自己维护：

summary prompt
消息压缩算法
复杂 RemoveMessage 逻辑

高级用户可以调整阈值，默认无需配置。

---

4. Simple Long-Term Memory

本期增加简单长期记忆。

区分：

Short-Term Memory
= Session / Thread / Checkpointer

Long-Term Memory
= Cross-session / Store

长期记忆底层直接使用：

LangGraph BaseStore

开发环境可以使用：

InMemoryStore

生产环境允许应用或平台注入：

PostgresStore
其他 BaseStore

不要自己开发 Memory Database。

---

4.1 长期记忆管理

不要自行开发：

Memory Extraction
Memory Deduplication
Memory Consolidation
Memory Compression
Memory Update

直接使用 LangMem，例如：

create_memory_store_manager()

负责从对话中：

提取重要信息
更新已有记忆
合并重复信息
压缩长期记忆

本期只做简单实用版本，不建设完整 Memory Platform。

---

4.2 长期记忆内容

默认主要保存：

稳定用户信息
用户偏好
重要业务背景
跨 Session 仍然有价值的信息

不要把全部聊天记录直接写入长期记忆。

默认可以使用简单非结构化 Memory。

高级用户允许提供：

memory_schema
memory instructions
memory model
store
namespace

但普通应用层不需要配置。

---

4.3 使用方式

提供简单高层概念，例如：

session_id
= 当前会话

memory_id
= 跨会话记忆主体

例如：

agent.invoke(
    "...",
    session_id="session-A",
    memory_id="user-001",
)

新的：

session-B

只要还是：

memory_id=user-001

即可读取对应长期记忆。

Harness 内部负责：

memory_id
→ Store namespace
→ Memory Search / Load / Update

不要让普通应用层操作 Store key / namespace 细节。

---

4.4 Memory Context

调用模型前：

当前问题
+
相关长期记忆
→ Context Manager
→ Model

执行结束后：

本轮有效信息
→ LangMem Memory Manager
→ Store

只把相关、压缩后的 Memory 放入 Context，不能把全部 Store 内容塞入 Prompt。

---

4.5 Multi-Agent Memory

默认保持隔离：

Main Agent Memory
SubAgent A Memory
SubAgent B Memory

SubAgent 不自动读取其他 Agent 的完整长期记忆。

Main Agent 负责把完成任务真正需要的信息传给 SubAgent。

高级场景允许通过配置共享 Memory namespace，但不是默认行为。

---

5. HITL

实现简单 Human-in-the-Loop。

主要场景：

高风险 Tool
→ interrupt
→ 用户确认
→ approve / reject / edit
→ resume

底层直接使用 LangGraph：

interrupt()
Command(resume=...)
Checkpointer

不要自行实现暂停恢复系统。

应用层提供高层接口，例如：

agent.resume(
    session_id="session-001",
    decision="approve"
)

应用层不需要知道：

thread_id
Command
interrupt internals

至少支持：

approve
reject
edit

Edit 可以修改 Tool 参数。

ReAct 和 PlanExecute 都必须支持 HITL。

PlanExecute 被中断后恢复时必须保留：

Plan
current_step
step_results
replan_count

---

6. Tool Approval

提供简单配置。

优先在现有 Tool 系统上增加轻量 approval metadata / policy。

例如概念上：

safe tool
→ 直接执行

sensitive tool
→ HITL

不要重写 Tool Framework。

默认无需业务配置。

只有需要审批的 Tool 才显式声明。

---

7. Guardrail

提供轻量：

Input Guardrail
Tool Guardrail
Output Guardrail

优先通过 Middleware / Hook 接入。

至少支持：

pass
reject
modify

不要建设复杂规则平台。

---

8. Model Fallback

实现：

Primary Model
→ Retry exhausted / unavailable
→ Fallback Model

通过现有 Middleware 扩展。

提供合理默认行为，高级用户可以配置 fallback models。

不要建设 Model Router 平台。

---

9. MCP

支持 MCP Tool 接入：

MCP Server
→ 获取 Tools
→ 转换为 BaseTool
→ 进入 Agent.tools

要求：

- 优先使用成熟 LangChain MCP Adapter / MCP SDK。
- 不自己实现 MCP Protocol。
- MCP Tool 必须进入现有 Tool Runtime。
- Middleware、HITL、Debug、CallLimit 等继续生效。
- 不创建 MCP 专属 Runtime。
- 不开发 MCP 管理平台。

---

10. Unified Error Model

增加简单统一异常：

AgentError
ModelError
ToolError
SkillError
SubAgentError
SessionError
MemoryError
StrategyError
MiddlewareError
HITLError
MCPError

要求：

- 保留原始异常作为 cause。
- 给应用层稳定、清晰的错误类型。
- 不定义几十种细粒度异常。
- Tool 可恢复错误仍可以作为 Agent observation 处理。

---

11. Default-first API

继续贯彻：

«普通业务开发者少做决定，高级用户按需覆盖。»

普通：

agent = create_agent(
    name="nvh",
    instructions="...",
    tools=[...],
    skills=[...],
    subagents=[...],
)

直接工作。

高级功能可以覆盖：

model
strategy
state_schema
middleware
runtime_config
checkpointer
store
memory
memory_schema
response_format
session_namespace
guardrail
fallback

不要要求普通应用开发者理解：

LangGraph StateGraph
thread_id
Command
Checkpoint internals
Middleware Pipeline internals
Store namespace internals

---

12. Debug

继续保持极简。

补充：

PLAN
PLAN STEP
REPLAN
HITL INTERRUPT
HITL RESUME
MEMORY LOAD
MEMORY UPDATE
GUARDRAIL
MODEL FALLBACK
MCP TOOL
ERROR

仍只支持现有：

print
json
md

不要开发 Tracing / Dashboard / Metrics 平台。

---

13. 本期不做

Agent Peer-to-Peer
Agent Handoff
Swarm
Reflection Strategy
Tree Search
复杂 Planning DSL
复杂 Memory Platform
自研 Memory Compression
自研 Structured Output
Skill Marketplace
正式 Observability
Testing / Eval Framework
大型 MCP 管理平台
复杂权限中心

---

14. 验收

必须人工跑通：

ReAct

原 ReAct Agent 正常运行

PlanExecute

复杂任务
→ Plan
→ 多 Step
→ Tool / Skill / SubAgent
→ Re-plan
→ Final

Multi-Agent

Main
→ SubAgent A
→ Main
→ SubAgent B
→ Main
→ Final

SubAgent A / B 不直接通信。

Structured Output

response_format
→ LangChain with_structured_output()
→ structured_response

ReAct / PlanExecute 都可用。

Short-Term Memory

同 session_id
→ 保持上下文
→ 达到阈值后使用现成 Summarization 压缩

Long-Term Memory

session-A + memory_id=user-1
→ 保存重要 Memory

session-B + memory_id=user-1
→ 能重新读取相关 Memory

长期记忆由 LangGraph Store 保存，提取/更新/合并使用 LangMem。

HITL

Tool
→ interrupt
→ approve/reject/edit
→ resume
→ Final

ReAct / PlanExecute 都可恢复。

MCP

MCP Server
→ MCP Tool
→ 统一 Tool Runtime
→ Result

Guardrail / Fallback

基础 Guardrail 和 Model Fallback 正常工作。

完成以上内容即结束 Phase 3。