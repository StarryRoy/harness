# Phase 2 Prompt：SubAgent + Middleware + Session/Context + Skills Advanced

你现在要在 **已经完成 Phase 1 的 Agent Harness 项目** 上继续开发 Phase 2。

请先阅读现有代码，复用 Phase 1 已有的 `Agent`、`create_agent()`、`AgentRuntime`、`AgentStrategy`、`ReActStrategy`、Tool 接入和 Skills Core。

**不要推翻 Phase 1 重写，也不要用 LangChain `create_agent()` 替换现有 Runtime。**

本期目标是：让 Harness 支持公司内部最重要的 **Main Agent + SubAgent** 架构，并补齐 Middleware、多轮 Session、基础 Context Manager，以及可长期使用的 Skills Advanced。

---

# 一、延续的总体原则

1. LangGraph 继续作为 Runtime 内核。
2. LangGraph 已有的 State / Checkpoint / Streaming 等能力直接使用。
3. LangChain 只作为成熟组件库选择性复用。
4. 不开发 RAG、数据库、文件等具体通用工具。
5. 不开发前端。
6. 不开发测试框架和 Eval 平台。
7. Observability 仍然只保留最简单 Debug。
8. 不开发 Phase 3 的 Plan & Execute、完整 HITL、MCP 等高级功能。

---

# 二、本期目标

Phase 2 完成后必须支持：

```python
nvh_agent = create_agent(
    name="nvh_agent",
    model=model,
    instructions="...",
    tools=[...],
    skills=[...],
)

cae_agent = create_agent(
    name="cae_agent",
    model=model,
    instructions="...",
    tools=[...],
    skills=[...],
)

main_agent = create_agent(
    name="main_agent",
    model=model,
    instructions="...",
    subagents=[nvh_agent, cae_agent],
    middleware=[...],
)
```

执行：

```text
User
↓
Main Agent
├→ NVH Agent
│   ├→ Skill
│   └→ Tool
├→ CAE Agent
│   ├→ Skill
│   └→ Tool
↓
Main Agent 汇总
↓
Final
```

并支持真正的多轮 Session。

---

# 三、SubAgent / Supervisor

公司内部不同业务板块相对独立，因此本期采用：

> **Main Agent → SubAgent as Tool → Main Agent 汇总**

这是一项核心能力。

---

## 1. Agent.as_tool()

为 `Agent` 增加：

```python
agent.as_tool(...)
```

它把一个 Agent 转换成主 Agent 可以调用的 Tool。

至少暴露：

```text
name
description
input schema
output schema
```

内部执行仍调用该 SubAgent 自己的 Runtime，例如：

```text
AgentTool
↓
sub_agent.invoke()/ainvoke()
↓
SubAgent Result
```

不要复制一份 SubAgent Runtime。

---

## 2. create_agent(subagents=...)

`create_agent()` 增加：

```python
subagents=[...]
```

Harness 自动把 SubAgent 转换成 Main Agent 可使用的 Tool 能力。

应用层不应该手动重复写：

```python
sub_agent.as_tool()
```

才能完成常用场景；`as_tool()` 应保留给高级自定义使用。

---

## 3. SubAgent 独立性

每个 SubAgent 必须保留自己的：

```text
instructions
tools
skills
state
context
runtime
middleware
```

不同业务团队可以独立开发自己的 Agent。

Main Agent 不读取 SubAgent 的完整内部 State。

---

## 4. SubAgent Context Isolation

默认策略：

```text
Main Agent 根据当前问题/任务
↓
生成给 SubAgent 的任务输入
↓
SubAgent 在自己的 Context 中执行
↓
返回最终结果
```

禁止默认把 Main Agent 全部消息、全部 State、全部 Skill 内容完整传给 SubAgent。

需要保留清晰的 Context 边界。

---

## 5. SubAgent Result Contract

定义统一 SubAgent 返回结构。

至少包括：

```text
content
status
metadata
error
```

要求：

1. Main Agent 默认只获取业务结果。
2. 不把 SubAgent 完整内部 State 塞回 Main Agent Context。
3. Error 能被 Main Agent 感知并继续决定下一步。
4. metadata 保持轻量，便于以后扩展。

---

## 6. 多次 SubAgent 调用

ReAct 主 Agent 必须支持：

```text
Main
→ Agent A
→ Main
→ Agent B
→ Main
→ Final
```

是否继续调用其他 SubAgent 由 Main Agent ReAct 决定。

本期不开发复杂固定 Supervisor Workflow。

---

## 7. 本期 Multi-Agent 不做

```text
Handoff
Peer-to-peer Agent Network
Swarm
复杂 DAG Multi-Agent
自动负载调度
复杂 Agent 并发框架
```

---

# 四、Middleware Framework

本期正式建立 Middleware 扩展机制。

Middleware 定义：

> 对 Agent Runtime 生命周期中的通用横切能力进行可插拔封装。

它不是 Tool，也不是 Skill。

---

## 1. Middleware Contract

建立轻量统一接口，例如：

```python
class AgentMiddleware:
    def before_agent(...): ...
    def before_model(...): ...
    def wrap_model_call(...): ...
    def after_model(...): ...
    def wrap_tool_call(...): ...
    def after_agent(...): ...
```

可以根据 Python 代码风格使用 sync/async 兼容设计，但不要创造复杂框架。

---

## 2. Pipeline

Runtime 统一执行 Middleware。

必须保证执行顺序稳定、可预测。

本期选择一个简单方案即可：

```text
注册顺序
```

或：

```text
priority
```

不要开发 Middleware Dependency Graph。

---

## 3. 首批 Built-in Middleware

只实现最有价值的少量功能：

```text
RetryMiddleware
TimeoutMiddleware
CallLimitMiddleware
```

如果 Timeout 受到底层库限制，保证接口设计合理并实现可行部分，不要为了 Timeout 重造 Runtime。

### Retry

至少支持 Model Call / Tool Call 中合理的 retry 配置。

### CallLimit

至少能限制：

```text
max_model_calls
max_tool_calls
```

避免异常死循环。

---

## 4. 一级能力不要强迫应用层手写 Middleware

例如 Skills 仍然是：

```python
skills=[...]
```

Session 仍然是 Agent API。

即使内部部分能力通过 Middleware 实现，也不要泄漏给普通应用开发者。

---

# 五、Session + Checkpoint

本期增加真正的多轮 Session。

应用层示例：

```python
agent.invoke(
    "继续刚才的问题",
    session_id="session-001",
)
```

Harness 内部将：

```text
session_id
↓
映射到 LangGraph thread_id
↓
Checkpointer
```

要求：

1. 同一 `session_id` 可以连续多轮对话。
2. 不同 Session 状态隔离。
3. 使用 LangGraph Checkpointer，不自己实现另一套 Persistence。
4. 应用层不需要理解 `thread_id`。
5. Checkpointer 实现应可替换，例如开发期内存，后续可换持久化实现。

不需要本期开发复杂数据库 Checkpoint 管理平台。

---

# 六、Context Manager

Phase 1 只有基础 Context，本期建立明确的 Context 组装机制。

本次 Model Call 的 Context 可能包含：

```text
Agent Instructions
Session Messages
Loaded Skills
Tool Results
SubAgent Results
Runtime Context
```

本期重点不是复杂算法，而是：

```text
谁能进入 Context
进入顺序
什么时候保留
什么时候删除
```

---

## 1. Context 组装顺序

请定义清晰、统一、容易维护的 Context 构建流程。

不要让各个 Node 自己拼 Prompt。

---

## 2. Tool Result Context

避免非常大的 Tool Result 永久无限堆积。

本期可以先做简单策略，例如：

```text
保留最近结果
或对超大结果进行长度限制/简单处理
```

不要开发复杂 Context Compression Engine。

---

## 3. SubAgent Result Context

Main Agent 默认只加入 SubAgent 的最终 Result，不加入内部完整轨迹。

---

## 4. Basic Summarization

支持长 Session 的基础 Summarization。

建议：

```text
达到阈值
↓
摘要旧 messages
↓
保存 summary
↓
保留最近消息
```

优先复用成熟能力。

不自研复杂摘要算法，不做长期 Memory。

---

# 七、Skills Advanced

Skills 是本项目核心，本期需要从“能用”提升到“适合各业务团队长期维护”。

---

## 1. Skill Dependencies

真正实现：

```text
Skill A depends on Skill B
```

至少支持：

```text
依赖检查
自动加载必要依赖
循环依赖检测
缺失依赖错误
```

不要做类似 pip 的复杂 Resolver。

---

## 2. required_tools

完善 Phase 1 的 `required_tools`。

加载 Skill 前：

```text
检查 required_tools
↓
缺失则拒绝加载
↓
返回清晰错误
```

---

## 3. References 按需加载

Skill 结构：

```text
SKILL.md
references/
```

不能默认把所有 references 全部塞入 Context。

提供统一的按需读取能力。

---

## 4. Resources

正式支持：

```text
resources/
```

用于：

```text
模板
示例
配置
静态业务资源
```

提供统一访问接口。

不要把资源内容全部自动加入 Prompt。

---

## 5. Scripts

正式支持：

```text
scripts/
```

核心思想：

```text
LLM 负责判断与编排
Script 负责确定性执行
```

要求：

1. Skill 可以声明/引用自己的 Script。
2. Script 执行走统一入口。
3. 不允许 Skill 随意执行任意系统路径代码。
4. 基础参数、返回值、错误能够统一处理。
5. 本期不需要做完整 Sandbox 平台。

---

## 6. Skill Validation

Skill 加载时统一检查：

```text
目录结构
Metadata
SKILL.md
required_tools
dependencies
references
resources
scripts
```

错误必须明确指出 Skill 和具体问题。

---

## 7. Skill Version

支持：

```text
name
version
```

至少可以区分：

```text
skill-a@1.0
skill-a@2.0
```

不开发发布平台、远程 Registry、Marketplace。

---

## 8. Skill Compatibility

实现轻量兼容约束即可，例如：

```text
required_tools
minimum_harness_version
required_capabilities
```

只做当前实际有价值的约束。

---

## 9. Dynamic Skill Discovery

当 Agent Skill 较多时，不应永远把全部描述都放进 Context。

实现一个轻量选择机制。

可采用简单方式：

```text
Metadata Filter
关键词
简单语义匹配（如现有组件实现容易）
```

目标：先筛出少量候选 Skill，再将其 name/description 暴露给 Agent。

不要开发独立 Skill Search Service。

---

## 10. Skill Context 生命周期

明确：

```text
什么时候加载
什么时候 active
什么时候保持
什么时候卸载
什么时候只留下必要结果
```

避免长 Session 中已加载 Skill 内容无限累积。

---

## 11. SubAgent Skill Isolation

不同 Agent 必须保持：

```text
Skill Registry 独立
Loaded Skill State 独立
Active Skill 独立
Skill Context 独立
```

Main Agent 不自动继承所有 SubAgent Skills。

---

# 八、Debug / Observability

继续保持极低优先级。

在 Phase 1 基础上，只增加必要事件：

```text
SUBAGENT CALL
SUBAGENT RESULT
MIDDLEWARE
SESSION
SKILL SCRIPT
```

仍然可以使用：

```text
print
JSON
MD
```

不要开发正式 Tracing / Metrics / Dashboard。

---

# 九、Testing

**仍然不开发测试框架或 Eval 平台。**

不要因为增加 SubAgent/Skill 就主动建设大规模 mock、benchmark、dataset。

只保证代码职责清晰、依赖可注入，未来能测试即可。

---

# 十、建议代码结构调整

在 Phase 1 代码基础上按需扩展，例如：

```text
harness/
├── agent.py
├── definition.py
├── runtime.py
├── state.py
├── strategy/
│   ├── base.py
│   └── react.py
├── subagents/
│   ├── tool.py
│   └── result.py
├── middleware/
│   ├── base.py
│   ├── pipeline.py
│   ├── retry.py
│   ├── timeout.py
│   └── limits.py
├── session.py
├── context.py
├── skills/
│   ├── models.py
│   ├── loader.py
│   ├── registry.py
│   ├── validation.py
│   └── scripts.py
└── debug.py
```

结构可以根据现有项目优化，不要机械重构。

---

# 十一、本期明确不做

```text
Plan & Execute
Reflection
HITL 完整实现
Tool Approval
Guardrail 系统
MCP
Long-Term Memory
正式 Observability
OpenTelemetry
LangSmith Integration
Testing Framework
Agent Eval
复杂权限平台
Skill Marketplace
Skill Governance Platform
复杂 Sandbox
任何 RAG / DB / 文件等具体通用工具研发
```

---

# 十二、Phase 2 验收标准

必须完整跑通以下场景。

## 场景 A：Main + 一个 SubAgent

```text
User
→ Main
→ SubAgent
→ Main
→ Final
```

## 场景 B：Main + 多 SubAgent

```text
User
→ Main
→ Agent A
→ Main
→ Agent B
→ Main
→ Final
```

## 场景 C：SubAgent 使用 Skill + Tool

```text
Main
→ SubAgent
→ Skill Discovery
→ Load Skill
→ Tool/Script
→ SubAgent Final
→ Main
```

## 场景 D：多轮 Session

```text
session-001 第 1 轮
session-001 第 2 轮继续上下文
session-002 与 session-001 隔离
```

## 场景 E：Middleware

Retry / CallLimit 等能够通过统一 Middleware Pipeline 工作，不侵入具体业务 Agent。

## 场景 F：Advanced Skills

至少验证：

```text
依赖 Skill
required_tools
reference 按需读取
resource 访问
script 统一执行
Skill version
Skill validation
Context 生命周期
SubAgent Skill Isolation
```

达到以上闭环即可结束 Phase 2，不要主动进入 Phase 3。
