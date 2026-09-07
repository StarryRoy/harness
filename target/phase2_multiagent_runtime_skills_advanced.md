# Phase 2：SubAgent + Middleware + Session/Context + Skills Advanced

基于当前 已经完成的 Phase 1 继续开发。先阅读现有代码，**不要推翻重构

核心原则：

* LangGraph 继续作为 Runtime。
* LangGraph 已有能力直接复用。
* 普通应用开发者应尽可能少配置，合理功能提供默认值。
* 高级能力保留可覆盖接口。
* 不开发测试体系。
* Observability 仍只保留极简 Debug。

---

# 1. SubAgent

实现公司主要多 Agent 模式：

```text
Main Agent
→ SubAgent as Tool
→ Main Agent 汇总
```

## 1.1 Agent.as_tool()

为 `Agent` 增加：

```python
agent.as_tool()
```

将 Agent 包装为标准 Tool。

至少包含：

```text
name
description
input
output
```

内部直接调用 SubAgent 自己的 Runtime。

---

## 1.2 create_agent(subagents=...)

支持：

```python
main_agent = create_agent(
    name="main",
    instructions="...",
    subagents=[
        agent_a,
        agent_b,
    ],
)
```

Harness 自动执行：

```text
SubAgent
→ AgentTool
→ Main Agent Tool List
```

应用层无需手动调用 `as_tool()`。

---

## 1.3 SubAgent 独立性

业务层只需要定义：

```text
name
description
instructions
tools
skills
model（可选）
response_format（可选）
state_schema（可选）
```

Harness 自动管理并隔离：

```text
Runtime
Harness State
Context
Session execution scope
Middleware
Checkpoint
Skill State
Debug
Error handling
```

Main Agent 不直接访问 SubAgent 完整 State。

---

## 1.4 Context Isolation

默认：

```text
Main Agent
→ 生成明确的 SubAgent Task
→ SubAgent 独立执行
→ 返回结果
→ Main Agent 继续 ReAct
```

默认禁止把 Main Agent 的：

```text
完整 messages
完整 State
loaded_skills
内部 Runtime 数据
```

全部传给 SubAgent。

---

## 1.5 SubAgent Result

定义统一内部返回结构，例如：

```text
content
status
metadata
error
```

Main Agent 默认只接收业务结果和必要错误信息，不接收 SubAgent 完整执行轨迹。

---

# 2. 默认配置原则

Phase 2 开始正式贯彻：

> 默认开箱即用，高级用户按需覆盖。

普通 Agent：

```python
agent = create_agent(
    name="nvh",
    instructions="...",
    tools=[...],
    skills=[...],
)
```

即可运行。

Harness 应提供合理默认值，例如：

```text
model
strategy = ReAct
middleware
retry
timeout
call_limit
context policy
checkpointer
subagent isolation policy
debug
```

高级用户有需求时允许覆盖。

不要要求普通应用开发者理解 LangGraph Runtime 细节。

---

# 3. Middleware

建立统一 Middleware 扩展机制。

至少支持生命周期：

```text
before_agent
before_model
wrap_model_call
after_model
wrap_tool_call
after_agent
```

Middleware Pipeline 必须有稳定执行顺序。

本期直接使用简单方案：

```text
注册顺序
```

不要开发复杂依赖图。

---

## 3.1 默认 Middleware

提供少量默认能力：

```text
RetryMiddleware
TimeoutMiddleware
CallLimitMiddleware
```

普通应用层不传 `middleware` 时自动使用默认配置。

高级用户允许：

```python
create_agent(
    ...,
    middleware=[...],
)
```

覆盖或扩展默认行为。

Middleware 是高级扩展接口，不要求普通业务开发者理解。

---

# 4. State 扩展

Harness 保留自己的默认 State，例如：

```text
messages
iteration
runtime_metadata
loaded_skills
active_skill
```

这些由 Harness 自动维护。

同时允许高级业务定义额外业务状态：

```python
class BusinessState(TypedDict, total=False):
    project_id: str
    approval_status: str
```

通过：

```python
create_agent(
    ...,
    state_schema=BusinessState,
)
```

实现。

最终概念上是：

```text
Harness State
+
Business State
```

业务自定义 State 不能覆盖 Harness 保留字段。

---

# 5. Session

公开高层接口：

```python
agent.invoke(
    "...",
    session_id="session-001",
)
```

应用层只使用 `session_id`。

Harness 内部负责：

```text
session_id
→ LangGraph thread_id
→ Checkpointer
```

不要向普通应用层暴露 `thread_id`。

要求：

* 同一 `session_id` 支持多轮。
* 不同 Session 隔离。
* Checkpointer 可替换。
* 不重新实现 LangGraph Persistence。

---

# 6. Context Manager

建立统一 Context 组装逻辑。

Model Context 主要来自：

```text
Agent Instructions
Session Messages
Loaded Skills
Tool Results
SubAgent Results
Business Runtime Context
```

要求：

* 不让各 Node 自己随意拼 Prompt。
* SubAgent 默认只返回最终结果。
* Tool Result 不允许无限堆积。
* Skill Context 有明确生命周期。

本期不做复杂 Context Engineering。

---

# 7. 基础 Summarization

长 Session 支持简单摘要：

```text
达到阈值
→ 摘要旧 messages
→ 保存 summary
→ 保留最近 messages
```

优先复用成熟实现。

不要自研复杂压缩算法。

---

# 8. Skills Advanced

Phase 1 已有：

```text
Skill
Metadata
Loader
Registry
Progressive Disclosure
```

本期继续增强。

---

## 8.1 Dependencies

真正实现：

```text
Skill A depends on Skill B
```

支持：

```text
依赖检查
依赖加载
缺失依赖错误
循环依赖检测
```

不要开发复杂包管理器。

---

## 8.2 required_tools

Skill 加载前检查：

```text
required_tools
```

缺失时拒绝加载并返回明确错误。

---

## 8.3 References

支持 `references/` 按需读取。

不要加载 Skill 时自动把全部 references 放进 Context。

---

## 8.4 Resources

支持：

```text
resources/
```

用于模板、示例、配置、静态资源。

提供统一读取接口，不自动全部加入 Prompt。

---

## 8.5 Scripts

支持：

```text
scripts/
```

原则：

```text
LLM 负责判断和编排
Script 负责确定性执行
```

要求：

* Skill 可以声明 Script。
* Script 统一入口执行。
* 参数、结果、错误统一处理。
* 不允许任意系统路径代码执行。
* 本期不开发完整 Sandbox。

---

## 8.6 Skill Validation

统一检查：

```text
目录
Metadata
SKILL.md
dependencies
required_tools
references
resources
scripts
```

错误必须明确指出具体 Skill 和具体问题。

---

## 8.7 Version

支持：

```text
skill_name + version
```

例如：

```text
analysis@1.0
analysis@2.0
```

不开发版本发布平台。

---

## 8.8 Dynamic Skill Discovery

当 Skill 数量较多时，不要始终把所有 Skill 描述送入模型。

先筛选候选 Skill，再暴露：

```text
name
description
```

可以使用：

```text
metadata
关键词
简单语义匹配
```

不要建设独立 Skill 搜索服务。

---

## 8.9 Skill Context Lifecycle

明确：

```text
什么时候加载
什么时候 active
什么时候保留
什么时候卸载
```

防止长 Session 中 Skill 内容无限累积。

---

## 8.10 SubAgent Skill Isolation

不同 Agent 必须保持独立：

```text
SkillRegistry
loaded_skills
active_skill
Skill Context
```

Main Agent 不继承 SubAgent Skills。

---

# 9. Debug

继续保持极简。

在 Phase 1 基础上增加必要信息：

```text
SUBAGENT CALL
SUBAGENT RESULT
SESSION
MIDDLEWARE
SKILL SCRIPT
```

仍允许：

```text
print
JSON
MD
```

不要开发正式 Tracing、Metrics、Dashboard。


# 11. 验收

必须完整跑通：

### SubAgent

```text
User
→ Main
→ SubAgent A
→ Main
→ SubAgent B
→ Main
→ Final
```

### SubAgent + Skill

```text
Main
→ SubAgent
→ Skill Discovery
→ Skill Load
→ Tool / Script
→ SubAgent Result
→ Main
```

### Session

```text
session-A 多轮保持上下文
session-B 与 session-A 隔离
```

### Middleware

默认无需业务配置即可生效，高级用户可以自定义。

### State

默认 Harness State 自动管理；业务 `state_schema` 可以增加自定义字段，不能破坏 Harness 内部字段。

### Skills Advanced

验证：

```text
dependencies
required_tools
references 按需加载
resources
scripts
validation
version
dynamic discovery
context lifecycle
SubAgent skill isolation
```

完成以上内容即结束 Phase 2
