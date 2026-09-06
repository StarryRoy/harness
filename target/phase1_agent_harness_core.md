# Phase 1 Prompt：Agent Harness Core + ReAct + Skills Core

你现在要在现有 Python 项目中开发 **Phase 1：Agent Harness 核心**。

本项目是公司内部 Agent Harness，不是面向 C 端的大型通用框架。目标是：**专业、可控、可扩展、尽快可用**，避免过度设计。

---

## 一、总体技术定位

开发一套基于 **LangGraph** 的 Agent Harness，让应用层工程师不需要直接操作 LangGraph 的 Node、Edge、StateGraph，就可以快速创建 Agent。

整体关系：

```text
Application
    ↓
Our Agent Harness
    ↓
LangGraph Runtime
```

### 技术边界

1. **LangGraph 是 Runtime 内核**。
2. LangGraph 已经提供的能力直接使用，不重复实现，包括但不限于：
   - StateGraph
   - State
   - Node / Edge
   - Conditional Edge
   - Command
   - Checkpoint / Persistence
   - Interrupt / Resume
   - Streaming
3. **不要使用 LangChain `create_agent()` 作为 Harness 底座**。
4. LangChain 只作为成熟组件库选择性复用，例如：
   - `BaseChatModel`
   - 各模型 Provider Adapter
   - `BaseTool`
   - `@tool`
   - Tool Schema
   - Provider Structured Output 的成熟能力
6. 不开发前端。
7. 不开发测试框架、Agent Eval、Benchmark、LLM-as-Judge。
8. Observability 本期优先级低，只做最简单 Debug 输出。

---

# 二、本期目标

Phase 1 完成后，应用层必须能够用类似下面的接口创建真实 Agent：

```python
agent = create_agent(
    name="demo_agent",
    description="示例 Agent",
    model=model,
    instructions="你是一个业务助手",
    tools=[tool_a, tool_b],
    skills=[...],
)

result = agent.invoke("帮我处理这个任务")
```

并支持：

```python
agent.invoke(...)
agent.ainvoke(...)
agent.stream(...)
agent.astream(...)
```

核心运行链：

```text
User
 ↓
Model
 ↓
Tool Call ?
 ├─ Yes → Tool → Model
 │              ↓
 │          可继续 Tool
 │
 └─ No → Final
```

必须支持连续多次 Tool 调用。

本期同时完成 **Skills Core**，Skill 必须是一等公民，而不是简单把全部 Skill 文本塞入 System Prompt。

---

# 三、核心对象设计

## 1. Agent

`Agent` 是 Harness 的核心公开类。

建议职责：

```text
Agent
├── definition
├── runtime
├── invoke()
├── ainvoke()
├── stream()
└── astream()
```

应用层拿到的最终对象必须是 `Agent`。

不要把所有执行细节直接堆在 `Agent` 类中。

---

## 2. create_agent()

`create_agent()` 是 **Agent 工厂函数**，不是另一种 Agent。

职责：

```text
接收参数
→ 基础参数校验
→ 构造 AgentDefinition
→ 创建 AgentStrategy
→ 创建 AgentRuntime
→ 构建/编译 LangGraph
→ 返回 Agent 实例
```

不要在 `create_agent()` 里直接实现完整运行逻辑。

---

## 3. AgentDefinition

独立保存 Agent 配置。

至少包括：

```text
name
description
model
instructions
tools
skills
response_format
runtime_config
```

后续会增加 subagents、middleware 等，不要把结构设计成无法扩展，但也不要为了未来提前创建大量空抽象。

---

## 4. AgentRuntime

负责真正执行 Agent。

Phase 1 主要职责：

```text
维护 State
调用 Strategy
调用 LangGraph compiled graph
处理 invoke / async / stream
基础 Debug
```

**ReAct 的具体 Graph 构建逻辑不能直接写死在 AgentRuntime 中。**

---

# 四、AgentStrategy 可替换接口

运行模式未来明确会扩展，因此 Phase 1 必须存在一个**很薄的可替换运行模式接口**。

例如可以使用 `Protocol`、`ABC` 或其他简单方式：

```python
class AgentStrategy:
    def build_graph(...):
        ...
```

本期只实现：

```text
ReActStrategy
```

不要开发：

```text
PlanExecuteStrategy
ReflectionStrategy
Strategy Registry
Strategy Plugin System
复杂 Strategy Factory
```

设计原则：

```text
AgentRuntime
    ↓
AgentStrategy
    ↓
ReActStrategy   # Phase 1 唯一实现
```

以后增加新的 Strategy 时，不应该修改 Agent/Tool/Skill 的核心使用方式。

---

# 五、ReActStrategy

基于 LangGraph 实现标准 ReAct Tool Loop。

最小 Graph：

```text
START
 ↓
MODEL
 ↓
是否存在 tool_calls？
 ├─ Yes → TOOL → MODEL
 └─ No  → END
```

要求：

1. 支持普通对话，无 Tool 时直接生成结果。
2. 支持一次 Tool 调用。
3. 支持多轮连续 Tool 调用。
4. Tool Result 必须正确写回 Message History。
5. Model 根据 Observation 继续判断下一步。
6. 有最大循环次数/最大 iteration 的简单保护，避免死循环。
7. 不自研复杂 Planner。

---

# 六、AgentState

基于 LangGraph State 定义 Harness 统一状态。

Phase 1 至少包括：

```text
messages
iteration
runtime_metadata
skill state
```

Skill 相关状态至少能够表达：

```text
available_skills
loaded_skills
active_skill
```

尽量复用 LangChain / LangGraph 已有 Message 类型，不要重新定义一整套 HumanMessage / AIMessage / ToolMessage。

---

# 七、Model 接入

直接使用 LangChain `BaseChatModel`。

应用层可以传入不同 Provider 的 ChatModel。

Harness 不负责自行实现：

```text
OpenAI Adapter
Claude Adapter
```

模型与 Tool Calling 的兼容处理优先利用现有成熟能力。

---

# 八、Tool 接入

本期只做 Harness 对 Tool 的统一接入，不开发具体通用工具。

支持：

```text
LangChain BaseTool
@tool
```

如果普通 Python callable 转 Tool 使用现有成熟组件即可，不要自研复杂 Schema 系统。

Runtime 需要完成：

```text
模型产生 Tool Call
→ 执行 Tool
→ 得到 Tool Result
→ 写入 State/messages
→ 返回 Model
```

不要创建第二套 Tool Runtime。

---

# 九、Skills Core（本期核心）

Skills 是 Harness 的一级核心能力。

Skill 定义：

> Skill 表示 Agent 完成某一类任务所需的专业方法、SOP、指令和参考资料。

Skill 不是 Tool：

```text
Tool  = 能执行什么动作
Skill = 怎么完成一类工作
```

---

## 1. Skill 目录规范

统一支持类似：

```text
skills/
└── road_noise_analysis/
    ├── SKILL.md
    ├── references/
    ├── resources/
    └── scripts/
```

Phase 1 真正要求实现：

```text
SKILL.md
metadata
references 基础读取
```

`resources/`、`scripts/` 可以识别目录结构，但高级执行能力放后续。

---

## 2. Skill 核心模型

至少实现：

```text
Skill
SkillMetadata
SkillLoader
SkillRegistry
```

### SkillMetadata 至少支持

```text
name
description
version
tags
required_tools
dependencies
```

Phase 1 中 `version/dependencies` 可以先解析和保存，不必实现完整高级管理。

---

## 3. SkillLoader

负责：

```text
读取 Skill 目录
解析 Metadata
读取 SKILL.md
读取 references
基础格式校验
返回 Skill 对象
```

错误需要明确，例如：

```text
缺失 SKILL.md
Metadata 格式错误
重复 Skill name
required_tools 缺失
```

---

## 4. SkillRegistry

负责：

```text
注册 Skill
按 name 获取 Skill
列出可用 Skill
检测重复 Skill
```

Registry 与 Agent 绑定，不设计全局复杂 Skill 平台。

---

## 5. Progressive Disclosure

这是 Phase 1 Skills 最重要的要求。

### 初始阶段

模型只看到可用 Skill 的：

```text
name
description
```

不要把所有 `SKILL.md` 全部加入 Prompt。

### 使用阶段

当 Agent 判断某个 Skill 有用时：

```text
选择 Skill
↓
load_skill
↓
加载完整 SKILL.md
↓
必要 reference
↓
注入当前 Context
↓
继续 ReAct
```

未使用的 Skill 不应完整进入 Context。

可以采用简单的内部 Skill Loader Tool / Runtime Action 实现，不需要复杂语义检索系统。

---

## 6. Skill 与 Tool

Skill 可以声明：

```text
required_tools
```

加载 Skill 时检查 Agent 是否存在所需 Tool。

缺失时返回明确错误，不静默忽略。

Tool 仍属于 Agent，不属于 Skill 自己的独立 Runtime。

---

# 十、基础 Context

本期 Context 不做复杂 Engineering。

只需要明确当前 Model Call 能够组合：

```text
Agent instructions
messages
当前已加载 Skill
Tool Result
```

不要实现复杂自动压缩、长期 Memory、复杂 Context Policy。

---

# 十一、Structured Output

使用 Provider / LangChain 现有 Structured Output 能力。

---

# 十二、Streaming

基于 LangGraph Streaming。

需要支持：

```python
invoke
ainvoke
stream
astream
```

Phase 1 不设计复杂统一 Event 协议，只保证调用过程和最终结果可用。

---

# 十三、Debug / Observability

本期 Observability 优先级极低。

只需要 `debug=True` 时能看到关键过程，例如：

```text
AGENT START
MODEL CALL
SKILL LOAD
TOOL CALL
TOOL RESULT
ERROR
AGENT END
```

实现可以非常简单：

```text
print
或写 debug.json / debug.md
```

不要开发：

```text
Tracing Framework
Metrics
Dashboard
OpenTelemetry
LangSmith Adapter
Token Cost Platform
正式 Event Bus
```

但不要把 `print()` 散落在几十个业务函数中。可以集中放一个极轻量 DebugHandler / debug helper，方便未来替换。

---

# 十四、Testing

**本期不开发测试系统。**

但需要保证后续方便开发测试系统，而不是需要大范围重构。

---

# 十五、建议代码结构

可以根据实际代码调整，但建议保持职责清晰。不要为了形式创建大量只有几行代码的目录和抽象类。



# 十七、Phase 1 验收标准

Phase 1 完成后必须可以真实运行以下场景：

### 场景 A：普通 Agent

```text
User
→ Model
→ Final
```

### 场景 B：单 Tool

```text
User
→ Model
→ Tool
→ Model
→ Final
```

### 场景 C：多 Tool Loop

```text
User
→ Model
→ Tool A
→ Model
→ Tool B
→ Model
→ Final
```

### 场景 D：Skill

```text
User
→ Model 发现需要 Skill A
→ 加载 Skill A
→ Model 根据 Skill 工作
→ Tool
→ Model
→ Final
```

同时满足：

1. `Agent` 是最终公开实例。
2. `create_agent()` 返回 `Agent`。
3. `AgentRuntime` 不写死 ReAct。
4. `AgentStrategy` 可替换。
5. 当前只有 `ReActStrategy` 一个实现。
6. 未使用 Skill 的完整内容不进入模型 Context。