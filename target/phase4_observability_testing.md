# 第四期：Observability 与测试体系

## 目标

建立统一、可扩展的运行观测能力，并把现有测试升级为稳定的 Harness 级测试体系。重点验证 Runtime 语义，而不是追求表面覆盖率。

## 1. 统一运行事件模型

新增稳定的运行事件结构，覆盖：

- Agent start / end / error
- Model start / end / retry / fallback
- Tool start / end / error
- SubAgent start / end / error
- Skill discover / load / unload / script
- HITL pause / resume / approve / reject / edit
- Plan create / step / replan / complete / fail
- Memory load / update
- Context summarize

事件至少包含：

- `event_type`
- `timestamp`
- `agent_name`
- `session_id`
- `trace_id`
- `span_id`
- `parent_span_id`
- `status`
- `duration_ms`
- `metadata`
- `error`

不要让业务层依赖 LangGraph node 名称或内部 state 字段。

## 2. Observer / EventSink 抽象

新增统一观察接口，例如：

```python
class EventSink(Protocol):
    def emit(self, event: RuntimeEvent) -> None: ...
```

支持：

- 单个或多个 EventSink
- Console / JSON 调试输出
- 自定义企业日志平台
- 后续接入 OpenTelemetry / LangSmith 的扩展点

现有 `DebugHandler` 改为 EventSink 的一个轻量实现，不再让 Runtime 到处直接 `print`。

## 3. Trace 与调用链传播

统一维护：

```text
Main Agent
  -> Model
  -> Tool
  -> SubAgent
      -> Model
      -> Tool
```

要求：

- 一次用户请求共享同一个 `trace_id`
- 每次 Model / Tool / SubAgent 调用拥有独立 `span_id`
- SubAgent 自动继承 Main Agent 的 `trace_id`
- 记录 `parent_span_id`
- HITL resume 后继续原 trace，而不是创建无关联的新调用链
- sync / async 行为一致

## 4. 基础指标

至少记录：

- Agent 总耗时
- Model 调用次数与耗时
- Tool 调用次数与耗时
- SubAgent 调用次数与耗时
- Retry / Fallback 次数
- HITL 暂停次数
- Skill 加载次数
- Context summary 次数
- Memory load / update 次数
- 可获取时记录模型 token usage

指标采集失败不能影响 Agent 正常执行。

## 5. 观测数据安全

增加统一脱敏与限制机制：

- API Key、Token、密码等敏感字段默认不进入事件
- Tool 参数与结果支持截断
- Model messages 默认不要求完整记录
- 可通过配置控制 payload 详细程度
- Observer 异常不能破坏 Agent 主流程

## 6. 应用层 Streaming 事件

在 LangGraph 原始 stream 之上定义稳定的应用层事件，例如：

```text
text_delta
tool_start
tool_end
subagent_start
subagent_end
approval_required
plan_update
final
error
```

应用层默认消费稳定事件，不需要理解 LangGraph chunk / node / state。

高级用户可保留访问 raw stream 的接口。

Streaming 事件与内部 Observability 事件共享 trace 信息，但不要把内部调试数据全部暴露给前端。

## 7. 测试分层

测试目录按职责拆分：

```text
tests/
├── unit/
├── contract/
├── integration/
├── acceptance/
└── regression/
```

### Unit

测试单个组件：

- Context
- Middleware
- SkillRegistry / SkillSelector
- Memory
- Result / Event
- Persistence configuration
- Error mapping

### Contract

固定 Harness 公共 API 行为：

- `create_agent`
- `invoke / ainvoke`
- `stream / astream`
- `resume / aresume`
- `AgentResult`
- `RuntimeEvent / StreamEvent`
- 自定义 State
- Middleware 生命周期

### Integration

验证组件组合：

- 持久化 Checkpointer
- 持久化 Store
- Session 恢复
- Memory 跨 Session
- SubAgent 独立持久化
- MCP 进入统一 Runtime
- Observer / EventSink

### Acceptance

从应用开发者视角验证完整业务流程：

- 普通 ReAct
- Tool Agent
- Skill Agent
- Main + SubAgent
- Plan & Execute
- Structured Output
- 长短期记忆
- HITL

### Regression

继续固定最容易出事故的 Runtime 语义：

- HITL resume 不重复执行已有副作用
- 连续多个 HITL 能正确恢复
- SubAgent HITL 能传播到 Main
- Summary 后新 Tool Result 不丢失
- Session 相互隔离
- Main / SubAgent Memory 相互隔离
- Retry / CallLimit / Timeout / Fallback 顺序不回归
- Plan 失败、重规划耗尽后的状态正确
- sync / async 语义一致

## 8. 持久化专项测试

由于 Harness 禁止运行时内存存储，增加持久化专项验证：

- 进程重新创建 Agent 后，同一 `session_id` 可以恢复
- HITL 暂停后重新创建 Agent 仍可 resume
- 长期记忆在重新创建 Agent 后仍可读取
- Main / SubAgent namespace 隔离正确
- 不同 Agent 同名 session 不发生串线
- Checkpoint / Store 初始化失败时返回稳定 Harness Error
- Session 清理和 retention 不影响其他 session

测试不得依赖 `MemorySaver` / `InMemorySaver` / `InMemoryStore` 作为运行时持久化实现。

## 9. 测试基础设施

建立可复用的：

- Deterministic FakeModel
- Fake Tool
- Fake SubAgent
- Fake EventSink
- 临时持久化测试环境
- sync / async 参数化测试工具

减少每个测试文件重复定义模型和 Tool。

测试重点是行为与执行语义，不以单纯覆盖率作为目标。

## 10. CI

新增 GitHub Actions：

```text
ruff
pytest
python 3.10
python 3.11
python 3.12
```

要求：

- Pull Request 自动运行
- 任一关键测试失败禁止通过
- 单元测试与不需要外部服务的测试默认执行
- 需要真实持久化服务的 Integration Test 使用独立 CI service/container

## 11. 第四期完成标准

完成后应满足：

1. Runtime 不再依赖散落的 `print` 进行观测。
2. Main / SubAgent / Model / Tool 能形成完整 trace。
3. 应用层 Streaming 不需要理解 LangGraph 内部事件。
4. 关键调用可以统计耗时、次数、失败和重试。
5. 敏感信息不会默认进入日志。
6. 公共 API 有稳定 Contract Test。
7. HITL、Memory、SubAgent、PlanExecute 等关键 Runtime 语义都有 Acceptance / Regression Test。
8. 持久化能力经过重启恢复测试。
9. CI 自动执行静态检查和核心测试。
