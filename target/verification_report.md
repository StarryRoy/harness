# 三阶段目标验收报告

验收日期：2026-09-08

## 结论

**当前不能认定 `target/` 中的三个阶段均已达标。**

从静态实现看，三个阶段要求的主要 API 和扩展点大多已经存在；但是三个目标都把
“真实/完整/人工跑通”列为验收条件，而仓库目前只有 fallback 与长期记忆边界的 8 个
单元测试原先没有覆盖 ReAct、Tool Loop、Skill、SubAgent、Session、Middleware、
PlanExecute、Structured Output、HITL、MCP、Guardrail 等验收链路。本次环境又无法从
包索引安装 LangChain/LangGraph/LangMem 依赖，所以现有测试在收集阶段即停止，无法用
动态证据确认这些链路可运行。

| 阶段 | 静态实现 | 动态验收 | 判定 |
| --- | --- | --- | --- |
| Phase 1 | 核心对象、ReAct 图、Tool Loop、Skill progressive disclosure、四种调用接口均可定位 | 四个规定场景均无自动化或人工运行记录 | **未完全达标（待动态验收）** |
| Phase 2 | SubAgent as Tool、Middleware、Session、业务 State、Skills Advanced 均可定位 | SubAgent/Session/Middleware/Skills Advanced 场景均未运行 | **未完全达标（待动态验收）** |
| Phase 3 | PlanExecute、Structured Output、LangMem/Store、HITL、Guardrail、Fallback、MCP、异常类均可定位 | fallback、memory adapter 与 Middleware 错误边界有单元测试；其余端到端场景未运行 | **未达标** |

## 验收方法与结果

### 已执行检查

1. `python -m compileall -q src`：通过。
2. `ruff check src tests`：通过（`All checks passed!`）。
3. `pytest -q`：失败于测试收集，`langchain_core` 未安装；0 个测试得到执行。
4. `python -m pip install -r requirements.txt`：环境访问包索引返回 `403 Forbidden`，无法补齐依赖。
5. 静态阅读 `src/agent_harness/` 全部 15 个模块、3 个目标文件及测试文件。

> 编译与 lint 只能证明语法和静态风格，不等价于目标文件要求的运行验收。

## Phase 1

### 已实现证据

- 公开 `Agent` 提供 `invoke`、`ainvoke`、`stream`、`astream`，工厂返回该对象。
- `AgentRuntime` 通过 `AgentStrategy.build_graph()` 获得图；ReAct 图由 Strategy 构建，而非写死在 Runtime。
- ReAct 路由是 Model → Tool → Model，并有 `max_iterations` 保护；Tool 结果写回 `ToolMessage`。
- Skill 有 metadata、loader、registry、依赖字段和按需加载工具。初始上下文只渲染候选的 name/description，加载后才加入完整 Skill 指令；reference/resource 不会自动注入。
- State 复用 LangChain message 类型，并包含 iteration、runtime metadata 及 Skill state。

### 未满足/未证明

- 验收标准明确要求场景 A（无 Tool）、B（单 Tool）、C（连续多 Tool）、D（Skill → Tool）
  “真实运行”；现有测试没有覆盖任何一个场景。
- 同步、异步及两种流式 API 虽存在，但没有运行测试证明其结果和 checkpoint 行为正确。

**判定：静态设计基本符合，但按目标的运行验收口径仍不能签字通过。**

## Phase 2

### 已实现证据

- `Agent.as_tool()` 只接受 task，返回 content/status/metadata/error；`create_agent(subagents=...)`
  自动把 SubAgent 包装为工具，且调用不传 Main Agent 的 messages/state/session。
- 默认 Middleware 包含 CallLimit、Retry、Timeout；Pipeline 的 before、wrapper、after 顺序明确且稳定。
- `session_id` 被转换成带 agent namespace 的 hash thread id；默认 checkpointer 可替换。
- Harness State 与业务 schema 合并并拒绝保留字段冲突；业务 runtime context 集中由 Context Manager 拼装。
- Skills 支持依赖拓扑、循环/缺失检查、required tools、按需 reference/resource、受控脚本、版本选择、候选筛选和轮次卸载。

### 未满足/未证明

- 没有运行 Main → SubAgent A → Main → SubAgent B → Main 的验收链路。
- 没有运行 SubAgent + Skill、Session A/B 隔离、同 Session 多轮、业务 State、Middleware
  默认与自定义顺序、Skill 依赖/脚本/生命周期/隔离等验收链路。
- 默认同步 `TimeoutMiddleware` 明确不执行超时，仅异步路径可取消；这是合理限制，但若验收方把
  “默认 timeout 生效”理解为同步接口也必须强制超时，需要进一步确认或补充 provider timeout。

**判定：静态能力覆盖较完整，但缺少目标要求的完整运行证据。**

## Phase 3

### 已实现证据

- `PlanExecuteStrategy` 复用同一图构建、Tool、Skill、SubAgent、Middleware 与 checkpoint，具有
  `max_steps`、`max_replans`、plan/step 状态、advance、replan 和 final synthesis。
- Planner 和最终 response format 都调用 LangChain `with_structured_output()`。
- 短期摘要使用 LangMem `SummarizationNode`；长期记忆使用 LangGraph Store 与 LangMem
  `create_memory_store_manager()`，namespace 默认按 agent + memory_id 隔离。
- HITL 使用 LangGraph `interrupt()` / `Command(resume=...)`，支持 approve/reject/edit。
- Guardrail、fallback 和 MCP adapter 都进入现有 Middleware/Tool runtime，没有第二套 runtime。
- 公共异常层次列出了目标要求的 10 种异常，并为 Model、Tool、SubAgent、Middleware、Memory、
  MCP、HITL 等边界保留 cause；SubAgent 的隔离结果同时提供稳定的 `error_type` metadata。
- HITL resume 现在和普通 invoke 一样执行 `before_agent` / `after_agent` Middleware 生命周期。

### 未满足/未证明

1. **绝大多数强制人工验收链路没有证据。** PlanExecute、replan、structured output、摘要、
   cross-session memory、HITL 三种决策、ReAct/PlanExecute 恢复、MCP、三类 guardrail 均无测试。
2. **现有测试范围不足。** `tests/test_fallback_order.py` 只检查 fallback 在 primary retry 耗尽后
   执行以及 fallback 的 tool/structured binding；`tests/test_memory.py` 只检查 LangMem manager
   参数和 namespace 隔离；新增的错误边界测试只验证 Middleware 同步/异步 hook 的稳定异常，
   都不是目标描述的完整端到端流程。

**判定：未达标。主要阻塞是缺少端到端验收。**

## 达标所需最小补充

1. 增加一个确定性 fake `BaseChatModel` 与测试 tools，覆盖 Phase 1 的 A/B/C/D，以及四种调用 API。
2. 增加 Phase 2 集成测试：双 SubAgent 顺序、SubAgent Skill、Session 同 ID 保持/异 ID 隔离、
   Middleware 顺序、业务 State 冲突、Skill dependencies/assets/scripts/version/lifecycle/isolation。
3. 增加 Phase 3 集成测试：PlanExecute 正常/replan/max bound、ReAct 与 PlanExecute structured output、
   LangMem summarization、cross-session memory、HITL approve/reject/edit + resume、Guardrail、MCP tool
   runtime、fallback sync/async。
4. 在可安装依赖的环境运行完整测试，并保存人工验收命令、输入、关键输出和通过结论。
