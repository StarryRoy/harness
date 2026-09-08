# 三阶段目标验收报告

验收日期：2026-09-08

## 结论

在当前仓库的确定性验收范围内，`target/` 中 Phase 1、Phase 2、Phase 3 的核心执行链路
均已跑通。本次没有替换现有 Agent / Runtime / Strategy 架构，修复集中在 Context 生命周期、
LangGraph 检查点边界、SubAgent 调用映射和 Middleware wrapper 语义。

| 阶段 | 当前结论 | 主要动态证据 |
| --- | --- | --- |
| Phase 1 | 通过 | 普通对话、单/多 Tool、Skill 渐进加载、同步/异步/流式 API |
| Phase 2 | 通过 | SubAgent、Session、Middleware、业务 State、Skills Advanced、隔离 |
| Phase 3 | 通过 | PlanExecute、摘要、长期记忆、HITL、Fallback、Guardrail、MCP |

## 自动检查结果

- `python -m pytest -q`：44 passed。
- `ruff check .`：All checks passed。
- `ruff format --check .`：全部文件已格式化。
- `python -m compileall -q src tests`：通过。
- `python -m pip check`：No broken requirements found。
- `import agent_harness`：通过。
- `git diff --check`：退出码 0，仅提示工作区 LF/CRLF 自动转换。

测试有一条来自第三方 `trustcall` 的 LangGraph 1.x 弃用警告；不是本仓库代码触发的失败。

## 本次正确性修复

1. Context 不再把 `summarized_messages` 当作永久完整历史，而是组合 LangMem 压缩快照与
   快照后追加的真实消息。同步、异步均验证 `Summary -> Tool Call -> Tool Result -> Model`
   时最新 `ToolMessage` 可见。
2. PlanExecute 入口调整为 `prepare -> summarize（按需）-> planner -> model`。长 Session
   已有摘要后，Planner 输入同时包含本轮最新 HumanMessage 与最新摘要。
3. 一条 AIMessage 中的多个 Tool Call 改为逐个图步骤执行并逐步检查点。普通副作用 Tool
   在后续敏感 Tool 暂停、approve/reject/edit 恢复后不会重放。
4. Main 调用 SubAgent 时，使用 parent execution、SubAgent 名称和 tool-call id 派生稳定
   子 Session。SubAgent 的 LangGraph interrupt 会传播到 Main，恢复 Main 后继续同一子执行；
   ReAct 与 PlanExecute 链路均已覆盖。
5. Main 的稳定 `memory_id` 会传给启用长期记忆的 SubAgent；Store namespace 仍包含
   agent name，因此 Main、SubAgent A、SubAgent B 默认隔离。已验证跨 Main Session 复用 A
   的记忆且 B 不可见。
6. Fallback 使用“从当前 Fallback Middleware 之后继续”的同一 model wrapper 链，避免
   自身递归，同时让降级模型继续受 Retry、CallLimit、Timeout 和自定义 wrapper 约束。
7. 默认 Middleware 顺序改为 Retry 外层、CallLimit 内层，使每次真实 Model attempt 和启用
   Tool retry 后的每次 Tool attempt 分别计数；达到上限后会在下一次底层调用前终止。
8. 同步 Timeout 保留安全的内联执行契约，不创建无法取消的后台线程；异步路径继续由
   `asyncio.wait_for` 强制超时。README 已明确 `timeout_seconds` 的这一边界，并有副作用只
   执行一次的回归测试。
9. Skill 版本排序键统一为可比较结构；无限定 `get("analysis")` 在无版本、1.0、2.0 混合
   注册时选择最高显式版本，显式版本仍可精确获取。

## 保留并复验的阶段能力

- ReAct 连续 Tool loop、Tool observation 回写及最大迭代保护。
- Skill dependencies、required_tools、references/resources、受控 script、动态发现与生命周期。
- Session Checkpointer 隔离、业务 State 扩展和 Middleware 生命周期顺序。
- PlanExecute 多步骤、失败重规划、最终合成与 Structured Output。
- LangGraph Store + LangMem manager 的跨 Session 长期记忆。
- Guardrail pass/reject/modify、统一错误边界与 debug 事件。
- 本地 stdio MCP Server Tool 进入统一异步 Tool Runtime。

## 明确限制

- 同步 Python 调用无法被 Harness 安全强制取消；同步超时必须依赖 Provider/Tool 客户端自身
  的 timeout，或改用异步 Agent API。
- 自动验收使用确定性的 `BaseChatModel` 测试替身，不代表任何外部 LLM Provider 的网络、
  配额或特定模型 Tool Calling 兼容性；MCP 验收使用本地 stdio Server。
- 长期记忆的语义检索质量取决于调用方注入的 Store/vector index/embedding 配置，Harness
  不选择或硬编码 embedding provider。
