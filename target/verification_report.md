# 三阶段目标验收报告

验收日期：2026-09-08

## 结论

**`target/` 中 Phase 1、Phase 2、Phase 3 的核心验收目标当前均已通过。**

本次使用确定性 `BaseChatModel` 测试替身驱动真实 LangGraph 图运行，并对 LangMem、
Checkpoint、HITL 和本地 stdio MCP Server 执行了集成验证。测试不依赖外部 LLM API，
因此可以稳定重复运行。

| 阶段 | 判定 | 动态验收证据 |
| --- | --- | --- |
| Phase 1 | **通过** | 普通 Agent、单 Tool、连续 Tool、Skill 渐进加载、同步/异步/流式 API 均通过 |
| Phase 2 | **通过** | 双 SubAgent、SubAgent + Skill、隔离、Session、Middleware、业务 State、Skills Advanced 均通过 |
| Phase 3 | **通过** | PlanExecute、Replan、Structured Output、摘要、长期记忆、HITL、Guardrail、Fallback、MCP 均通过 |

## 自动检查结果

1. `python -m pytest -q`：**29 passed**。
2. `ruff check src tests`：**All checks passed**。
3. `python -m compileall -q src`：通过。
4. `python -m pip check`：`No broken requirements found`。
5. `git diff --check`：通过。
6. Editable 安装及 `import agent_harness`：通过。

测试过程会看到一条来自 `trustcall` 的 LangGraph 1.x 弃用警告；警告位于第三方依赖，
不影响当前功能和验收结果。

## Phase 1

已验证：

- `create_agent()` 返回公开 `Agent` 实例。
- 无 Tool 对话直接得到最终响应。
- 单 Tool 和连续多 Tool 调用均将 `ToolMessage` 正确写回历史。
- Skill 初始只暴露 name/description，调用 `load_skill` 后才注入完整指令。
- `invoke`、`ainvoke`、`stream`、`astream` 均能完成执行。
- `max_iterations` 能终止持续 Tool Loop。
- `AgentRuntime` 继续通过可替换 `AgentStrategy` 构建图，没有写死 ReAct。

## Phase 2

已验证：

- Main Agent 依次调用 SubAgent A、SubAgent B 并汇总。
- SubAgent 输入只有明确 task，结果只暴露统一业务结果结构。
- SubAgent Skill 内容、loaded state 和执行轨迹不会泄漏给 Main Agent。
- 同一 `session_id` 保持多轮消息，不同 Session 相互隔离。
- Middleware before/wrapper/after 顺序稳定。
- 默认 Retry、CallLimit 和异步 Timeout 生效。
- 业务 State 可以扩展，不能覆盖 Harness 保留字段。
- Skills Advanced 覆盖 dependencies、required_tools、reference/resource 按需读取、
  受控 script、version、动态发现、循环依赖检查和生命周期。

同步 `TimeoutMiddleware` 仍按设计直接执行，因为 Python 无法安全取消任意同步调用；
强制超时使用异步接口或底层 Provider timeout。

## Phase 3

已验证：

- `PlanExecuteStrategy` 完成多步骤执行、最终合成、失败重规划和次数边界。
- ReAct 与 PlanExecute 均通过 LangChain `with_structured_output()` 生成
  `structured_response`。
- 消息数量阈值和 token 阈值现在都会驱动 LangMem `SummarizationNode` 真正压缩历史。
- LangMem 提取的长期记忆通过 LangGraph Store 在不同 Session、相同 `memory_id` 下复用。
- HITL 的 approve、reject、edit 均通过；PlanExecute 恢复后保留 plan/current step/result。
- Guardrail 的 pass/reject/modify 路径和 Model Fallback 顺序通过。
- 本地 stdio MCP Server 的 Tool 能加载为标准 Tool，并进入统一异步 Tool Runtime。
- Debug 覆盖 PLAN、REPLAN、HITL、MEMORY、MCP、ERROR，以及本次补齐的
  `GUARDRAIL` 和 `MODEL FALLBACK` 事件。
- 统一异常边界及原始 cause 保留测试通过。

## 本次修复

1. 修复“达到消息数量阈值但 LangMem 不实际摘要”的问题。摘要 token counter 现在同时
   表达 token 与消息数量预算，最终压缩仍完全交由 LangMem 执行。
2. 为 Guardrail 的 input/tool/output 决策补充 `GUARDRAIL` Debug 事件。
3. 为主模型失败后的每次降级尝试补充 `MODEL FALLBACK` Debug 事件。
4. 增加三阶段端到端验收测试和上述问题的回归测试。
5. 清理原有 Ruff import、`__all__`、生成器和宽泛异常检查问题。

完成以上修改后，三个阶段按目标文件列出的核心验收链路均可重复运行并通过。
