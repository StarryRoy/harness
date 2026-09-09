# 三阶段目标验收报告

验收日期：2026-09-09

## 结论

在当前仓库的确定性验收范围内，Phase 1、Phase 2、Phase 3 的既有核心链路继续通过。
本次在不重写 ReAct、PlanExecute、SubAgent、HITL、Skill、Middleware 和 LangMem 架构的
前提下，补齐了应用层结果 API、持久化边界和 Session 生命周期，并修正 Plan 失败状态。

## 自动检查结果

- `python -m pytest -q`：59 passed。
- `ruff check .`：All checks passed。
- `ruff format --check .`：全部文件已格式化。
- `python -m compileall -q src tests`：通过。
- 官方文件型 SQLite Checkpointer 烟雾验证：自动 Session、调用与清理通过。
- `git diff --check`：通过。

测试有一条来自第三方 `trustcall` 的 LangGraph 1.x 弃用警告，不影响当前验收结果。

## 本次新增验证

- `invoke/ainvoke/resume/aresume` 返回统一 `AgentResult`；覆盖普通 output、Structured
  Output 优先返回、完整 state 访问，以及 completed/paused/error 状态。
- 未提供 `session_id` 时返回可公开使用的自动 Session ID；同步和异步 HITL 都能直接使用
  返回的 ID 恢复，结果不暴露内部 thread ID。
- `create_agent()` 缺少持久化 Checkpointer 时立即失败；显式和全局持久化配置均已验证。
  已知内存 Checkpointer/Store 会被拒绝，`memory=True` 缺少持久化 Store 时立即失败。
- `clear_session/aclear_session` 会删除转换后的内部 thread checkpoint；再次使用同一公开
  Session ID 时从空历史开始，后端异常统一包装为 `PersistenceError`。
- Skill 默认字面选择行为迁入 `LexicalSkillSelector`，并验证自定义 `SkillSelector` 注入。
- 新 `assets/` 与兼容的 `resources/` 均可按需读取；Skill Script 默认不继承完整父进程
  环境，只有 allowlist 中显式允许的业务变量可见。
- PlanExecute 在重规划耗尽后的状态已验证：无成功步骤为 `failed`，部分成功为 `partial`，
  当前失败步骤为 `failed`，未执行步骤为 `skipped`。Planner/Replanner 控制消息为
  `SystemMessage`。

## 既有封版回归

- 摘要后的最新 Tool Call/Tool Result 保持可见，Planner 在摘要后读取本轮最新输入。
- 多 Tool HITL 恢复不重放前置副作用；SubAgent 连续多次 HITL 的同步、异步及 reject 路径
  保持通过。
- SubAgent 长期记忆按 `(namespace, agent_name, memory_id)` 复用并隔离。
- Fallback 继续受 Retry、CallLimit、Timeout 和自定义 wrapper 约束。
- PlanExecute 与 Structured Output 的组合图保持通过。
- Skill dependencies、版本、动态发现、references/resources/scripts 及路径限制保持通过。
- 本地 stdio MCP Tool、Guardrail、统一错误边界和 Debug 事件保持通过。

## 明确限制

- 自动验收使用确定性 `BaseChatModel` 测试替身，不代表外部 LLM Provider 的网络、配额或
  特定模型兼容性；MCP 验收使用本地 stdio Server。
- Harness 不创建或管理数据库连接。持久化 Checkpointer/Store 的初始化、schema setup、
  连接池及生命周期由应用负责；正式部署推荐 LangGraph 官方 PostgreSQL 实现。
- 同步 Python Tool/Model 无法由 Harness 安全强制取消；同步超时依赖底层客户端，或使用
  异步 Agent API。
