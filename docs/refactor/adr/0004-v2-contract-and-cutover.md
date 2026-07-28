# ADR-0004：OpenAPI 正式契约与一次性切换

- 状态：已接受
- 日期：2026-07-28
- 对应方案：§12、§14、§16

## 决策

`openapi/v2.yaml` 是 v2 HTTP 请求、响应、错误码、枚举与 operationId 的正式契约。前端类型由 `openapi-typescript` 生成到 `vue-frontend/src/api/generated/v2.ts`，不得另维护一套手写 v2 DTO。

阶段 0 优先冻结：

- 页面 batch document mutation 与整页 revision CAS。
- 统一 page repair multipart 的 bubble/mask `oneOf`。
- operation/job 状态和 kind 闭集。
- `Idempotency-Key`。
- paused 的 resume 与 interrupted 的 continue 分离。
- Studio session revision/generation fencing。

v2 在独立入口、独立蓝图和独立 `data-v2` 中开发。阶段 6 全量门禁通过前，正式 legacy 入口不分页面切换、不双写、不读取或迁移旧数据；通过后一次切换 Launcher、API 注册和 Vue 消费方，并在同一变更中删除旧入口。

## 后果

临时兼容 handler、通用任意 payload、读时修复和双事实源均不接受。契约变化必须先修改 OpenAPI/ADR并重新生成类型，再修改实现。
