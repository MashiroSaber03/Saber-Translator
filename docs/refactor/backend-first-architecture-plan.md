# Saber Translator Backend-First 当前架构基线

> 文档状态：当前实现基线，不是历史迁移方案或未来路线图。
> 最后更新：2026-08-20

## 1. 目标

Saber Translator 采用后端中心化架构：后端保存事实数据、执行任务并发布结果，前端负责输入、展示和交互。

实现时遵守以下原则：

1. 一个事实只保存在一个权威位置，不做前后端双写，也不维护平行镜像结构。
2. 长任务必须可查看、暂停、继续和取消，但控制逻辑只落在必要的状态与检查点上。
3. 只为已证实的故障场景增加机制，不为假设中的分布式部署设计协议。
4. 当前数据结构就是唯一受支持的数据结构；项目不读取、迁移或转换旧数据。
5. 本地模型和外部 Provider 的失败必须区分。外部服务不稳定不等于项目代码错误。

## 2. 运行拓扑

正式运行由三个角色组成：

- Launcher：创建本次运行身份，启动和监督 API、Worker。
- API：提供 REST/SSE、媒体访问、设置、内容管理以及不需要本地模型的短操作。
- Worker：执行全局任务队列、本地模型推理、插件和需要后台运行的操作。

桌面 GUI 是 Launcher 的用户界面，不另建一套业务后端。网页前端和桌面 GUI 都通过同一套 v2 API 读取状态。

API 与 Worker 各自在 `process_epochs` 中保存一个进程身份和心跳。进程失效后由恢复逻辑处理中断的工作。项目不再维护额外的 Worker lease、API executor lease 或每次尝试 heartbeat 表。

## 3. 存储

### 3.1 SQLite 与文件资产

- SQLite 保存书籍、章节、页面、气泡、设置、任务、操作及引用关系。
- 文件系统保存原图、缩略图、译图、导出包和其他大型二进制资产。
- 所有正式资产通过 `asset_id` 和数据库关系引用，不通过文件名猜测业务身份。
- SQLite 开启外键并使用短事务。`BUSY/LOCKED` 只做有限重试；一次锁竞争不得触发 API 重启。

### 3.2 唯一当前结构

数据库直接由 SQLAlchemy metadata 创建，并在 `schema_metadata` 中记录唯一当前 revision。启动时会检查：

- revision 必须与代码一致；
- 表集合必须与 metadata 完全一致；
- `integrity_check` 与 `foreign_key_check` 必须通过。

不使用 Alembic，不提供迁移脚本、兼容读取、旧字段探测或自动备份。现有数据库不是当前 revision 时，启动会拒绝使用，并明确要求清空 `data-v2` 后重新创建。

协议或导入文件中的 `schemaVersion` 只用于校验当前消息格式，不代表支持旧版本升级。

## 4. 内容模型

层级关系为 `Book → Chapter → Page → Bubble`：

- `pages` 保存页面顺序、源图身份、文档 revision 和当前渲染状态。
- `bubbles` 每行保存一个气泡；位置、文本和样式以该气泡的列与 payload 为唯一事实。
- 页面批量修改使用 document revision 做 CAS，避免旧编辑覆盖新编辑。
- 单气泡编辑直接更新该气泡，不维护整页气泡 JSON 镜像。
- 原图、缩略图、去字图和译图通过 `page_assets` 的明确 role 关联。

普通图片导入按图片逐张提交：后端完成原图落盘、缩略图生成和数据库提交后才返回成功。失败重试复用同一个幂等键，避免响应丢失造成重复落库。容器和网页导入使用后台任务。

## 5. 后台工作模型

### 5.1 Jobs

Jobs 用于翻译、检测、去字、批量导入导出、Insight、续写和插件生成等长任务。

- 全局一次只运行一个 job，queued job 可排序。
- job 由 `job_items` 和 `job_steps` 构成，数据库状态是恢复依据。
- 顺序模式按页面执行；并行模式使用有界流水线。
- 并行流水线的上游最多领先保存阶段 50 页，不能让检测或 OCR 提前堆积上千页结果。
- `MemoryError`、CUDA OOM 和原生分配失败必须让当前步骤失败，不能转换成空结果后标记成功。

主要状态：

`queued → running → completed / completed_with_errors / failed`

控制状态：

- 暂停：`running → pausing → paused`
- 继续：`paused / interrupted → queued`
- 取消：`queued → cancelled`，或 `running / pausing → cancelling → cancelled`

Worker 收到暂停或取消后停止投递新步骤，已开始的原子步骤到达安全点后写入检查点。当数据库中没有 running step 时，统一 `finalize_control` 完成状态转换。不为每个 Pool slot 保存 drain ACK。

### 5.2 执行权

一次领取只使用三元组作为写回围栏：

`job_id + attempt_id + worker_epoch_id`

所有步骤、检查点、进度和终态写入都必须匹配该三元组。Worker 进程失效后，旧 epoch 的迟到结果自然失去写入资格。项目不再叠加 attempt lease token 或单独续租线程。

### 5.3 章节写锁

会修改章节内容的 job 在开始前取得 `chapter_write_locks`：

1. 当前事务确认目标章节没有其他 job 的锁；
2. 确认此前已开始的短 operation/render 链已经结束；
3. 在同一写事务中创建锁并领取 job。

任务暂停或中断时保留锁，取消或终止时释放。编辑和删除遇到锁返回 423，前端打开任务中心让用户处理，不自动取消任务、轮询猜测结果或重试删除。

项目不维护 queued 阶段的 write-intent 表。等待旧短操作结束时只记录 job 的 blocked reason。

### 5.4 Operations 与瞬时请求

- `operations`：需要保存结果的单页检测、单泡 OCR/翻译/颜色、修复、Studio 生成等短操作。
- `render_requests`：页面渲染请求。
- `transient_requests`：只在当前请求生命周期内有意义的临时协作请求。

Operation 用自身的 operation identity 与 executor epoch 防止迟到写回，不再维护第二套 attempt lease。

### 5.5 模型释放

API 请求释放 Worker 模型时，直接在当前 Worker 的 `process_epochs` 行记录请求 ID、处理 ID、结果或错误。Worker 在安全点处理。无需通用 `worker_commands` 队列表。

## 6. 幂等与并发

`Idempotency-Key` 只用于重放可能创建新事实或发布不可逆结果的请求，例如：

- 图片上传及其网络重试；
- 创建/重试长任务；
- 创建持久 operation；
- 页面文档与气泡批量修改；
- 设置事务、导入提交、插件发布和其他持久工作流提交。

普通 CRUD 和状态切换不经过持久幂等记录，例如书籍/章节/标签增删改、任务暂停/继续/取消、队列排序和清理历史。它们依赖数据库事务、CAS 或目标当前状态保证一致性。

所有可能冲突的写入应在 Repository 内使用短 `BEGIN IMMEDIATE` 或 revision 条件完成。前端不能通过二次 GET 猜测一次失败写入是否已经成功。

## 7. 设置与 Provider

- 全局设置、Provider 设置和凭据在一个后端事务中保存。
- 凭据正文不回传前端，前端只接收是否已保存及版本摘要。
- 设置弹窗加载成功后才能编辑；字段修改会短暂防抖并自动保存，不提供“保存设置”或“取消并恢复快照”的第二套语义。
- Provider 是否需要 API Key 由 Provider 能力和实际 base URL 共同判断。本地 URL 不应因为选择了一个命名 Provider 而强制要求 Key。
- 模型列表获取失败不妨碍用户手工填写模型名。

## 8. 前端边界

- OpenAPI v2 是前后端 HTTP 契约，TypeScript 类型由它生成。
- API 模块负责线协议转换；Pinia store 保存页面所需投影；组件只处理交互与显示。
- 任务中心只读取后端任务事实，不在浏览器中实现另一套任务状态机。
- 删除遇到 423 时只提示并打开任务中心，不自动取消、等待或重试。
- 后端设置加载失败只禁用设置表单；不设置全局“受限模式”来阻断无关 API。
- UI 视觉可以继承旧版观感，但组件结构、设计 token 和响应式行为以重构后的前端规范为准。

## 9. 错误处理

- 参数、契约和持久数据错误应返回稳定错误码及可理解消息。
- 网络中断、Provider 5xx、限流或返回格式异常应保留为外部服务错误，不归因于本地模型或任务调度。
- 进程级内存分配失败不能被吞掉；任务必须明确失败，并尽快释放当前步骤持有的对象。
- 日志不得输出 API Key、凭据正文或完整敏感请求。
- 失败处理只面向当前结构，不增加旧数据修复、字段回填或运行时迁移分支。

## 10. 代码组织

- `src/backend_v2/api`：应用装配与系统路由。
- `src/backend_v2/content`：书籍、章节、页面和气泡。
- `src/backend_v2/jobs`：长任务状态、图和 Worker 执行。
- `src/backend_v2/operations`：保存型短操作。
- `src/backend_v2/storage`：当前 schema、数据库连接、进程 epoch 与种子数据。
- `src/backend_v2/settings`：设置、Provider 与凭据。
- `src/backend_v2/translation`、`insight`、`studio`、`web_import`、`plugins`：各业务领域。
- `vue-frontend/src/api`：HTTP 契约适配。
- `vue-frontend/src/stores`：前端状态投影。
- `vue-frontend/src/components`、`views`：UI。

业务写入必须通过所属 Repository/Service；不得从组件、路由或其他领域直接拼 SQL 修改表。

## 11. 变更检查

架构修改至少验证：

1. OpenAPI 路由集合与 Flask 运行时一致，生成类型已同步。
2. SQLite 当前 schema、外键和表集合测试通过。
3. job pause/cancel/recovery、operation fencing 和章节锁测试通过。
4. 前端类型、ESLint、UI 架构检查和单元测试通过。
5. Python compile、后端全量测试和打包依赖检查通过。
6. 搜索确认没有已删除表、旧兼容层或迁移工具的运行时代码残留。

当新需求可以直接放入现有内容、job、operation 或 transient 模型时，不新增平行框架。
