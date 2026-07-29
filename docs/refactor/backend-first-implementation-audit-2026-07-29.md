# Backend-First 重构实现审计（2026-07-29）

## 1. 审计结论

审计基线：

- 方案：`docs/refactor/backend-first-architecture-plan.md`（4086 行，24 节）。
- 代码基线：`codex/backend-first-v2`，审计开始时 HEAD 为 `da082b97`。
- 审计范围：Launcher/API/Worker、SQLite/Alembic/资产、任务与 operation、翻译、Insight、Studio、书架、任务中心、阅读器、设置、插件、前端媒体加载和测试。

结论：

1. 后端中心化的基础架构已经建立，浏览器关闭后持久任务继续执行、SQLite 事实源、不可变资产、凭据版本、Worker 队列、operation fencing、缩略图和 v2-only 生产入口都不是空壳。
2. 当前实现**不能判定为“方案全部实现”**，也不应关闭重构计划。完整 API 契约、HQ/多轮校对、任务重试/批次控制、任务中心核心控制，以及翻译 Pool 进度/刷新恢复/章节设置记忆已在审计后关闭；剩余主要缺口集中在书架批量功能、快速工作区前端入口、Insight 虚拟化和轮询清理、Studio 中止/归档删除接线、阅读器状态提示、设置受限态与验收矩阵。
3. 本轮发现的 API Key 保存/使用故障不是后端没有保存，而是“后端凭据摘要”和“前端空白密钥输入框”之间的语义断裂。本轮已修复该故障及同类入口，并增加后端任务准入校验。
4. 本轮还修复了翻译页“最后访问页只读不写”的数据闭环缺陷，并将导航更新改为与方案一致的独立 last-write-wins，不再因旧标签页 revision 产生普通切页冲突。
5. 用户已确认的两项偏差不作为缺陷：
   - 可信局域网完全开放可以接受，不要求鉴权。
   - 不增加单图像素预算、解码安全切片等设计；保留逐页解码和及时释放。

## 2. 本轮已修复

### 2.1 API Key 保存后仍被判定为空

根因链路：

1. `PUT /api/v2/settings/transactions` 已把密钥写入不可变 `credential_versions`。
2. 后端按安全设计只返回 `hasKey/currentVersion`，不回传明文。
3. `settingsStore` 重新 hydrate 后会清空浏览器中的 `apiKey` 字符串。
4. 翻译页校验只检查该空字符串，没有查询后端 `hasCredential(domain, provider)`，因而误报“请填写 API Key”。
5. 密钥框没有“已保存、留空保持不变”的状态，用户看到的表现像保存失败。

已完成：

- 翻译、HQ、校对、AI Vision OCR、百度 OCR 校验均识别后端已存凭据。
- 百度 OCR 更换凭据时必须同时提交 API Key 与 Secret Key，禁止半更新导致旧凭据字段丢失。
- 翻译、HQ、校对、OCR、Plugin Agent、网页导入、Insight 五类 Provider 设置均显示后端已存凭据提示和安全占位符。
- 模型发现和连接测试继续支持“未保存明文”或“后端已存 domain 凭据”两种模式。
- 翻译任务在数据库正式准入前，由后端校验 Provider 能力、模型、Base URL 和 credential version；缺失时返回明确设置指引，不创建注定失败的 job。
- 增加 HTTP 级保存/重载/不回显测试、前端校验测试和组件状态测试。

### 2.2 最后访问页未写回

原实现：

- bootstrap 返回 `lastVisitedPageId/revision`。
- 翻译页初始化会恢复该页。
- 切页从未调用后端 PATCH，所以实际值一直不更新。

已完成：

- 新增前端 `updateLastVisitedPage` API 调用。
- 初始 hydrate 不制造无意义写入；用户切页后串行写回。
- 后端导航 revision 仅用于观察写入，更新采用 last-write-wins；旧标签页不会得到 409。
- 增加连续切页 revision 串行测试和后端 stale revision 覆盖测试。

### 2.3 翻译 Pool 进度、刷新恢复与章节工作态

已完成：

- `jobs.latest_progress` 现在由数据库中的 item/step 图重建，持久化总页数、终态计数、执行模式、顺序模式当前页/步骤，以及每个实际参与阶段的完成/失败/跳过/等待/处理中页和深度学习锁等待状态。
- Worker 只为当前 job 实际存在的 step kind 启动 Pool；共享深度学习信号量的等待状态由 Worker 覆盖写入，计数与当前页始终来自 SQLite。
- claim、阶段开始/完成、页面成败、暂停/取消 drain、失败和终态都会保持同一份合法进度结构；不再在 job 失败时用单独 error 对象破坏 progress schema。
- standard/HQ/proofread 将原来揉在一起的 render 正式拆为 `render` 资产生成检查点和 `save` 原子发布步骤，因此并行界面中的保存 Pool 是真实执行阶段。
- bootstrap 的活动任务新增冻结 `pageIds`，翻译页刷新后恢复 `activeJobId`、页范围、队列/过渡/暂停/中断状态、总进度和 Pool 进度；SSE 事件与任务中心 REST 刷新都会重新投影同一后端快照。
- 并行模式恢复专属多行 Pool 进度条；顺序模式继续显示单总进度条、当前页和当前原子步骤。
- 章节 `settings_memory` 已接入 bootstrap hydrate 和 400ms debounce CAS PATCH；只允许翻译相关非样式工作态，前后端同时排除 API key/secret、字体、字号、方向、颜色、描边、行距、对齐和修复方式。

## 3. 逐章节实现状态

状态定义：

- **完成**：核心设计和主要验收链路均存在。
- **基本完成**：主体存在，仍有非阻断性契约或验收缺口。
- **部分完成**：后端或前端只实现了一半，无法满足完整产品口径。
- **未完成**：方案明确要求的主功能/API 不存在。
- **接受偏差**：与方案文字不一致，但已由用户明确接受。

| 方案章节 | 状态 | 审计结论 |
| --- | --- | --- |
| §1–§3 决策与边界 | 基本完成 | v2-only、后端事实源、浏览器交互/显示边界成立；部分前端仍用轮询或保留未接线工作态。 |
| §4 运行架构 | 基本完成 | Launcher/API/Worker 分离、epoch/heartbeat/fencing、单实例和打包入口存在；CORS 与方案同源限制文字不一致，按用户意见记为接受偏差。 |
| §5 存储架构 | 基本完成 | SQLite WAL、Alembic 0001–0010、不可变 assets、关系表、凭据/插件版本、任务/operation 快照和重试血缘均存在；完整查询计划、删除矩阵和故障注入验收不足。 |
| §6 快速工作区 | 部分完成 | 后端播种、bootstrap、reset、promote、423 保护和约束处理存在；前端缺“新建快速翻译”和“保存到书架”顶栏入口，promote 无前端 API/弹窗；测试只覆盖部分 promote 语义。最后访问页本轮已补齐。 |
| §7 统一任务系统 | 基本完成 | 持久队列、状态机、SSE、pause/resume/continue/cancel、reorder、关联 replacement retry、批次取消/优先/继续、脱敏详情和事件游标分页均已实现；系统模型释放和完整跨领域重试验收仍待补齐。 |
| §8 翻译任务后端化 | 基本完成 | 创建任务即冻结配置/资产/插件/凭据，Worker 流水线可脱离浏览器执行；HQ 稳定 ID 批处理、多轮校对、真实 render/save 分界、后端 Pool 进度和基于后端失败事实的关联重试均已闭环。 |
| §9 Insight 任务统一 | 部分完成 | 全书/局部分析、派生物、向量、续写和导出均有持久任务；页面仍保留 3 秒轮询，未完全统一到全局 SSE/快照。 |
| §10 插件 v3 | 基本完成 | 不可变版本、快照、能力/失败策略、Worker 执行、Agent handoff 和管理 API 已实现；运行时与 OpenAPI 已完成双向闭集。 |
| §11 图片导入与缩略图 | 基本完成 | 普通图片逐页上传、容器后端任务、同步 source thumbnail、长条特判、无 Base64 响应已实现；完整导入失败矩阵和多场景大数据验收不足。 |
| §12 媒体 API 与加载 | 部分完成 | asset URL/ETag/条件请求、翻译侧栏和编辑侧栏虚拟化、Reader 虚拟流存在；指定页码弹窗、Insight 大章节树和部分续写选择器仍一次创建大量缩略图节点。 |
| §13 页面职责 | 基本完成 | 生产前端主要为投影和交互，业务长任务、Pool 进度恢复、章节工作态和失败项重试均由后端事实驱动；Insight 等处仍有前端轮询。 |
| §14 实施阶段 | 部分完成 | 阶段 1–6 的大量基础代码已经提交，阶段 0 契约门禁已关闭；产品页和验收矩阵仍未全部满足。 |
| §15 全局验收 | 部分完成 | OpenAPI 已覆盖全部 180 个运行时操作，后端 v2 有 100+ 测试，前端有大量单元/属性/视觉测试；仍缺 EXPLAIN QUERY PLAN、Insight/翻译 1000 页 DOM/内存、批量产品流和多项 crash-window 矩阵。 |
| §16 翻译页 | 基本完成 | 后端任务、编辑 CAS、修复 operation、独立 render/save、文本/导出、HQ 稳定 ID batch、多轮校对、后端关联失败项重试、并行多行 Pool 进度和刷新恢复已闭环；剩余主要是指定页码大列表虚拟化和完整性能矩阵。 |
| §17 Insight | 部分完成 | 主要分析/概览/时间线/问答/笔记/续写/设置功能已切 v2；PagesTree 展开章节会创建整章节点，状态与向量重建仍轮询，专项 1000 页验收缺失。 |
| §18 Character Studio | 基本完成 | 文档 CAS、保存型 generate/chat/summary operation、SSE chunk、会话数据、导入导出和诊断主体完整；前端没有调用 abort API，也没有归档会话永久删除入口。 |
| §19 书架 | 部分完成 | 后端 CRUD、搜索/标签/排序、批量删除/标签和 jobStatusSummary 已实现；前端丢弃 jobStatusSummary，store 的 batchMode/selectedBookIds/expandBook 是未消费状态，章节多选翻译、书籍批量翻译/删除/标签和任务跳转未实现。 |
| §20 任务中心 | 基本完成 | 全局抽屉、队列/历史、SSE、暂停/继续/取消、产物下载、单任务排序、状态/类型/书籍筛选、脱敏详情、事件向前分页、整任务/失败项双策略重试和批次取消/优先/继续已实现；释放显存、新建批量分析和跨页定位仍缺。 |
| §21 阅读器 | 基本完成 | 后端页列表、translated→source 回退、VirtualPageStream 和懒加载存在；缺“未翻译”持久徽标和“已翻译 m/N”统计。 |
| §22 设置/提示词/Provider | 基本完成 | SQLite 事实源、统一 transaction、不可变凭据、统一诊断、提示词、字体和章节级非样式 settings_memory 已接线；本轮修复凭据 UI/校验；受限态仍未统一禁止所有任务/Provider 操作，文本默认值仍保留父子特殊握手。 |
| §23 插件管理 | 基本完成 | v3 manifest/version/snapshot/runtime/Agent/管理 UI、契约和测试主体存在。 |
| §24 当前不做事项 | 完成/接受偏差 | 未引入账号、多租户、云对象存储、瓦片金字塔或单图像素预算；符合用户最终口径。 |

## 4. 阻断“方案完成”结论的问题

### 4.1 已关闭：OpenAPI 完整唯一契约

运行时 Flask 路由与 `openapi/v2.yaml` 逐 method/path 标准化比对结果：

- 运行时 v2 操作：180。
- OpenAPI 已描述操作：180。
- 运行时存在但 OpenAPI 缺失：0。
- OpenAPI 描述但运行时不存在：0。

已补齐所有运行时操作、命名请求/响应 schema、operationId 和幂等头约束；移除 `GenericObject`/`GenericSuccess` 占位响应。`test_openapi_contract.py` 现在执行 Flask `url_map` ↔ OpenAPI 双向闭集、`$ref`、operationId、响应、幂等性和前端生成类型门禁。前端 v2 HTTP DTO 已全部改为从生成的 `components['schemas']` 派生。

### 4.2 已关闭：翻译并行进度与刷新恢复

后端现已持久化实际 Pool 图和锁等待状态；bootstrap 返回活动任务页范围，前端从 bootstrap、SSE event payload 和任务中心 REST 快照恢复同一投影。并行模式显示实际启用阶段的多行进度，顺序模式保持单总进度和当前页/步骤；未参与当前工作流的 Pool 不会被创建或伪装为已执行。

### 4.3 已关闭：失败项重试后端事实源

已完成：

- 新增 `/jobs/{id}/retry` 与 `/jobs/{id}/retry-failed`，仅接受对应的 `failed` / `completed_with_errors` 来源状态，并始终创建带 `retry_of_job_id/retry_mode` 的 replacement job。
- 翻译、检测和去字默认重新读取当前后端设置并冻结新配置、凭据与插件快照；可显式沿用原任务配置/凭据/插件/字体快照。
- `style_apply` 强制沿用原 selected_fields、冻结样式与 font snapshot，只缩小到失败 page_id。
- 原快照重试重新校验 create-phase asset 完整性；item-start/checkpoint 输入由新任务重新绑定。
- 翻译页不再读取 `imageStore.getFailedImageIndices()` 来创建普通任务，而是定位当前章节最近一次同类型部分失败任务并调用关联重试。
- 任务详情提供失败项、步骤、错误、计数、耗时、脱敏配置、步骤资源和最近事件；事件端点支持 `before/after` 游标。
- 历史清理按重试血缘从叶子到根删除，并保护仍被非终态 replacement job 引用的来源任务。

### 4.4 已关闭：HQ 和多轮校对真实后端流水线

方案 §16.3.6–§16.3.7 要求 HQ 使用稳定 page/bubble ID 的批量协议，并要求校对的每一轮独立冻结 provider、model、URL、batch size、RPM、重试、JSON、流式、extra body 和 prompt。

已完成：

- HQ 按冻结的 1–10 batch size 在 Worker 中聚合当前批次页面，模型协议使用稳定 `pageId/bubbleId`。
- 缺页、未知/重复 ID、气泡集合不一致、非法 JSON 和非字符串/空译文均作为批次失败处理，不使用原始模型文本降级成成功结果。
- 每批保存输入 fingerprint、解析结果、失败原因，并把模型原始输出保存为不可变 JSON asset；崩溃窗口重复执行可替换旧 step-output 绑定。
- 校对任务冻结全部轮次的 provider、credential version、model、URL、batch size、OpenAI request/execution options、prompt 和 provider revision；每页生成逐轮独立步骤。
- 每轮按自己的 batch size 执行；无可校对译文页整体 `skipped`，某批失败页不进入后续轮次，全部校对轮次完成后才开放 render。
- 专项测试覆盖 3 页 HQ 的 2+1 分批、两轮不同模型/批大小、非法稳定 ID、无译文页、部分失败继续，以及暂停/继续不重复已完成 batch。

## 5. 主要产品缺口

### 5.1 快速工作区

后端已有：

- `GET /translation/bootstrap` 无参数工作区解析。
- `POST /quick-workspace/reset`。
- `POST /quick-workspace/promote`。
- 非终态 job/operation/import lease 保护。

前端缺失：

- 快速模式顶栏“新建快速翻译”按钮。
- “保存到书架”按钮、new_book/existing_book 弹窗。
- promote API wrapper。
- 423 后打开/定位任务中心的引导。

现有“清空全部”在快速模式调用 reset，不能替代方案定义的两个顶栏产品动作。

### 5.2 书架

后端 `list_books` 已查询 `jobStatusSummary`，但 `api/bookshelf.ts::toBook` 没有投影该字段，UI 因此无法显示任务徽标。

`bookshelfStore` 有 `batchMode`、`selectedBookIds`、batch tag helpers 和 `expandBook`，但书架视图/组件没有消费者。缺少：

- 首页批量管理模式。
- 批量翻译、删除、加/删标签。
- 详情章节多选与“翻译选中章节”。
- 书籍/章节任务状态徽标和任务中心跳转。
- 后端 translation batch 的前端调用。

### 5.3 任务中心

本轮已完成 `jobs/reorder` 的 store/UI 接线、状态/类型/书籍筛选、任务详情和事件向前分页、单任务/失败项重试，以及批次取消/优先/继续。仍缺：

- 释放本地模型/显存。
- 新建批量分析入口。
- 从书架/翻译/Insight 定位具体 job 或 batch。

### 5.4 Insight

- `PagesTree.vue` 展开章节时直接调用 `createPageThumbnailItems(startPage, endPage)`，1000 页章节会创建整章缩略图 DOM。
- 无章节时只有每次 100 页的“加载更多”，有章节时没有相同保护。
- `InsightView.vue` 的任务状态和 `QAPanel.vue` 的向量重建仍使用 3 秒轮询。
- 现有 100/500/1000 页 Playwright 内存测试只覆盖 Reader，没有覆盖 Insight 页面树。

### 5.5 Character Studio

后端已有且测试过：

- `/chat/sessions/{id}/abort` 的 generation fencing。
- 归档会话永久删除。

前端缺：

- 主动中止按钮/API 调用；当前 `AbortController` 仅断开浏览器订阅，不执行服务端 abort 语义。
- 归档会话永久删除入口/API wrapper。

### 5.6 阅读器

虚拟流和源图回退已完成，但方案要求的以下信息缺失：

- translated 模式回退到 source 时的“未翻译”持久徽标。
- 顶栏“已翻译 m/N”。

### 5.7 设置受限态

- `settingsStore.isBackendReady` 当前主要用于禁用“保存设置”。
- 翻译、Insight、Studio、网页导入等任务/Provider 操作没有统一受限态门禁。
- 后端设置加载失败时仍可能使用前端出厂默认参与 UI 校验或发起命令；后端会再次解析真实设置，但不符合方案“加载成功前禁止创建任务/调用 Provider”的明确口径。
- `TextStyleDefaultsSettings` 最终确实与其他设置进入同一个 backend transaction，但仍通过 `saveRequestId/save-complete` 特殊父子握手，未完成 §22.8 要求的结构清理。

## 6. 图片与浏览器内存审计

已确认完成：

- 普通图片逐张 multipart 上传，没有 FileReader/Data URL 导入链。
- source 与 thumbnail 为独立不可变资产。
- 页面列表返回元数据和 URL，不返回图片 Base64。
- 翻译右侧栏使用 `VirtualThumbnailList`。
- 编辑缩略图有窗口化逻辑。
- Reader 使用 `VirtualPageStream` 和 IntersectionObserver。
- 当前翻译主图按 URL 加载，不在 Pinia 保存整章原图。
- 媒体响应支持 ETag/If-None-Match。

仍需完成：

- `PageSelectionModal.vue` 使用普通 `ProductThumbnailGrid` 渲染全部候选。
- `PagesTree.vue` 有章节时渲染整章节点。
- 部分 continuation/reference selector 仍构造完整缩略图列表。
- 内存趋势验收只覆盖 Reader；需要分别覆盖翻译页、指定页码弹窗、编辑缩略图、Insight 页面树。
- Worker 有界队列/并发实现已有测试，但“章节总页数增长时 RSS 不线性增长”的端到端 Worker 证明不足。

## 7. 存储与数据加载审计

### 7.1 已实现结构

`schema.py` 已包含以下主要事实域：

- 内容：books、chapters、pages、bubbles、tags、constraints、page_assets。
- 资产：assets、object_commit_journal、fonts。
- 设置：app_settings、book_settings、provider_settings、prompts。
- 凭据/插件：credentials + immutable versions/current pointers，plugins + immutable versions/current pointers。
- 任务：job_batches、jobs、items、steps、events、各类 snapshot/input/output/artifact。
- 运行控制：epochs、leases、write intents/locks、import leases、rate limits、idempotency records。
- 即时保存型工作：operations、events、snapshots、inputs/artifacts、render_requests、transient_requests。
- Studio：documents、sessions、messages、message_assets。
- Insight/续写：runs/targets/results/heads/layers/artifacts、timeline、vectors、notes/citations、continuation projects/scripts/pages/characters/forms/image versions。

### 7.2 已实现数据加载

- 翻译 bootstrap 一次返回章节元数据、设置、字体、提示词、活动任务和页面摘要。
- 当前页 document 按需读取；主图和缩略图为独立 URL。
- 书架列表由后端完成查询/排序/标签聚合。
- Reader 拉取页摘要后使用虚拟视口加载图片。
- Studio 先加载 index，再按需加载文档和会话。
- Insight 主面板使用 `v-if` 按标签挂载，不再把所有隐藏面板同时常驻。

### 7.3 仍缺的存储验收

- 没有 `EXPLAIN QUERY PLAN` 测试证明 1000 页章节、200 任务批次的核心查询稳定命中预期索引。
- 资产 GC、删除和 crash-window 有基础测试，但未覆盖方案列出的全量 book/chapter/page/run/export/plugin/font 删除矩阵。
- quick reset/promote 的 job/operation/import lease、两种 promote 和 constraints 组合测试不完整。
- API Key 不回显已有定向测试，但尚无覆盖所有响应/SSE/log/snapshot 的全局 secret canary 测试。

## 8. 测试与发布状态

当前测试资产的优点：

- 后端 v2 测试覆盖状态机、epoch、WAL 备份、资产发布/GC、任务 fencing、并行 Pool、operation、翻译、Insight、续写、Studio、插件和打包入口。
- 前端有大量单元、属性和视觉回归测试。
- Reader 有 100/500/1000 页 DOM/heap/process memory 趋势测试。

不能据此宣称完整的原因：

- 多项方案验收没有对应测试，也没有实现。
- 原生产静态目录与源码曾存在构建漂移；本轮已执行 `npm run build:check` 重建 `src/backend_v2/static/vue`，后续每次前端改动仍须保持该门禁。
- 方案要求的性能、并发、失败注入和跨页面产品流测试尚不完整。

本轮实际验证结果：

- `npm run typecheck`、`npm run lint`、`npm run lint:css`、`npm run lint:ui` 全部通过。
- 前端 Vitest：237 个测试文件、1710 项测试全部通过；typecheck、ESLint、Stylelint 和 UI architecture audit 同步通过。
- 后端 v2 Pytest：130 项测试全部通过；2 条 warning 为 SQLAlchemy 对表达式索引反射的既有提示。
- `npm run check:api` 与运行时 Flask 路由双向闭集测试通过；OpenAPI/运行时均为 185 个操作。
- `npm run build:check` 通过，生产静态包已更新。

## 9. 建议的后续关闭顺序

在再次宣布“重构完成”前，建议按以下门禁顺序处理：

1. **产品页门禁**：快速工作区两个顶栏动作、书架批量/状态，以及任务中心系统模型释放/批量分析/跨页定位。
2. **加载门禁**：Insight/指定页码/续写缩略图虚拟化，并补齐各场景 1000 页内存测试。
3. **Studio/Reader/设置门禁**：Studio 服务端 abort 与归档删除接线、Reader 徽标/统计、全站 settings restricted mode。
4. **最终验收门禁**：执行方案 §15 和各页面章节的剩余故障注入、并发、删除、GC、查询计划、断线恢复和打包验收。

当前可以继续在该分支重构，但不能以“只剩零碎优化”描述剩余工作；多个页面级功能和最终验收仍属于正式方案范围内的未完成项。
