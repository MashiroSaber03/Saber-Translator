# Backend-First 重构实现审计（2026-07-29）

## 1. 审计结论

审计基线：

- 方案：`docs/refactor/backend-first-architecture-plan.md`（4086 行，24 节）。
- 代码基线：`codex/backend-first-v2`，审计开始时 HEAD 为 `da082b97`。
- 审计范围：Launcher/API/Worker、SQLite/Alembic/资产、任务与 operation、翻译、Insight、Studio、书架、任务中心、阅读器、设置、插件、前端媒体加载和测试。

结论：

1. 后端中心化的基础架构已经建立，浏览器关闭后持久任务继续执行、SQLite 事实源、不可变资产、凭据版本、Worker 队列、operation fencing、缩略图和 v2-only 生产入口都不是空壳。
2. 当前实现**尚不能判定为“方案全部实现”**，也不应关闭重构计划。完整 API 契约、HQ/多轮校对、任务重试/批次控制、翻译 Pool 进度/刷新恢复/章节设置记忆、快速工作区、书架批量产品流、任务中心系统控制，以及 Insight/Studio/Reader/设置页面级缺口已在审计后关闭；当前剩余工作已收敛为最终存储、查询计划、故障注入、内存趋势和端到端验收矩阵。
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
| §5 存储架构 | 基本完成 | SQLite WAL、Alembic 0001–0011、不可变 assets、关系表、凭据/插件版本、任务/operation 快照、重试血缘和 Worker 控制命令均存在；完整查询计划、删除矩阵和故障注入验收不足。 |
| §6 快速工作区 | 完成 | 后端播种、bootstrap、reset、promote、423 保护和约束处理完整；前端已有“新建快速翻译”“保存到书架”、new/existing book 弹窗和任务中心引导，promote 保持资产原位并拒绝重复目标。 |
| §7 统一任务系统 | 基本完成 | 持久队列、状态机、SSE、pause/resume/continue/cancel、reorder、关联 replacement retry、批次取消/优先/继续、脱敏详情、事件游标分页和安全点模型释放均已实现；完整跨领域故障矩阵仍待补齐。 |
| §8 翻译任务后端化 | 基本完成 | 创建任务即冻结配置/资产/插件/凭据，Worker 流水线可脱离浏览器执行；HQ 稳定 ID 批处理、多轮校对、真实 render/save 分界、后端 Pool 进度和基于后端失败事实的关联重试均已闭环。 |
| §9 Insight 任务统一 | 基本完成 | 全书/局部分析、派生物、向量、续写和导出均有持久任务；分析和向量重建已改为消费全局任务中心的 SSE/快照投影，不再保留页面级 3 秒轮询。 |
| §10 插件 v3 | 基本完成 | 不可变版本、快照、能力/失败策略、Worker 执行、Agent handoff 和管理 API 已实现；运行时与 OpenAPI 已完成双向闭集。 |
| §11 图片导入与缩略图 | 基本完成 | 普通图片逐页上传、容器后端任务、同步 source thumbnail、长条特判、无 Base64 响应已实现；完整导入失败矩阵和多场景大数据验收不足。 |
| §12 媒体 API 与加载 | 基本完成 | asset URL/ETag/条件请求、翻译/编辑侧栏、指定页码弹窗、Insight 大章节树和续写参考图选择器均已窗口化；Reader 使用虚拟流和懒加载。剩余为跨场景浏览器进程内存趋势验收。 |
| §13 页面职责 | 完成 | 生产前端主要负责交互和后端事实投影；长任务、Pool 进度恢复、章节工作态、失败项重试、Insight 分析和向量重建均由后端持久状态及全局 SSE/快照驱动。 |
| §14 实施阶段 | 基本完成 | 阶段 0–6 的契约、后端能力和产品页面缺口均已关闭；只剩方案定义的最终验收矩阵。 |
| §15 全局验收 | 部分完成 | OpenAPI 已覆盖全部运行时操作，后端 v2 有 100+ 测试，前端有 1700+ 单元/属性/视觉测试；仍缺 EXPLAIN QUERY PLAN、跨页面 1000 页内存趋势、完整删除/GC/secret canary 和多项 crash-window 矩阵。 |
| §16 翻译页 | 基本完成 | 后端任务、编辑 CAS、修复 operation、独立 render/save、文本/导出、HQ 稳定 ID batch、多轮校对、后端关联失败项重试、并行多行 Pool 进度、刷新恢复和指定页码虚拟化已闭环；剩余为完整性能矩阵。 |
| §17 Insight | 基本完成 | 分析/概览/时间线/问答/笔记/续写/设置已切 v2；PagesTree 与续写参考图大列表已虚拟化，分析及向量重建消费任务中心 SSE/快照。剩余为专项进程内存和故障矩阵。 |
| §18 Character Studio | 基本完成 | 文档 CAS、保存型 generate/chat/summary operation、SSE chunk、会话数据、导入导出和诊断主体完整；前端停止操作会调用后端 abort 推进 generation fencing，归档会话支持携带 revision 永久删除。剩余为最终故障矩阵。 |
| §19 书架 | 基本完成 | 后端 CRUD、搜索/标签/排序、批量删除/标签和 jobStatusSummary 已实现；前端已接入排序、书籍批量翻译/删除/标签、章节多选翻译、任务徽章和任务中心定位。剩余为完整批量产品流与并发删除验收。 |
| §20 任务中心 | 基本完成 | 全局抽屉、队列/历史、SSE、暂停/继续/取消、产物下载、单任务排序、状态/类型/书籍筛选、脱敏详情、事件向前分页、整任务/失败项双策略重试、批次取消/优先/继续、释放显存、新建批量分析和跨页定位均已实现；剩余为完整 crash-window/200 批清理验收。 |
| §21 阅读器 | 完成 | 后端页列表、translated→source 回退、VirtualPageStream 和懒加载存在；回退页显示“未翻译”持久徽标，顶栏显示“已翻译 m/N”。 |
| §22 设置/提示词/Provider | 基本完成 | SQLite 事实源、统一 transaction、不可变凭据、统一诊断、提示词、字体和章节级非样式 settings_memory 已接线；设置加载失败时全站写请求及直连 Provider 调用统一受限，文本默认值直接进入同一 store 草稿和 transaction。剩余为最终 secret canary/故障验收。 |
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

### 5.1 已关闭：快速工作区

现已完成：

- `GET /translation/bootstrap` 无参数解析固定快速工作区；新建快速翻译调用后端 reset。
- 快速模式顶栏已有“新建快速翻译”和“保存到书架”两个正式动作。
- promote 弹窗支持 new_book/existing_book，使用显式 mode 契约；资产关系原子转正而不移动对象文件。
- 新书标题、目标书籍章节标题重复会拒绝；成功后新建空快速章节，快速约束清空，已有书籍约束保持。
- reset/promote 遇到非终态 job、active operation 或导入租约时返回 423，前端打开并定位任务中心。
- 后端路由、关系移动、重复目标和前端交互均有专项回归测试。

### 5.2 已关闭主体：书架

已接入：

- `jobStatusSummary` 从 OpenAPI DTO 经 bookshelf API 投影到书籍/章节，徽章优先消费全局任务 store 的实时状态并以数据库聚合兜底。
- 首页批量管理模式支持多选书籍、批量翻译、批量删除和批量加/删标签；排序参数由后端执行。
- 详情支持全选可翻译章节、逐章多选和“翻译选中章节”。
- 后端批量翻译逐章准入：空章节、同类 active job、缺失凭据、非法设置或目标不存在按项跳过，并返回结构化原因；全不可用时返回 422。
- 书籍/章节徽章和批量命令结果可打开任务中心并定位 job/batch/目标。

仍需在最终验收矩阵中覆盖大批量、423 部分拒绝和并发删除产品流。

### 5.3 已关闭主体：任务中心

除既有的排序、筛选、详情、事件分页、重试和批次控制外，本轮新增：

- “释放显存”通过 SQLite `worker_commands` 从 API 投递到隔离 Worker；Worker 只在原子步骤安全点领取，绑定 worker epoch，崩溃遗留 running 命令可恢复。
- API 在本地模型 job step、Worker model operation 或 transient 模型请求运行时返回 409；空闲时卸载已加载的检测/OCR/Lama/Paddle 模型和插件运行时缓存。
- 无 running/pausing/cancelling job、无 running Worker operation/transient request 持续 10 分钟后自动执行同一卸载逻辑。
- 面板“新建批量分析”支持选择书籍、全书/增量/章节范围和章节多选，提交 Insight 后端任务。
- 书架、翻译和其他入口通过 job/batch/book/chapter 目标打开抽屉、切换队列/历史、展开并滚动定位。

### 5.4 已关闭：Insight

- 新增固定行虚拟缩略图网格；无章节和展开章节两种路径都只挂载视口附近节点，1000 页专项测试证明 DOM 有界。
- 指定页码弹窗和续写漫画参考图复用同一虚拟网格。
- `InsightView.vue` 和 `QAPanel.vue` 已移除 3 秒轮询，分析与向量重建状态统一投影全局任务中心队列/历史和 SSE 事件。
- 切换书籍时会清理旧书籍的向量重建状态，避免跨书串态。

### 5.5 已关闭：Character Studio

- “停止”先调用 `/chat/sessions/{id}/abort` 推进后端 generation fencing，再中断本地订阅；服务端任务不会因浏览器断流继续向旧 generation 写入。
- 归档会话列表增加永久删除入口，携带当前 revision 执行 CAS 删除。
- API、store、事件透传和 UI 均有专项测试。

### 5.6 已关闭：阅读器

- translated 模式回退到 source 时显示“未翻译”持久徽标。
- 顶栏显示后端页面摘要推导的“已翻译 m/N”。
- 虚拟流仍只挂载窗口附近图片，回退不引入整章原图预载。

### 5.7 已关闭主体：设置受限态

- 应用启动时同步进入 restricted mode，后端设置成功 hydrate 后才解除。
- restricted mode 下 API client 统一阻断 POST/PUT/PATCH/DELETE 和上传，Studio Agent、Insight QA 等直连 Provider 路径也执行同一门禁；GET 仍可用于恢复与诊断。
- 全局状态条和设置弹窗均显示加载失败并提供重试，设置表单和保存动作不可用。
- `TextStyleDefaultsSettings` 已移除 `saveRequestId/save-complete` 特殊握手，直接修改 settings store 草稿；取消恢复父级快照，保存只执行一次统一 transaction。

## 6. 图片与浏览器内存审计

已确认完成：

- 普通图片逐张 multipart 上传，没有 FileReader/Data URL 导入链。
- source 与 thumbnail 为独立不可变资产。
- 页面列表返回元数据和 URL，不返回图片 Base64。
- 翻译右侧栏使用 `VirtualThumbnailList`。
- 编辑缩略图有窗口化逻辑。
- Reader 使用 `VirtualPageStream` 和 IntersectionObserver。
- 指定页码弹窗、Insight PagesTree 和续写漫画参考图选择器使用固定行虚拟网格。
- 当前翻译主图按 URL 加载，不在 Pinia 保存整章原图。
- 媒体响应支持 ETag/If-None-Match。

仍需完成：

- 单元测试已经证明各大列表的 DOM 节点数量有界；浏览器进程内存趋势验收仍只覆盖 Reader，需要分别覆盖翻译页、指定页码弹窗、编辑缩略图、Insight 页面树。
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
- 前端 Vitest：240 个测试文件、1725 项测试全部通过；typecheck、ESLint、Stylelint 和 UI architecture audit 同步通过。
- 后端 v2 Pytest：136 项测试全部通过；2 条 warning 为 SQLAlchemy 对表达式索引反射的既有提示。
- 生成类型确定性检查与运行时 Flask 路由双向闭集测试通过；OpenAPI/运行时均为 186 个操作。
- `npm run build:check` 通过，生产静态包已更新。

## 9. 建议的后续关闭顺序

在再次宣布“重构完成”前，建议按以下门禁顺序处理：

1. **存储门禁**：补齐核心查询的 `EXPLAIN QUERY PLAN`、删除/GC 矩阵和 secret canary。
2. **可靠性门禁**：补齐 API/Worker 重启、断线恢复、operation/job fencing、quick reset/promote 和并发删除的故障注入矩阵。
3. **性能与发布门禁**：补齐翻译/编辑/指定页码/Insight 的 1000 页浏览器进程内存趋势、Worker RSS 趋势、批量产品流与最终打包验收。

当前页面级功能缺口已关闭，剩余工作是方案正式定义的最终验收门禁；通过前仍不能宣布整个重构完成。
