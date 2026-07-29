# Backend-First 重构最终实现审计（2026-07-29）

## 1. 最终结论

审计基线：

- 唯一方案：`docs/refactor/backend-first-architecture-plan.md`（4086 行、24 节）。
- 实现分支：`codex/backend-first-v2`。
- 审计范围：Launcher/API/Worker、SQLite/Alembic/资产、jobs/operations、翻译、Insight、Character Studio、书架、任务中心、阅读器、设置、插件、媒体加载、生产静态包和自动化验收。

结论：

1. 方案中要求的后端中心化架构、页面功能、存储模型、加载策略和最终验收门禁均已实现；本审计未再发现方案内的未实现项，可以关闭本轮重构计划。
2. 浏览器已退化为交互和显示端。已提交的翻译、分析、导入导出、续写、Character Studio 保存型操作等由后端持久状态驱动，浏览器关闭或崩溃不会使任务事实和已完成产物丢失。
3. SQLite 是结构化事实源，不可变 assets 是文件事实源；任务、步骤、事件、快照、凭据版本、插件版本、operation attempt、进程 epoch 和写锁均可恢复、可 fencing。
4. 图片列表统一消费缩略图并窗口化；主图只按当前页或阅读器视口加载。100/500/1000 页浏览器和 Worker 趋势验收均已通过。
5. 用户确认的两项偏差作为最终需求，不是缺陷：
   - 可信局域网完全开放，不增加 Origin/鉴权限制。
   - 不增加单图像素预算、解码安全切片等机制；采用逐页解码、及时释放和有界队列。

## 2. API Key 故障结论与修复

### 2.1 根因

API Key 实际已经写入后端不可变 `credential_versions`。安全响应只返回 `hasKey/currentVersion`，不会回传明文；前端重新加载设置后把密钥输入框置空，但翻译页旧校验只看这个空字符串，于是误报“请先填写 API Key”，用户也误以为保存失败。

### 2.2 已完成

- 翻译、HQ、逐轮校对、AI Vision OCR、百度 OCR、Plugin Agent、网页导入和 Insight 都识别后端 `hasCredential(domain, provider)`。
- 设置框明确显示“已保存在后端；留空保持不变”，不会伪装回显明文。
- 更换百度凭据时要求 API Key 与 Secret Key 成对提交，避免半更新覆盖。
- 模型获取、连接测试和正式任务准入均可使用后端已存凭据。
- 后端在创建任务前校验 Provider 能力、模型、Base URL 和 credential version；无效配置不会创建注定失败的 job。
- 所有响应、SSE、错误、任务/operation 快照和插件输出增加递归脱敏；secret canary 回归证明明文不会进入可观察面。

## 3. 最终运行架构

```text
Launcher
  ├─ 单实例锁、data root、迁移/恢复、API/Worker 进程监护
  ├─ API
  │   ├─ REST / SSE / 媒体流 / 后端设置事务
  │   ├─ 有界 CPU render 与远程保存型 operation
  │   └─ 不导入 Torch、Chroma、插件业务代码或旧模型接口
  └─ Worker
      ├─ 全局持久 job 队列、顺序/并行 Pool
      ├─ 本地模型、插件 v3、Insight/Chroma、导入导出
      ├─ operation 安全点优先处理与模型生命周期
      └─ 周期性 journal/integrity/asset/object/Chroma 对账清理
```

API 与 Worker 的业务协作只通过 SQLite 和不可变文件资产。epoch、heartbeat、lease、attempt 和 CAS 行数共同决定写权限；续租或发布更新影响 0 行时立即失权，迟到结果只能丢弃。

## 4. 方案逐章关闭矩阵

| 方案章节 | 状态 | 最终证据 |
| --- | --- | --- |
| §1–§3 决策与边界 | 完成 | v2-only 生产入口、浏览器仅交互/显示、后端事实源和任务范围边界成立。 |
| §4 运行架构 | 完成 | Launcher/API/Worker 分进程、单实例、epoch/heartbeat/fencing、自失租退出和独立拉起均有进程集成测试。局域网开放按用户最终口径执行。 |
| §5 存储架构 | 完成 | Alembic head `0012`；SQLite WAL/FK/busy timeout；明确 FK、删除语义和热路径索引；不可变资产、journal、凭据/插件版本及所有领域关系闭环。 |
| §6 快速工作区 | 完成 | 固定系统书/章、显式 reset、快速 bootstrap、new/existing book promote、约束处理和 job/operation/import lease 423 保护。 |
| §7 统一任务系统 | 完成 | 单全局队列、批次、排序、暂停/继续/取消/drain、replacement retry、SSE、事件游标、最近 200 批历史、Worker 控制命令和 crash recovery。 |
| §8 翻译后端化 | 完成 | 章节独立 job、批量创建、冻结配置/凭据/插件/字体、顺序/并行 Pool、HQ 稳定 ID batch、多轮校对、render/save 分界和失败项重试。 |
| §9 Insight | 完成 | 全书/增量/章节分析、派生物、时间线、向量、QA、续写和导出均进入后端持久任务或明确 transient contract。 |
| §10 插件 v3 | 完成 | 不可变版本、任务快照、Worker hooks、失败策略、Agent handoff、配置/管理 UI 和 API 元数据边界完整。 |
| §11 图片导入与缩略图 | 完成 | 普通图片逐页 multipart、容器后端任务、source/translated 各自缩略图、长条图特判、导入租约和失败恢复。 |
| §12 媒体 API 与加载 | 完成 | asset URL、ETag/304、列表缩略图、主图按需、Reader/侧栏/页码选择/Insight 树窗口化和离窗释放。 |
| §13 页面职责 | 完成 | 页面只提交命令并投影 REST/SSE 后端状态，不保存任务执行事实或整章图片。 |
| §14 实施阶段 | 完成 | 阶段 0–6 的契约、骨架、存储、任务、页面迁移、性能和最终清理均关闭。 |
| §15 全局验收 | 完成 | 确定性测试 Provider、串/并行一致性、存储一致性脚本、查询计划、故障注入、secret canary、浏览器/Worker 内存和生产构建全部落地。 |
| §16 翻译页 | 完成 | bootstrap/刷新恢复、章节工作态、最后访问页、编辑 CAS、修复 operation、指定页码、导出、Pool 进度和重试完整。 |
| §17 Insight 页 | 完成 | PagesTree/缩略图虚拟化、概览/时间线/QA/笔记/续写/设置、任务中心状态投影和切书隔离完整。 |
| §18 Character Studio | 完成 | 文档 CAS、generate/chat/summary operation、SSE chunk、abort generation fencing、会话/资产、归档删除和导入导出完整。 |
| §19 书架 | 完成 | 搜索/标签/排序、任务摘要、书籍/章节批量翻译、批量删除/标签、快速入口和任务定位完整。 |
| §20 任务中心 | 完成 | 全局抽屉、队列/历史、批次展开、筛选/定位、排序、取消/继续、失败项重试、产物、显存释放和批量分析完整。 |
| §21 阅读器 | 完成 | 全章摘要 + VirtualPageStream，translated→source 回退、未翻译徽标、m/N 统计和懒加载完整。 |
| §22 设置/提示词/Provider | 完成 | 分域设置、单书 Insight 覆盖、原子设置事务、不可变凭据、Provider 记忆、提示词/字体、诊断和 restricted mode 完整。 |
| §23 插件管理 | 完成 | v3 manifest/version/snapshot/runtime/Agent/管理 API/UI 与 OpenAPI 闭集完整。 |
| §24 当前不做 | 完成 | 未引入账号、多租户、云对象存储、瓦片、preview 层或单图解码预算，符合最终范围。 |

## 5. 存储、删除、加载与维护

### 5.1 结构

- 内容：books、chapters、pages、bubbles、tags、constraints、page_assets。
- 文件：assets、object_commit_journal、fonts 和明确的领域资产关联。
- 设置：app_settings、book_settings、provider_settings、prompts。
- 版本：credential_versions、plugin_versions 及稳定 current pointer。
- 长任务：job_batches、jobs、items、steps、events、config/credential/plugin/font snapshots、inputs/outputs/artifacts。
- 保存型短操作：operations、events、snapshots、inputs/artifacts、render_requests。
- 运行控制：process epochs、worker/API leases、write intents/locks、import leases、rate limits、idempotency records、worker_commands。
- Insight/Studio/续写均使用自己的规范化表与真实 FK，不使用多态 owner 字段或整页双事实 JSON。

### 5.2 查询与删除

- 迁移 `0012` 为所有 FK 和核心过滤/排序组合补齐索引。
- `EXPLAIN QUERY PLAN` 回归覆盖 1000 pages/assets、200 batches/jobs、10000 events 和 200 drafts，断言命中预期索引。
- 删除矩阵覆盖 book/chapter/page、Insight run、Studio、plugin/font、终态历史 `SET NULL` 和非终态领域保护。
- 快速 reset/promote 覆盖 job、operation、import lease、两种 promote 和 constraints 组合。

### 5.3 文件与向量一致性

- journal 覆盖 staging、object publish、DB 绑定前后的全部崩溃窗口。
- SQL 资产 GC、无 DB 记录 object 对账、journal/年轻文件保护和完整性状态修复由 Worker 周期执行；单项维护失败只记录并等待下轮，不中断任务调度。
- Chroma 统一存放在 `data_root/chroma`；按数据库 vector generation 识别缺失和孤儿 collection，只删除 Saber 管理的孤儿。
- 开发检查工具：`python -m scripts.check_v2_consistency --data-dir <data-root>`；只读检查 FK、路径、缺失文件、integrity 状态、孤儿 object 和向量 collection，发现分歧退出码为 1。

### 5.4 数据加载

- 翻译 bootstrap 一次返回章级摘要、设置、提示词、字体、活动任务和页面元数据；当前 document 与主图按需加载。
- 书架搜索/排序/标签/任务摘要在后端聚合。
- Reader 只保留视口窗口；翻译、编辑、指定页码和 Insight 大列表只挂载窗口附近缩略图。
- Studio 先取索引，再按需取文档/会话；Insight 非活动面板不常驻。

## 6. 可靠性、性能与安全门禁

- 确定性假 Provider 以测试作用域注册进共享 Provider registry，返回可预测文本和图像；生产 UI 不暴露它。
- 顺序与并行翻译以相同输入分别跑完整 Worker 流水线，并比较每页步骤序列、文档字段和资产结构。
- job/operation 续租 0 行、pause/cancel drain、API/Worker epoch 失效、迟到发布、Launcher 单独拉起和浏览器断线均有故障注入。
- 任务进度改为增量更新，消除了大 job 每个步骤全图重算造成的 O(N²)；1000 项持久任务图在独立进程内通过耗时和 RSS 有界门禁。
- 浏览器使用独立 Chromium + CDP + psutil，对 100/500/1000 页验证：
  - Reader；
  - 翻译右侧缩略图；
  - 编辑缩略图；
  - 指定页码弹窗；
  - Insight 页面树。
- `PageSelectionModal` 已修复把原图 URL 当缩略图加载的问题；`ThumbnailSidebar` 和页码弹窗的 flex 高度已限制，虚拟列表不会因父容器无限增高而一次挂载全部节点。
- API/Worker 的任务、事件、operation、Insight、Studio 和插件输出执行统一递归脱敏；测试 secret canary 不出现在 REST、SSE、错误、日志事实或快照。
- API 独立进程 probe 证明不加载 Torch、TorchVision、Chroma、旧 interfaces、旧 Insight 或插件业务模块。

## 7. 最终自动化结果

在项目 `venv/` 和前端本地依赖中完成：

- 全部后端：`268 passed`；其中 v2 独立套件 `166 passed`。
- 后端仅有 2 条 SQLAlchemy/Alembic 反射 SQLite 表达式索引的已知 warning，不影响迁移或运行。
- 前端 Vitest：`240` 个测试文件、`1725 passed`。
- Playwright：`32 passed`，包含全部 100/500/1000 页内存趋势和桌面/移动端视觉契约。
- `npm run typecheck`、`npm run lint`、`npm run lint:css`、`npm run lint:ui` 全部通过。
- `npm run check:api` 通过，OpenAPI 生成类型无漂移；运行时 Flask route set 与 OpenAPI operation set 双向闭集相等。
- `npm run build:check` 通过，`src/backend_v2/static/vue` 已由当前源码重建。

## 8. 收口判断

当前没有阻断重构完成的已知问题，也没有与最终方案不一致但未获接受的偏差。后续新增功能应继续遵守本方案中的事实源、进程边界、持久任务、资产关系、窗口化加载、幂等、fencing 和验收纪律；这属于日常演进，不再是本轮重构遗留。
