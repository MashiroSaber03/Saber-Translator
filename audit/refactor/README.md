# Backend-First V2 合并前全量审查清单

## 1. 审查目标与基线

- 目标：证明当前 `codex/backend-first-v2` 工作区可以合并到 `main`，且不会保留与 Backend-First V2 冲突的旧逻辑、旧规范、双轨状态、冗余兼容/迁移代码或无依据的过度设计。
- 当前分支基线：`1a140a5890c3955273aee6bd08b6111140986e7e`，并包含当前未提交工作区修改。
- 对照基线：本地 `main` 与 `origin/main` 均为 `0a7367adf64f084e63ed30281acd95df87fd2a8e`。
- 架构依据：`docs/refactor/backend-first-architecture-plan.md`、`openapi/v2.yaml`、当前正式路由/存储/任务契约与现有代码风格门禁。
- 最终盘点：977 个现存文本代码/配置/测试/生成文件，297 个仅存在于 `main` 的旧文本文件，89 个当前资源或视觉基线，以及 41 个仅在 `main` 存在的旧资源；审查过程中删除的冗余文件不再伪装成“当前文件”，其原因和验证分别保留在依赖图、问题日志和删除清单中。

本清单是审查事实记录，不是用测试结果替代人工阅读。只有对应文件已完整阅读、与 `main` 对照、调用链已核实且结论/证据已写入明细时，状态才允许从 `PENDING` 改为 `PASS`、`FIXED`、`DELETE-VERIFIED` 或 `GENERATED-VERIFIED`。

## 2. 不可违反的产品与架构约束

- 前端只负责展示、输入和交互；持久事实、任务执行、重试、暂停、取消与恢复由后端拥有。
- 所有正式业务接口只走 `/api/v2` 和 OpenAPI 契约，不保留旧 API、旧 session、旧 manager 或前端流水线的双轨实现。
- 不添加旧数据兼容、迁移、镜像字段或过渡读取；不兼容的旧数据直接拒绝或舍弃。
- 不添加用户未要求的图片大小限制、局域网限制、上传批次数量限制或其他任意产品约束。
- 不以“预防未来”为理由增加抽象层、适配器、仓储、状态镜像、缓存或 fallback；抽象必须由当前至少两个真实职责或明确契约证明。
- 模型服务商不稳定造成的请求失败不是项目逻辑缺陷；只检查错误传播、重试边界和状态一致性，不伪造成功或无限重试。
- 生成文件不是第二份手写架构：审查其源文件、构建过程、引用闭包和产物卫生，不直接修改压缩产物。
- 保留的 `src/core`、`src/interfaces`、`src/shared`、`src/utils` 只能包含当前 Worker/领域服务真实调用的算法与适配层，不能继续承载旧应用编排。

## 3. 状态与逐文件完成标准

状态：

- `PENDING`：尚未完成逐行审查。
- `REVIEWING`：正在阅读、追踪调用或修正。
- `PASS`：逐行审查和对照完成，没有问题。
- `FIXED`：发现问题并已修正，证据栏必须包含修改与验证。
- `DELETE-VERIFIED`：`main` 中的旧文件已确认被新架构完整取代，且当前无引用/打包残留。
- `GENERATED-VERIFIED`：生成产物已通过干净构建、引用闭包和陈旧文件检查。
- `RESOURCE-VERIFIED`：二进制/资源已验证来源、格式、引用与打包路径。

每个当前文本文件必须完成：

1. 阅读当前文件 `L1-LN`，不得只看 diff 或搜索命中。
2. 查看 `main` 对应文件或确认其为新增文件，并阅读相关差异。
3. 追踪入口、导入者、被调用者、状态所有者、API/数据库边界和测试。
4. 检查职责边界、命名、类型、异常、并发、资源生命周期、幂等、事务与安全边界。
5. 检查死代码、身份函数、重复实现、无效 fallback、旧版兼容、迁移、镜像状态和过度抽象。
6. 对 UI 文件额外检查结构、布局、组件契约、视觉 token、响应式、键盘/可访问性和滚动所有者。
7. 将结论、问题编号、修改文件和验证证据写回明细；没有证据不得标记完成。

## 4. 页面级清单（从外到内）

### 4.1 全局应用壳与跨页面能力

- [x] `main.ts`：应用初始化、Pinia、路由、全局样式和错误处理。
- [x] `App.vue`：后端受限提示、RouterView、任务中心、确认/文本输入 Provider、Toast 和 Overlay 层级。
- [x] `AppShell`/页眉：导航、放大比例、溢出、响应式、主题、图标及跨页面一致性。
- [x] 任务中心：快照/SSE、状态投影、暂停、取消、继续、重试、排序和错误反馈。
- [x] 全局设置、主题、Toast、确认框、文本输入、文件选择和所有 Teleport/Overlay 层级。

### 4.2 书架 `/`

- [x] 路由进入、pageshow 恢复、加载/空态/错误态。
- [x] 搜索、标签过滤、排序、选择、批量操作和主题切换。
- [x] 创建/编辑/删除书籍、章节管理和标签管理。
- [x] BookCard、BookSearch、BookModal、BookDetailModal、TagManageModal 及其所有下属组件。
- [x] 与 `main` 的视觉、行为和后端持久化差异逐项说明。

### 4.3 翻译 `/translate`

- [x] book/chapter 路由上下文、初始化、刷新恢复和设置记忆。
- [x] 图片逐张上传、缩略图、导入租约、幂等、错误中断和重新上传。
- [x] 顺序/并行翻译、所有模式、进度/SSE、页面状态、暂停/取消/继续/失败重试。
- [x] 设置侧栏全部分区与设置弹窗 9 个标签。
- [x] 原图/结果/编辑模式、气泡编辑、画笔、修复、渲染、快捷键和退出保存。
- [x] 缩略图、页面选择、术语表、不翻译表、网页导入、快速工作区提升和新手引导。
- [x] 赞助弹窗的内容、资源、响应式布局和关闭行为。
- [x] TranslateView、useTranslateViewActions 及 translate/edit/settings 下全部组件和组合函数。
- [x] 与 `main` 的所有结构、视觉和功能差异逐项说明。

### 4.4 阅读器 `/reader`

- [x] 缺少 book/chapter 时的路由保护。
- [x] 页面流虚拟化、懒加载、滚动定位、原图/译图、单双页与阅读方向。
- [x] 阅读控制、设置面板、键盘、缩放、移动端和大章节资源边界。
- [x] ReaderView、ReaderCanvas、ReaderControls 及其依赖逐项审查。

### 4.5 漫画分析 `/insight`

- [x] 书籍选择、bootstrap、分析任务、进度与结果刷新。
- [x] 总览、页面树、页面详情、时间线、QA、笔记、章节选择和设置。
- [x] 续写向导：角色、形态、剧本、参考图、图片生成、导出及状态恢复。
- [x] 所有 AI 服务错误传播、重试、空结果、部分结果和任务终态。
- [x] InsightView 与 insight/continuation/settings 下全部组件、store、composable 和 API。
- [x] 与 `main` 的所有结构、视觉和功能差异逐项说明。

### 4.6 角色工坊 `/insight/character-studio`

- [x] book/doc 路由上下文、候选角色、文档索引和空态。
- [x] Sidebar、Editor、Preview、Topbar、问候语、世界书、聊天、附件和版本/补丁工作流。
- [x] 图片/附件 URL 必须直接使用后端拥有的资源地址，不保留前端路径重建或身份适配器。
- [x] 独立滚动区、窄屏布局、未保存状态、Agent 操作与错误反馈。
- [x] CharacterStudioView 与 studio 下全部组件、store、API 和纯工具函数。

## 5. 组件级清单

- [x] `components/ui`：每个基础组件的唯一职责、公开 token、键盘/ARIA、Overlay 和原生控件边界。
- [x] `components/product`：产品级组合组件不得反向依赖业务领域，不复制 UI primitive 内部样式。
- [x] `components/common`：Modal、Toast、文件/颜色/文件夹等跨域能力的资源清理与错误反馈。
- [x] `components/bookshelf`：书籍/章节/标签域组件。
- [x] `components/translate` 与 `components/edit`：翻译/编辑全部组件和子目录。
- [x] `components/settings`：全部设置标签、字段类型、即时持久化和诊断流程。
- [x] `components/reader`：虚拟页面、控制与设置。
- [x] `components/insight`：总览、QA、笔记、时间线、续写、角色工坊和设置全部子目录。
- [x] `components/task-center`：任务投影和控制组件。
- [x] 所有 247 个现存组件目录文本文件均已在 `current-files.tsv` 获得逐文件结论；审查中删除的 10 个冗余组件/测试辅助文件由依赖图保留删除记录。

## 6. 前端支撑代码清单

- [x] router/constants：唯一正式路由和路径常量。
- [x] api/generated：OpenAPI 生成契约、请求客户端、SSE、下载和各领域 API。
- [x] stores：后端事实投影、本地纯 UI 状态、刷新/恢复、并发请求保护。
- [x] composables：生命周期、清理、竞态、职责拆分和无重复状态。
- [x] types：领域类型来源唯一，不保留旧 DTO/旧 session 类型。
- [x] utils/directives/styles：纯函数、浏览器资源回收、消毒、剪贴板、虚拟化、token 所有权。
- [x] Vite/Vitest/Playwright/ESLint/Stylelint/TypeScript 配置和测试架构。
- [x] 所有现存页面、入口与支撑文件均在 `current-files.tsv` 获得逐文件结论。

## 7. 后端领域清单

- [x] API 组装、请求/错误边界、静态 SPA、dispatch 和 OpenAPI 路由闭包。
- [x] content：书籍/章节/页面、上传、缩略图、样式与约束。
- [x] jobs：任务状态机、队列、步骤、批次、SSE、暂停/取消/继续/重试和并行领先窗口。
- [x] operations：短操作队列、领取、心跳、锁竞争、repair 和 fencing。
- [x] worker：注册、维护、模型生命周期、epoch、心跳和安全点。
- [x] translation/rendering：所有翻译模式、检测/OCR/颜色/修复/排版/保存，失败不得伪装成功。
- [x] insight/studio：分析、派生结果、QA、笔记、时间线、续写、角色文档和媒体。
- [x] web_import/transfer：下载、草稿、提交、容器导入导出、校验和与临时文件。
- [x] settings/plugins：当前配置解析/验证/诊断、插件契约/运行时/Agent。
- [x] storage：单一 foundation、当前 revision 拒绝、SQLite 事务/锁、资产、epoch、一致性和 GC。
- [x] redaction/logging/paths/runtime identity 等平台能力。
- [x] 所有 111 个现存 Backend V2 非桌面文本文件均在 `current-files.tsv` 获得逐文件结论。

## 8. 保留算法、桌面与项目外围清单

- [x] `src/core`、`src/interfaces`、`src/shared`、`src/utils` 的 87 个现存文件逐个确认仍被当前 Worker/领域服务使用，且无旧应用编排。
- [x] 桌面 GUI、桌宠、任务客户端、即时设置、主题、托盘、统一图标和跨平台启动。
- [x] Launcher/API/Worker 子进程、Windows Job Object、端口、日志和退出生命周期。
- [x] 示例插件及插件文档只符合当前 V2 契约。
- [x] app.spec、requirements、release workflow、脚本、README、OpenAPI 和架构文档。
- [x] 279 个现存测试文件逐个确认测试当前契约，不保留推动旧架构继续存在的测试。

## 9. `main` 删除覆盖清单

- [x] 逐个审查 `main-removed-files.tsv` 中 297 个旧文本文件。
- [x] 每个删除项均已对应当前替代入口/领域，或证明该能力按重构方案明确取消。
- [x] 已搜索导入、字符串路径、PyInstaller hidden import、文档、测试和生成产物中的残留。
- [x] 删除结论包含逐路径替代/取消证据，不以测试结果单独代替审查。

## 10. 资源与生成产物清单

- [x] 89 个现存字体、图片、图标、桌宠 spritesheet 与视觉基线逐个验证引用和打包路径。
- [x] 41 个仅在 `main` 存在的资源逐个确认已删除、重命名或迁入当前正式资源目录。
- [x] 桌面 PNG/ICO 由项目 Logo 生成，圆角透明，多尺寸完整，并统一用于应用/窗口/托盘/侧栏。
- [x] `src/backend_v2/static/vue` 来自当前前端干净构建，39 个文件全部可达，无旧哈希孤儿文件。
- [x] 第三方模型权重不纳入业务源码风格审查；其装载路径、打包边界和错误传播已核对。

## 11. 最终验证门禁

- [x] 所有明细行不再包含 `PENDING` 或 `REVIEWING`；完成性复审重新打开的 43 条空占位记录已逐项重读并补齐具体结论和证据。
- [x] 所有问题有编号、根因、修改、无兼容/迁移说明和验证证据。
- [x] Python 静态检查、编译和完整后端测试。
- [x] TypeScript、ESLint、Stylelint、UI 架构、OpenAPI、完整前端测试和生产构建。
- [x] 全部浏览器视觉基线与五个页面/全局弹层实机检查。
- [x] 桌面四页、桌宠状态、托盘/图标、即时设置和启动/退出实机检查。
- [x] `git diff --check`、生成物闭包、未跟踪文件和最终完整 diff 复核。
- [x] 已按本清单重新完成一次独立完成性审计；补齐漏记明细、历史行数和陈旧 OpenAPI 生成物后，全部集合与门禁重新通过。

## 12. 明细文件

- `current-files.tsv`：记录 977 个现存文本代码、配置、测试与生成文件的完整行范围、分组、相对 `main` 状态、结论和证据。
- `main-removed-files.tsv`：仅在 `main` 存在的 297 个文本文件，记录原行数、替代关系和删除验证。
- `resources.tsv`：89 个现存资源和 41 个 `main` 已移除资源，记录大小、来源、引用和打包验证。
- `page-component-map.md`：五个页面、全局壳与其直接/传递组件和支撑文件映射。
- `backend-domain-map.md`：API、Worker、桌面、Launcher、插件与保留算法的角色依赖闭包。
- `functional-contracts.tsv`：181 个当前 HTTP 操作、3 个已删除 HTTP 入口的删除证据、45 个 Worker 步骤及批处理、Operation、页面、设置和插件契约；动态 Insight layer 以一个无固定上限的模式契约记录。
- `findings.md`：逐项问题、修正和验证日志。

## 13. 进度日志

- 2026-08-10：建立审查方法、基线、页面/组件依赖图、后端角色图和功能契约清单。
- 2026-08-10：清单集合校验通过：983 个当前文本文件、285 个 `main` 缺失文本文件、104 个当前资源、38 个 `main` 缺失资源和初始 280 个逐项功能契约均无漏项、重复项或错误行范围；后续审查已将 8 个固定 Insight layer 条目规范化为 1 个动态模式契约，当前为 273 项；尚未将任何逐文件项目标记为完成。
- 2026-08-10：完成入口/路由第一轮和任务中心主体逐行审查，记录并修正 AUDIT-001 至 AUDIT-006；任务中心仍等待 SSE 事件全集、弹层依赖和浏览器视觉核对，因此主体文件保持 `REVIEWING`。
- 2026-08-10：核对任务中心队列/历史端到端契约，修正实时队列被历史 `limit=200` 错误截断的问题（AUDIT-007）；205 个非终态任务的完整可见性回归测试通过。
- 2026-08-10：完成全局弹层公共契约、Toast、任务中心空态/移动端/嵌套对话框实机审查，修正 AUDIT-008 至 AUDIT-012；任务中心带真实任务卡片的最终浏览器回归仍留在页面总体验收阶段。
- 2026-08-10：完成书架页从路由、列表、搜索/筛选/排序、批量操作到书籍/章节/标签弹窗的逐文件与桌面/390px 浏览器审查，修正 AUDIT-013 至 AUDIT-017；9 个定向文件 102 tests、typecheck、UI architecture 与差异检查通过。当前文本清单状态为 30 FIXED、17 PASS、13 REVIEWING、924 PENDING，尚未进入全量完成状态。
- 2026-08-10：完成角色工坊从 book/doc 路由上下文、候选/文档/版本到编辑、世界书、会话、聊天、附件、Agent、诊断和 Studio 后端的逐文件审查，修正 AUDIT-018 至 AUDIT-025；前端角色工坊 183 tests、Backend Studio 35 tests、typecheck、三类 lint、OpenAPI 确定性生成及桌面/390px 与 `main` 全状态浏览器对照通过。当前文本清单状态为 61 FIXED、49 PASS、1 GENERATED-VERIFIED、14 REVIEWING、859 PENDING，尚未进入全量完成状态。
- 2026-08-10：完成 Insight 设置弹窗七个标签、共享设置组件和当前 settings/prompt transaction 契约审查，修正 AUDIT-026 至 AUDIT-028；删除 Batch 旧字段兼容和重复状态，修复末层 `units=0` 被改写，收紧异步请求身份及提示词事务。内置浏览器与 `main` 对照并实测自定义架构，设置定向 110 tests、typecheck、ESLint、Stylelint 和差异检查通过；全项目审查继续进行。
- 2026-08-10：完成 Insight 续写四步、角色/形态/参考图/三视图弹层、前端状态 owner、OpenAPI/HTTP/Repository/Worker 与版本历史审查，修正 AUDIT-029 至 AUDIT-032；删除浏览器实体缓存、旧快照回退、任意数量上限、虚假多图入口和五版本清理，并分离剧情 ready 与图片待生成投影。续写前端 121 tests、补充弹层契约 41 tests、Backend Insight 44 tests、OpenAPI 9 tests、typecheck、三类 lint、compileall、确定性生成及内置浏览器四步全状态通过。当前文本清单状态为 105 FIXED、68 PASS、1 GENERATED-VERIFIED、16 REVIEWING、796 PENDING，仍未进入全量完成状态。
- 2026-08-10：完成 Content 领域七个实现文件和 Stage2 测试逐行审查，连同 Insight 当前配置及导入数量边界修正 AUDIT-033 至 AUDIT-035；删除任意聚合/数量门槛、标量强转、旧 schema 复制和静默修正，HTML 改为增量解析。Content 60 tests、Platform/Content/Web Import 组合 110 tests、Insight Backend 60 tests、前端配置 30 tests、compileall 与差异检查通过。当前文本清单状态为 111 FIXED、70 PASS、1 GENERATED-VERIFIED、16 REVIEWING、788 PENDING，仍未进入全量完成状态。
- 2026-08-10：完成 Jobs 与 Operations 基础设施逐行审查并修正 AUDIT-036 至 AUDIT-037；闭合取消/失败子图、严格当前重试快照和 Worker 配置、消除任务详情 N+1，补齐 attempt 租约恢复、远程副作用裁决、新旧 render revision 隔离、active operation 编辑屏障和 repair 当前契约。Jobs 43 tests、Operations 27 tests、Content 60 tests、OpenAPI 9 tests 通过。当前文本清单状态为 121 FIXED、73 PASS、1 GENERATED-VERIFIED、14 REVIEWING、777 PENDING，仍未进入全量完成状态。
- 2026-08-10：完成 Worker、epoch heartbeat、API/Launcher 运行监督与模型生命周期逐行审查并修正 AUDIT-038 至 AUDIT-041；失联恢复闭合任务子图/operation event/render 页面状态，模型释放可在 SQLite BUSY 后幂等续办，Paddle ONNX 纳入全模型释放，并删除 Insight Worker 的 8 层处理器上限；同时修复共享恢复测试假阳性。Runtime 4 tests、Worker lifecycle/maintenance 8 tests、Stage1 48 tests、角色边界/Windows 集成 14 tests 通过。当前文本清单状态为 129 FIXED、82 PASS、1 GENERATED-VERIFIED、15 REVIEWING、759 PENDING，仍未进入全量完成状态。
- 2026-08-10：完成翻译流水线的当前气泡文档、检测、OCR/颜色、修复/LaMA 核心与接口逐文件审查并修正 AUDIT-042 至 AUDIT-045；删除旧字段/语言/模型 fallback、伪空成功、跨模型切换、冗余修复副本与逐页强制清理，补齐 AI 视觉冻结选项物化、精确配置准入和 repair 页面/气泡 revision 原子一致性。OCR/Provider/Stage4/算法定向 133 tests passed。当前文本清单状态为 168 FIXED、88 PASS、1 GENERATED-VERIFIED、14 REVIEWING、716 PENDING，仍未进入全量完成状态。
- 2026-08-10：完成权威文字渲染、字体物化、渲染服务和翻译 HTTP 入口逐文件审查并修正 AUDIT-046；删除字体/字号/无效气泡静默 fallback，自动字号与任一气泡失败均严格传播，修复旋转气泡有限小数位置偏移。渲染+Stage4 91 tests、Content/样式 76 tests 通过。当前文本清单状态为 170 FIXED、93 PASS、1 GENERATED-VERIFIED、14 REVIEWING、709 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成普通翻译服务商与共享 AI 选项/执行/传输/限流链逐文件审查并修正 AUDIT-047；删除全局可变凭据、请求覆盖旁路、旧/缺配置补齐、畸形响应容忍、错误消息重试判断和每页 4000 字任意拆批，补齐严格 SSE/JSON/嵌入/模型列表/截止时间及分层有限重试。Provider/Transport/Insight 96 tests、翻译+Stage4 99 tests、上层 204 tests 通过。当前文本清单状态为 183 FIXED、93 PASS、1 GENERATED-VERIFIED、14 REVIEWING、696 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成 Manga Insight Provider 配置、保留客户端、分析命令与 Worker run 事务的当前批次审查并修正 AUDIT-048 至 AUDIT-049；删除旧配置/基类/猜测解析器、死客户端 API、分析 `force` 与 runId 兼容壳，修复验证失败回滚 run 终态。Backend Stage5 67 tests、前端相关 26 tests、typecheck 与 py_compile 通过。滚动清单状态为 201 FIXED、94 PASS、1 GENERATED-VERIFIED、18 REVIEWING、674 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成 Insight QA 临时请求状态机、检索/重排/回答服务商边界及前端 SSE/切书状态的逐行审查并修正 AUDIT-050；补齐 fenced 过期领取、API/Worker 共同心跳与失活清理，拒绝畸形/部分/空结果，删除 QA 任意数量上限和前端类型强转，不保留旧数据兼容。Backend Stage5 72 tests、前端 QA/API 32 tests、typecheck、ESLint 与 compileall 通过。滚动清单状态为 204 FIXED、94 PASS、1 GENERATED-VERIFIED、19 REVIEWING、670 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成 Insight 派生层、总览、时间线、向量 generation 与前端时间线响应边界的逐行审查并修正 AUDIT-051；阻止空/部分模型结果发布、旧全量 run 和陈旧压缩上下文复用、损坏存储强转及跨 generation 分页混合，不添加旧数据迁移或兼容。Backend Stage5 85 tests、前端时间线/API 28 tests、typecheck、ESLint、Prettier、py_compile、OpenAPI 确定性生成与 diff check 通过。滚动清单状态为 207 FIXED、94 PASS、1 GENERATED-VERIFIED、18 REVIEWING、668 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成 Insight 分析/导出事务准入、页面发布窗口、run 预览与续写图片 preflight 审查并修正 AUDIT-052；目标或快照并发变化时整笔拒绝，页面在完整 run 发布前保持 running，模型调用前后都核对页面 revision，并将生产 assert 改为显式领域冲突。Backend Stage5 99 tests 通过。滚动清单状态为 211 FIXED、96 PASS、1 GENERATED-VERIFIED、16 REVIEWING、664 PENDING，仍未进入全量完成状态。
- 2026-08-11：完成 Insight 前端页身份/时间线缩略图、页面树/详情五态、笔记筛选分页和概览加载区段审查并修正 AUDIT-053 至 AUDIT-056；删除模块缓存、300 条任意上限、重复笔记 DTO/mapper 与 answer 镜像，修复同计数重分析刷新、手动笔记页码更新、元数据失败遮蔽内容和最近分析伪空态。相关前端 105 tests、typecheck、ESLint、Stylelint 通过。滚动清单状态为 219 FIXED、97 PASS、1 GENERATED-VERIFIED、24 REVIEWING、647 PENDING；其中 8 个组件/测试保留 REVIEWING 等待 Insight 页面整体浏览器验收，仍未进入全量完成状态。
- 2026-08-11：完成 Insight 分析进度组件、状态类型与两组测试的逐行审查并修正 AUDIT-057；新任务不再继承旧任务进度，删除无生产来源的 `error` 兼容状态和重复字段间距。定向 21 tests、typecheck、ESLint、Stylelint 通过。滚动清单状态为 221 FIXED、97 PASS、1 GENERATED-VERIFIED、25 REVIEWING、644 PENDING；组件仍等待 Insight 页面整体浏览器视觉验收，全量审查尚未完成。
- 2026-08-11：完成 Insight 时间线面板、九个呈现子组件、组合状态、API 映射及后端剧情弧/伏笔发布边界审查并修正 AUDIT-058；错误/空态分离，生成期间保留旧版本，删除伏笔主题镜像和前五条截断，稳定剧情弧/角色身份并收紧 OpenAPI。Stage5 100 tests、OpenAPI 9 tests、前端时间线/API 48 tests、typecheck、ESLint、Stylelint 与确定性生成通过。滚动清单状态为 222 FIXED、97 PASS、1 GENERATED-VERIFIED、35 REVIEWING、633 PENDING；时间线呈现组件等待 Insight 页面浏览器总体验收，全量审查尚未完成。
- 2026-08-11：完成 Insight QA 呈现、向量/派生任务恢复、流式协议、模式切换和问答笔记保存审查并修正 AUDIT-059；中断任务与已受理修复可从任务中心恢复，切书/切模式严格隔离旧流，删除重复请求字段、建议问题与消息镜像，SSE 收敛为当前唯一顺序。前端 QA/API 58 tests、Backend Stage5 100 tests、typecheck、ESLint、Stylelint 与 py_compile 通过。滚动清单状态为 227 FIXED、97 PASS、1 GENERATED-VERIFIED、40 REVIEWING、623 PENDING；QA 呈现组件等待 Insight 页面浏览器总体验收，全量审查尚未完成。
- 2026-08-11：完成 Insight 入口、无筛选书籍选择、章节弹窗、三栏/标签工作区、路由和任务事件投影审查并修正 AUDIT-060；核心书籍快照改为原子装载，删除章节/状态伪回退，补齐卸载与跨书竞态 fencing、真实历史任务恢复、ARIA/键盘焦点和空右栏控制。96 个单元测试、typecheck、ESLint、Stylelint、8 个 Insight 桌面/移动视觉用例及与 `main` 的空态实机对照通过。滚动清单状态为 238 FIXED、98 PASS、1 GENERATED-VERIFIED、40 REVIEWING、611 PENDING；全量审查尚未完成。
- 2026-08-11：完成任务中心/Insight 页进度投影及其测试逐行审查并修正 AUDIT-061；删除已由 OpenAPI 保证字段上的 `Number`/`Boolean` 强转、静默归零、超量夹断和目标页数回退，任务中心并行池补齐总数/失败/跳过显示。18 个定向 tests、84 个相邻组合 tests、typecheck、ESLint、Stylelint 与 diff check 通过。滚动清单状态为 242 FIXED、98 PASS、1 GENERATED-VERIFIED、40 REVIEWING、607 PENDING；带真实任务卡片的最终视觉验收仍待后续，全量审查尚未完成。
- 2026-08-11：完成 Insight Store、服务商草稿缓存、设置模态和笔记写入边界审查并修正 AUDIT-062；真实写入异常不再被 `null`/布尔值和笼统提示覆盖，删除重复页面选择入口，设置草稿改为深拷贝，服务商切换保留完整 DTO 的明确空值，提示词采用完整替换。相关 121 tests、typecheck、ESLint、Stylelint 与 diff check 通过。滚动清单状态为 246 FIXED、102 PASS、1 GENERATED-VERIFIED、39 REVIEWING、600 PENDING；全量审查尚未完成。
- 2026-08-11：完成 Insight API facade 余下区段和设置事务端到端审查并修正 AUDIT-063；删除三份旧式 snake_case 中间 DTO 映射，Store/API 统一当前设置快照，事务成功后直接采用无密钥结果并删除整套重复 GET 及假失败分支。相关 115 tests、typecheck、ESLint 与残留搜索通过。滚动清单状态为 246 FIXED、101 PASS、3 DELETE-VERIFIED、1 GENERATED-VERIFIED、38 REVIEWING、599 PENDING；全量审查尚未完成。
- 2026-08-11：在全新隔离数据库和真实浏览器任务中心验收中修正 AUDIT-064 至 AUDIT-065；未保存的 Insight provider 草稿现直接使用项目共享 Manifest 默认模型，任务卡和展开详情统一为页进度，不对内部发布 item 重复计数。实测覆盖设置保存/脱敏回显、真实分析排队、详情、历史与取消；相关 25 tests、typecheck、ESLint 通过。滚动清单状态为 247 FIXED、101 PASS、3 DELETE-VERIFIED、1 GENERATED-VERIFIED、37 REVIEWING、599 PENDING；全量审查尚未完成。
- 2026-08-11：完成阅读器设置、书/章/页身份、窗口化资产错误和沉浸式全局生命周期审查并修正 AUDIT-066；宽度/间距/背景现直接作用于真实画布，失败装载不再重现旧图，错误图片保留几何占位，阅读路由不再挂载任务中心或 SSE。阅读器与 App 生命周期 50 tests、typecheck、ESLint、Stylelint、2 个 Playwright 视觉用例及桌面/390px 内置浏览器实测通过。滚动清单状态为 256 FIXED、105 PASS、3 DELETE-VERIFIED、1 GENERATED-VERIFIED、37 REVIEWING、587 PENDING；全量审查尚未完成。
- 2026-08-13：完成最终全量闭环并修正 AUDIT-163 至 AUDIT-164；桌面任务 SSE 停止不再受同步 `finished` 回调竞态影响，根目录依赖缓存与独立探针目录纳入生成物边界。当前文件 977 项（383 FIXED、557 PASS、37 GENERATED-VERIFIED）、重构删除项 297 项、资源 130 项、功能契约 273 项均已审查，`PENDING/REVIEWING` 为 0。Backend 858 tests、前端 1969 tests、Playwright 34 tests、桌面 21 tests、打包契约 11 tests、typecheck、ESLint、Stylelint、UI architecture、Python compileall、OpenAPI 确定性生成、静态资源闭包、依赖审计及 diff check 全部通过；五个前端路由、全局弹层、桌面 GUI、托盘/窗口图标、设置即时生效、桌宠与服务生命周期均完成实机验证。
- 2026-08-13：重新执行完成性审计并修正 AUDIT-165 至 AUDIT-166；补齐 12 个 `main` 删除文本文件、3 个旧资源和 43 条空证据记录，校正 9 个历史行范围，并从当前 OpenAPI 重建陈旧的 TypeScript 客户端。集合复核为当前 977 文本 + 89 资源、`main` 删除 297 文本 + 41 资源、273 功能契约，全部零漏项/重复/交叉/未完成。修正后重新通过 Backend 858 tests、前端 226 files / 1969 tests、Playwright 34/34、桌面 33 tests、打包 6 tests、typecheck、ESLint、Stylelint、UI architecture、生产构建、compileall、依赖审计、39/39 静态闭包和 diff check。
- 2026-08-13：按最终工作区再次全量复审并修正 AUDIT-167 至 AUDIT-171；归并前后端任务状态与多消费者小工具，修复 Studio 诊断草稿解码和伪 OpenAPI 上限，统一 CLI/GUI 本机监听默认值，删除检测链隐式 `**kwargs`、未用参数和前端提示词伪上限，并闭合 Playwright 点文件忽略规则。当前清单为 977 文本（397 FIXED、543 PASS、37 GENERATED-VERIFIED）+ 89 资源，`main` 删除 297 文本 + 41 资源，273 功能契约（181 当前 HTTP + 3 删除证据），全部零漏项/重复/交叉/未完成。最终 Backend 858/858、前端 226 files / 1970 tests、Playwright 34/34、typecheck、ESLint、Stylelint、UI architecture、Pyflakes、compileall、OpenAPI 双生成、生产构建、npm audit 0、39/39 静态闭包与 diff check 全部通过。
- 2026-08-13：最终审阅结果自检修正 AUDIT-172；恢复两份被终端编码破坏的派生映射状态，并校正一条生成静态文件行号。主台账与生产代码没有受影响；复核后 1066 个当前文件/资源、338 个 `main` 删除文件/资源、273 个功能契约和 172 条发现仍保持零漏项、零重复、零交叉。
- 2026-08-13：实机普通翻译回归修正 AUDIT-173；SiliconFlow 等内置服务商的空自定义 Base URL 在单条与批量普通翻译请求边界统一转为 `None`，由唯一 Provider 清单采用默认地址。未放宽严格请求 DTO，未增加旧数据兼容或迁移。翻译故障语义 34 tests、Stage4/Provider 114 tests 通过。
- 2026-08-13：针对审阅后实机回归再次完成全工作区复核并修正 AUDIT-174 至 AUDIT-175；Studio 同类空 Base URL 在现有请求边界归一化，主台账 41 条被终端编码破坏的审阅结果已按当前文件重新核实并恢复。当前 977 个文本文件、89 个资源、297 个 `main` 删除文本、41 个删除资源、273 条功能契约和 175 条发现均通过集合、行范围、状态与证据校验；Backend 860/860、前端 226 files / 1970 tests、Playwright 34/34、typecheck、ESLint、Stylelint、UI architecture、Pyflakes、compileall、生产构建、npm audit 0、39/39 静态资源闭包及 diff check 全部通过。两处 Base URL 修复均保持严格 DTO，只在真实 Provider 请求边界处理空可选值；未增加旧数据兼容、迁移或通用抽象。
