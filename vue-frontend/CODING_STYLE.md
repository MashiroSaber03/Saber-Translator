# 前端 UI 架构规范

> 最后更新：2026-06-25
> 目标：新增和重构 UI 默认走统一 token、layout shell、overlay layer 和 primitives。

## 样式分层

| 文件 | 责任 |
|------|------|
| `src/styles/tokens/*.css` | 唯一 token 来源，按 `foundation(custom media/layout constants) -> semantic -> component -> domain` 顺序显式导入 |
| `src/styles/reset.css` | 最小浏览器 reset；不得承载业务布局或组件样式 |
| `src/styles/base.css` | 保持空壳或极少低层工具；业务样式必须下沉到组件或 primitive |
| `src/components/ui/*` | 不绑定业务的按钮、字段、输入、面板、布局外壳等基础 UI |
| `src/components/**/*.global.styles.css` | 仅用于 Teleport、slot reach-through 或 body 状态这类明确全局 owner |

## 默认实现路径

新增页面顶层布局使用 `AppShell`。有左右侧栏或三栏结构时，外层使用 `SidebarLayout`，页面文件只负责自己的 grid/flex 尺寸和业务区域命名。

普通按钮使用 `UiButton`，图标/关闭/工具栏按钮优先使用 `UiIconButton`。禁止在业务源码新增 raw `<button>`，也禁止恢复 `.btn`、`.primary-btn`、`.ghost-btn`、`.ui-action-btn` 等旧按钮类。

表单字段默认由 `UiField`、`UiInput`、`UiTextarea`、`UiModalSection` 承载。业务组件只写业务命名空间样式，不再通过父组件 `:deep()` 修改子组件内部。

普通弹窗使用 `BaseModal` 或 `ConfirmModal`。需要定制 Teleport 容器时，必须使用明确的 `custom-class`/`overlay-class` 和同目录命名空间 `.global.styles.css`；业务 SFC 中禁止 `:global()`。

## Token 规则

颜色、圆角、阴影、断点、z-index 必须来自 `src/styles/tokens/*`。业务组件禁止新增硬编码颜色、`--ui-color-*`、`--border-radius-*`、裸 `z-index` 数字和裸 `@media (...px)` 断点。

业务组件只允许引用语义 token，例如 `--color-action-primary`、`--color-text-default`、`--color-border-muted`、`--color-surface-card`、组件 scoped owner token、`--radius-*`、`--z-*`。`--palette-*` 中间层不再使用；跨 owner 的颜色值归属到 semantic/component/domain token，单组件私有视觉值只归属到该组件 scoped owner 根变量。历史值命名颜色 token 已废弃，全仓库不得出现。

Owner token 不是状态矩阵仓库。组件 scoped 根变量必须同时满足三个条件：命名包含明确 owner 和 UI 角色，至少有一个真实 `var(...)` / `setProperty` / `removeProperty` 引用，且定义位置属于消费它的组件 owner。不要在父组件集中定义子 section、toolbar、modal body、canvas、card 等私有视觉变量；这些变量应下沉到对应子组件根选择器。`npm run lint:ui` 会拒绝未被引用的 owner token 定义，`npm run lint:ui:audit` 会列出 owner token density 作为维护审查信号。

`domain.css` 只保留跨页面、跨业务域共享主题 token；当前只允许 `--insight-*` 这类被页面和多个子组件共同消费的 domain theme。单个组件或页面私有 token 必须定义在自己的 `<style scoped>` owner 根选择器中；`domain.css` 超过 50 个 token、出现 `*-panel-*`、`*-modal-*`、`*-card-*`、`*-tab-*`、`*-canvas-*` 等组件私有 token，或出现 `variant-012` 这类生成式命名，都会让 `npm run lint:ui` 失败。

业务 CSS 不得重新定义全局/历史通用 token，例如 `--text-primary`、`--text-secondary`、`--text-muted`、`--color-primary`、`--bg-primary`、`--bg-secondary`、`--success-color`、`--error-color`、`--primary`、`--danger`。页面或组件需要局部主题变量时，必须使用 owner 命名空间，例如 `--insight-text-primary`、`--translate-surface-main`、`--reader-control-text`、`--studio-accent`。这条规则避免页面根变量通过 CSS 继承污染子组件，是 `npm run lint:ui` 的强制检查项。

所有 semantic、domain、component、layout token 必须定义在全局 `:root` 中。`body` 不得定义 token 或兼容别名；旧 `--text-color`、`--bg-color`、`--card-bg-color`、`--border-color`、`--input-bg` 等变量的定义和引用都禁止出现。`:root` token 不得引用 body-only token，也不得引用未定义 token 或形成循环依赖；`npm run lint:ui` 会解析所有 token 文件的 `var(...)` 依赖链并在提交前失败。这条规则用于防止编辑模式背景这类“变量存在但作用域无效”的隐性视觉回归。

`--radius-*-legacy` 和 `--border-radius-*` 不允许继续存在；需要 16px 这类历史尺寸时，先在对应 token 文件增加有语义的 `--radius-*` token，再迁移引用。

断点统一使用 `@custom-media`：

```css
@media (--breakpoint-md-down) {
  /* ... */
}
```

不要在组件内写 `@media (max-width: 768px)` 这类裸数字断点。

## 命名与边界

组件样式默认 `<style scoped>`。类名使用组件命名空间或 BEM 风格，例如 `.qa-panel__header`、`.book-card__cover`。禁止 CSS ID 选择器和 `!important`。

禁止 `<style src>`、CSS `@import`、`*.generated-*.css`、`part1/part2` 以及 `*.base/layout/panels/responsive.styles.css` 这类横切拆分。业务组件不得通过 `<script setup>` 导入普通 `*.styles.css`，因为它会绕过 Vue scoped 边界并形成隐藏全局样式入口。组件样式默认写在同一 SFC 的 `<style scoped>` 中；确实需要 Teleport 或 slot reach-through 时，使用同目录命名空间 `*.global.styles.css`，并让选择器从明确的 modal/body/owner class 开始。共享样式应沉淀为 UI primitive 或明确业务命名空间，不要用 scoped 共享 CSS 模拟全局组件库。

行数预算只是硬安全网，不是拆分目标。SFC 的 owner 门禁按模板+脚本逻辑行检查，scoped 样式另设安全网上限；不要为了贴近 300 行而横向切成 `base/layout/panels`。如果一个组件虽然较大但状态、模板和交互必须一起理解，可以保留整体。拆分只在真实 owner 边界清晰、能减少状态耦合或测试困难时进行。

`docs/ui-maintenance-decisions.md` 记录最终维护策略，不记录“以后再拆”的待办项。未登记的超硬安全网上限文件会让 `lint:ui` 失败；正常 audit 不输出大文件候选或待拆分决策。

页面级 fixed、`calc(100vh - ...)`、以及用固定 margin 推开侧栏的布局都属于 layout bypass。新增这类布局前必须优先扩展 `AppShell`、`SidebarLayout` 或 `OverlayLayer`；确实属于 Teleport、编辑器覆盖层、Reader 沉浸控件、面板内部滚动等局部 owner 时，必须纳入 `check-ui-architecture.mjs` 的永久 layout owner 表。永久 owner 表只表达当前所有权，不记录迁移待办。

业务 CSS 禁止直接选择 UI primitive 内部类，例如 `.ui-input`、`.ui-select`、`.ui-textarea`、`.ui-form-field`。需要不同尺寸、密度或视觉状态时，先扩展 primitive props/class contract，或给业务组件自己的元素加业务命名空间类。

页面样式不得重写 `AppHeader` 内部结构。`AppHeader` 内部类归 `AppHeader` 自己所有；页面通过 header slot 放入的元素必须使用页面自己的命名空间类。

`:global()` 和 `:deep()` 在 UI 源码中禁止使用。Teleport、body 状态类和 slot 内容样式必须放到同目录命名空间 `.global.styles.css` 或 UI primitive 中。

## 自动化门禁

`npm run lint:ui` 对 UI 架构债使用零预算：旧兼容变量定义/引用、旧短别名变量、业务视觉兼容 token、实现型 semantic token、未引用 owner token、历史值命名颜色 token、编号式 owner token、raw form controls、`variant="tool"`、`variant="unstyled"`、普通 script CSS import、`style src`、CSS `@import`、`*.generated-*.css`、`:global()`、`:deep()`、旧 settings shared 类、未归属的全局选择器、旧实现心智注释和业务 CSS 直接选择 primitive 内部类都会失败。默认输出必须保持安静，通过时只输出 `UI architecture check passed`。

`npm run lint:ui:audit` 用于主动维护审查：它输出 token 文件数量、`:root` token 数量、token 依赖数量、关键视觉覆盖数量、heavy owner review signals 和 owner token density signals。它不输出 accepted debt 或待处理 layout bypass；如果某个文件从“保留更清晰”变成“多 owner 混杂”，应按真实 UI 边界拆分或抽 composable，并同步更新维护策略。

关键页面状态必须有视觉覆盖登记。默认 `lint:ui` 会检查 `tests/visual/ui-regression.spec.ts` 中是否仍覆盖 Translate 普通/已加载/编辑态、EditWorkspace 选中气泡编辑面板、Insight 侧栏、Reader、Character Studio 等高风险 UI shell。新增或重构关键状态时，应补视觉测试，而不是只依赖普通页面截图。

每次 UI 架构或样式改动至少运行：

```bash
npm run lint:ui
npm run lint:css
npm run lint
npm run typecheck
```

触及布局、弹窗、侧栏、表单、按钮、token 时还必须运行：

```bash
npm run visual:test
```

最终合并前运行：

```bash
npm run build
npm test
npm run visual:test
```

`visual:baseline` 只在确认当前视觉就是目标视觉时使用，不能用来掩盖结构改动造成的视觉漂移。
