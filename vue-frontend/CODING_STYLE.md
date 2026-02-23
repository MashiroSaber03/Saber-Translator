# 前端 CSS 编码规范

> 本文档约束 **新代码**，已有类名不改动。

## 命名规范

| 规则 | 说明 |
|------|------|
| 新组件类名 | 使用 `组件名-元素名` 格式（如 `.qa-panel-header`）|
| 已有类名 | 不改动（`.btn`、`.form-group`、`.settings-group` 等）|
| CSS 变量 | 颜色用 `--color-xxx`，间距用 `--spacing-xxx`，圆角用 `--radius-xxx` |

## 强制规则

| 规则 | 说明 |
|------|------|
| **禁止 `!important`** | 通过 stylelint 强制 |
| **禁止 ID 选择器** | 通过 stylelint 强制 |
| **CSS 变量优先** | 颜色/圆角/阴影/字体必须用 `var(--xxx)` |
| **`<style scoped>` 为默认** | 除非有文档说明的例外（如 Teleport 场景）|
| **动画复用** | 通用 `@keyframes` 必须定义在 `global.css`，组件中不重复定义 |
| **z-index 统一** | 使用 `var(--z-*)` token，禁止新增硬编码 z-index > 50 |
| **font-family 统一** | 使用 `var(--font-sans)` / `var(--font-mono)` / `var(--font-jp)` |

## CSS 变量 Token 参考

```css
/* z-index 体系 */
--z-base: 1;       --z-dropdown: 100;  --z-sticky: 200;
--z-sidebar: 300;   --z-overlay: 1000;  --z-modal: 1100;
--z-popover: 1200;  --z-toast: 1300;

/* 字体族 */
--font-sans: 'Arial', sans-serif;
--font-mono: 'Consolas', 'Monaco', monospace;
--font-jp: 'Noto Sans JP', 'Yu Gothic', 'MS Gothic', sans-serif;

/* 断点（@media max-width） */
/* 480px=小屏  768px=平板  900px=中屏  1024px=大平板  1200px=桌面 */
```
