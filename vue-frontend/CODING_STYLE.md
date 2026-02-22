# Saber-Translator CSS 编码规范

## 核心原则

> **重构不改变前端 UI** — 所有样式修改必须保持像素级视觉一致性

## CSS 规范

### 1. 样式作用域
- **默认使用 `<style scoped>`**
- 仅在必要时使用非 scoped（如 Teleport 组件），需添加命名空间前缀
- 禁止在 scoped 样式中使用 `body`、`:root` 等非组件选择器

### 2. 选择器规范
- ❌ **禁止使用 ID 选择器** (`#element-id`)
- ✅ 使用 class 选择器
- ✅ 新组件采用 `组件名-元素名` 格式（如 `.modal-header`, `.card-title`）
- 已有类名保持不变，仅约束新代码

### 3. 样式优先级
- ❌ **禁止使用 `!important`**
- ✅ 通过提高选择器特异性解决优先级问题
- ✅ 合理使用组件层级和 BEM 命名

### 4. CSS 变量
- ✅ **颜色必须使用 CSS 变量** (`var(--color-primary)`)
- ✅ **圆角必须使用 CSS 变量** (`var(--radius-md)`)
- ✅ **阴影必须使用 CSS 变量** (`var(--shadow-base)`)
- ✅ **字体必须使用 CSS 变量** (`var(--font-sans)`)
- 禁止硬编码 `#hex` 颜色值、`px` 圆角值

### 5. 动画复用
- ✅ 通用 `@keyframes` 定义在 `global.css`
- ✅ 组件特定动画可在组件内定义
- 避免重复定义同名动画

### 6. 响应式断点
推荐使用统一断点：
- 小屏：`480px`
- 平板：`768px`
- 中屏：`900px`
- 大平板：`1024px`
- 桌面：`1200px`

### 7. 性能优化
- ✅ `transition` 指定具体属性，避免 `transition: all`
- ✅ 避免深层嵌套选择器（3层以内）
- ✅ 合理使用 `will-change`

### 8. 可访问性
- ✅ 为交互元素提供可见的 `:focus` 样式
- ⚠️  慎用 `outline: none`，必须提供替代方案

## 工具强制执行

通过 `stylelint` 自动检查以下规则：
- `declaration-no-important: true` — 禁止 !important
- `selector-max-id: 0` — 禁止 ID 选择器
- `declaration-block-no-duplicate-properties: true` — 禁止重复属性

## Lint 命令

```bash
# 检查所有 CSS
npm run lint:css

# 自动修复可修复的问题
npm run lint:css -- --fix
```

## 渐进式收紧

初期将已有违规文件加入 `.stylelintignore`，完成重构后逐步移出。
