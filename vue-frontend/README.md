# Saber-Translator 前端开发说明

> 最后更新：2026-07-29

这是 Saber-Translator 的 Vue 3 + TypeScript + Vite 前端工程说明。

---

## 1. 技术栈

- Vue 3
- TypeScript
- Vite
- Pinia
- Vitest

---

## 2. 常用命令

在 `vue-frontend/` 目录下执行：

```bash
npm install
npm run dev
npm run build
npm test
npm run lint:ui
npm run lint:css
npm run lint
npm run typecheck
```

---

## 3. 目录重点

### 页面与组件

- `src/views/`
  - 主要页面入口，如书架页、翻译页、阅读页、Insight 页、角色工坊页

- `src/components/`
  - 页面组件与通用组件

### 状态管理

- `src/stores/`
  - 页面、任务、编辑和设置状态均是后端事实的前端投影
  - 允许保存主题、面板尺寸等纯 UI 偏好
  - 禁止以 localStorage/Pinia 作为书籍、页面、任务或分析结果的事实源

### 后端 API

- `src/api/v2/`：按内容、任务、设置、Insight、Studio 和插件领域组织的 v2 客户端
- `src/api/generated/v2.ts`：由 `openapi/v2.yaml` 生成的契约类型
- 页面使用 REST 获取快照，用 SSE 接收任务/operation 事件；断线后按事件序号补发

### 文字样式与编辑

- `src/composables/useTextStyleSync.ts`
  - 图片与侧边栏文字样式同步
  - 应用到全部
  - 设置变更后的重渲染

- `src/composables/useEditRender.ts`
  - 编辑模式下的重渲染辅助

---

## 4. 当前前端架构要点

### 4.1 浏览器职责边界

浏览器只负责交互和显示：

- 不在浏览器执行跨页翻译、分析、PDF 解析、插件或导出循环。
- 用户命令创建后端 job/operation，前端投影状态并展示进度。
- 列表、侧栏和页码选择器只请求缩略图并懒加载。
- 主图/编辑图只加载当前页；阅读器只加载可见窗口并及时释放离屏资源。
- 浏览器关闭或崩溃不会终止持久任务。

### 4.2 文字样式同步

图片与侧边栏样式同步逻辑已经集中在：

- [src/composables/useTextStyleSync.ts](src/composables/useTextStyleSync.ts)

不要再把这部分逻辑直接塞回 `TranslateView.vue`。

### 4.3 数据写入

- 图片导入后立即成为后端页面与不可变资产。
- 编辑使用 revision/`If-Match` 乐观并发，不提供手动保存和可选自动保存。
- 不得重新引入 Base64 图片事实、客户端绝对路径或浏览器生成的业务 ID。

---

## 5. 开发文档入口

推荐先看：

- [后端优先架构方案](../docs/refactor/backend-first-architecture-plan.md)
- [架构决策记录](../docs/refactor/adr/)
- [OpenAPI v2](../openapi/v2.yaml)

---

## 6. 代码风格

前端样式规范见：

- [CODING_STYLE.md](CODING_STYLE.md)

架构维护策略见：

- [docs/frontend-architecture.md](docs/frontend-architecture.md)
- [docs/ui-maintenance-decisions.md](docs/ui-maintenance-decisions.md)
