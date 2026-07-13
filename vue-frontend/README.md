# Saber-Translator 前端开发说明

> 最后更新：2026-07-11

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
  - `imageStore`：图片与翻译状态
  - `bubbleStore`：编辑模式气泡状态
  - `sessionStore`：会话与书架章节保存/加载
  - `settingsStore`：全局设置

### 翻译主链

- `src/composables/translation/core/`
  - 当前翻译主链核心
  - `pipeline.ts`：统一入口
  - `pipelineRegistry.ts`：步骤链真相源
  - `runtime.ts`：`TaskContext` / `PipelineRuntime`
  - `atomicSteps.ts`：原子步骤入口
  - `persistenceService.ts`：保存编排

- `src/composables/translation/parallel/`
  - 并行调度层

### 文字样式与编辑

- `src/composables/useTextStyleSync.ts`
  - 图片与侧边栏文字样式同步
  - 应用到全部
  - 设置变更后的重渲染

- `src/composables/useEditRender.ts`
  - 编辑模式下的重渲染辅助

---

## 4. 当前前端架构要点

### 4.1 翻译执行架构

当前前端不是“每种翻译模式各写一套流程”，而是：

```text
useTranslationPipeline
  -> usePipeline
  -> SequentialPipeline / ParallelPipeline
  -> atomic steps
  -> taskProjector / persistenceService
```

如果你要改翻译流程，优先阅读：

- [docs/parallel-mode-development-guide.md](../docs/parallel-mode-development-guide.md)
- [docs/save-load-development-guide.md](../docs/save-load-development-guide.md)

### 4.2 文字样式同步

图片与侧边栏样式同步逻辑已经集中在：

- [src/composables/useTextStyleSync.ts](src/composables/useTextStyleSync.ts)

不要再把这部分逻辑直接塞回 `TranslateView.vue`。

### 4.3 书架模式保存

当前书架模式保存主链已经统一到：

- `saveStep.ts`
- `persistenceService.ts`
- `sessionStore.ts`

不要在统一主链之外新增整批前端保存 helper。

---

## 5. 开发文档入口

推荐先看：

- [docs/README.md](../docs/README.md)
- [docs/text-settings-development-guide.md](../docs/text-settings-development-guide.md)
- [docs/parallel-mode-development-guide.md](../docs/parallel-mode-development-guide.md)
- [docs/OpenAI-Compatible主链开发手册.md](../docs/OpenAI-Compatible主链开发手册.md)

---

## 6. 代码风格

前端样式规范见：

- [CODING_STYLE.md](CODING_STYLE.md)

架构维护策略见：

- [docs/frontend-architecture.md](docs/frontend-architecture.md)
- [docs/ui-maintenance-decisions.md](docs/ui-maintenance-decisions.md)
