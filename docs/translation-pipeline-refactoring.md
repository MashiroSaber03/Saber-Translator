# 翻译管线架构重构说明

## 概述

本次重构将顺序翻译管线改为与并行管线**完全一致**的原子步骤设计。

## 核心设计：7 个原子步骤

| 步骤 | 说明 | API |
|------|------|-----|
| `detection` | 气泡检测 | `/api/parallel/detect` |
| `ocr` | 文字识别 | `/api/parallel/ocr` |
| `color` | 颜色提取 | `/api/parallel/color` |
| `translate` | 普通翻译 | `/api/parallel/translate` |
| `aiTranslate` | AI翻译（高质量翻译 & 校对共用） | `/api/hq_translate_batch` |
| `inpaint` | 背景修复 | `/api/parallel/inpaint` |
| `render` | 渲染译文 | `/api/parallel/render` |

## 步骤链配置

```typescript
STEP_CHAIN_CONFIGS = {
    standard:    ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
    hq:          ['detection', 'ocr', 'color', 'aiTranslate', 'inpaint', 'render'],
    proofread:   ['aiTranslate', 'render'],
    removeText:  ['detection', 'inpaint', 'render']
}
```

## 模式流程图

```
标准翻译:     检测 → OCR → 颜色 → 翻译 → 修复 → 渲染
高质量翻译:   检测 → OCR → 颜色 → AI翻译 → 修复 → 渲染
AI校对:       AI翻译(校对模式) → 渲染
消除文字:     检测 → 修复
```

## 最终文件结构

```
composables/translation/
├── core/
│   ├── index.ts              # 导出（18行）
│   ├── pipeline.ts           # 统一入口（152行）
│   ├── SequentialPipeline.ts # 顺序管线（803行）
│   ├── progressManager.ts    # 进度管理
│   └── types.ts              # 类型定义（91行）
├── parallel/                 # 并行管线
│   ├── ParallelPipeline.ts
│   ├── types.ts
│   └── pools/
├── modes/                    # 模式配置
│   ├── index.ts
│   ├── standardMode.ts      （18行）
│   ├── hqMode.ts            （28行）
│   ├── proofreadMode.ts     （28行）
│   └── removeTextMode.ts    （17行）
└── index.ts                  # 模块入口（31行）
```

## 已删除的文件

| 文件/目录 | 代码量 | 说明 |
|-----------|--------|------|
| `steps/` 目录 | ~500 行 | 完全删除 |
| `utils.ts` | 231 行 | 完全删除（废弃函数） |

## 已移除的类型

- `StepType`
- `TranslateMethod`
- `PrepareStepOptions`
- `TranslateStepOptions`
- `RenderStepOptions`
- `StepConfig`
- `ImageExecutionContext`
- `BatchExecutionContext`
- `PrepareStepResult`
- `TranslateStepResult`
- `RenderStepResult`
- `StepResult`
- `StepOptions`
- `StepExecutor`
- `ExistingBubbleData`
- `TranslationOptions`
- `TranslationJsonData`（重复定义）

## 代码量对比

| 项目 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| `types.ts` | 304 行 | 91 行 | -70% |
| `modes/*.ts` | ~150 行 | ~70 行 | -53% |
| `steps/` 目录 | ~500 行 | 0 行 | 删除 |
| `utils.ts` | 301 行 | 0 行 | 删除 |
| **总计删除** | | **~1000+ 行** | |
