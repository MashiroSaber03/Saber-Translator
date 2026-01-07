# 并行翻译模式开发手册

本手册介绍并行翻译模式的架构设计和开发指南，帮助开发者理解、维护和扩展并行翻译功能。

---

## 目录

1. [架构概述](#1-架构概述)
2. [核心组件](#2-核心组件)
3. [池子链配置](#3-池子链配置)
4. [开发指南](#4-开发指南)
5. [API参考](#5-api参考)
6. [常见问题](#6-常见问题)
7. [使用并行模式](#7-使用并行模式)
8. [注意事项](#8-注意事项)

---

## 1. 架构概述

### 1.1 设计理念

并行模式采用**池子链（Pool Chain）**架构，将翻译流程拆分为多个独立的处理池，每个池子负责一个特定步骤。任务在池子间流动，实现流水线并行处理。

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Detection│ → │   OCR   │ → │  Color  │ → │Translate│ → │ Inpaint │ → │ Render  │
│   Pool   │   │  Pool   │   │  Pool   │   │  Pool   │   │  Pool   │   │  Pool   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### 1.2 核心特性

- **配置驱动**：通过配置定义不同模式的池子链
- **核心复用**：每个池子只调用现有的核心函数，不重复实现逻辑
- **资源控制**：通过`DeepLearningLock`控制GPU/CPU密集型任务的并发数
- **响应式进度**：实时追踪每个池子的处理状态

### 1.3 文件结构

```
vue-frontend/src/composables/translation/parallel/
├── index.ts                    # 模块导出入口
├── types.ts                    # 类型定义
├── DeepLearningLock.ts         # 深度学习资源锁
├── TaskPool.ts                 # 池子基类
├── ParallelPipeline.ts         # 主控制器 + 池子链配置
├── ParallelProgressTracker.ts  # 进度追踪器
├── ResultCollector.ts          # 结果收集器
├── useParallelTranslation.ts   # Vue Composable入口
└── pools/                      # 具体池子实现
    ├── index.ts
    ├── DetectionPool.ts
    ├── OcrPool.ts
    ├── ColorPool.ts
    ├── TranslatePool.ts
    ├── InpaintPool.ts
    └── RenderPool.ts

src/app/api/translation/
└── parallel_routes.py          # 后端并行API
```

---

## 2. 核心组件

### 2.1 TaskPool（池子基类）

所有池子继承自`TaskPool`，提供通用的任务排队、处理、传递逻辑。

```typescript
abstract class TaskPool {
  protected queue: PipelineTask[] = []           // 任务队列
  protected nextPool: TaskPool | null            // 下一个池子
  protected lock: DeepLearningLock | null        // 资源锁（可选）
  protected name: string                         // 池子名称
  protected icon: string                         // 池子图标
  protected progressTracker: ParallelProgressTracker
  
  constructor(
    name: string,
    icon: string,
    nextPool: TaskPool | null,
    lock: DeepLearningLock | null,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  )
  
  // 子类必须实现
  protected abstract process(task: PipelineTask): Promise<PipelineTask>
  
  // 通用方法
  enqueue(task: PipelineTask): void         // 入队
  enqueueBatch(tasks: PipelineTask[]): void // 批量入队
  setNextPool(pool: TaskPool | null): void  // 设置下一个池子
  getName(): string                         // 获取池子名称
  cancel(): void                            // 取消
  reset(): void                             // 重置
}
```

### 2.2 DeepLearningLock（资源锁）

控制GPU/CPU密集型任务的并发数，防止资源竞争。

```typescript
class DeepLearningLock {
  constructor(maxCount: number = 1)
  
  acquire(poolName: string): Promise<void>                    // 获取锁
  release(poolName: string): void                             // 释放锁
  withLock<T>(poolName: string, fn: () => Promise<T>): Promise<T>  // 自动管理锁
  setSize(size: number): void                                 // 动态调整并发数
  getSize(): number                                           // 获取当前大小
  isWaiting(poolName: string): boolean                        // 检查是否在等待
  reset(): void                                               // 重置
}
```

### 2.3 ParallelProgressTracker（进度追踪器）

响应式追踪各池子的处理状态。

```typescript
class ParallelProgressTracker {
  readonly progress: ParallelProgress  // 响应式进度对象
  
  init(totalPages: number): void                              // 初始化
  updatePool(poolName: string, update: PoolProgressUpdate): void  // 更新池子状态
  incrementCompleted(): void                                  // 增加完成数
  incrementFailed(): void                                     // 增加失败数
  reset(): void                                               // 重置
}
```

### 2.4 ResultCollector（结果收集器）

收集所有完成的任务，提供等待机制。

```typescript
class ResultCollector {
  init(totalExpected: number): void                           // 初始化
  add(task: PipelineTask): void                               // 添加结果
  waitForAll(totalExpected: number): Promise<{success: number, failed: number}>
  getAll(): PipelineTask[]                                    // 获取所有结果
  getSuccessful(): PipelineTask[]                             // 获取成功结果
  getFailed(): PipelineTask[]                                 // 获取失败结果
  reset(): void                                               // 重置
}
```

---

## 3. 池子链配置

### 3.1 配置位置

`ParallelPipeline.ts` 中的 `POOL_CHAIN_CONFIGS`：

```typescript
export const POOL_CHAIN_CONFIGS: Record<ParallelTranslationMode, string[]> = {
  standard: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  hq: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  proofread: ['translate', 'render'],
  removeText: ['detection', 'inpaint', 'render']  // 跳过OCR/颜色/翻译
}
```

### 3.2 配置说明

| 模式 | 说明 | 池子链 |
|------|------|--------|
| `standard` | 标准翻译 | 完整流程 |
| `hq` | 高质量翻译 | 完整流程，翻译池使用批量处理 |
| `proofread` | AI校对 | 跳过检测/OCR/颜色/修复，直接翻译+渲染 |
| `removeText` | 消除文字 | 仅检测+修复+更新UI，跳过OCR/颜色/翻译 |

### 3.3 池子名称映射

`ParallelPipeline.getPoolMap()` 定义了名称到实例的映射：

```typescript
private getPoolMap(): Record<string, TaskPool> {
  return {
    detection: this.detectionPool,
    ocr: this.ocrPool,
    color: this.colorPool,
    translate: this.translatePool,
    inpaint: this.inpaintPool,
    render: this.renderPool
  }
}
```

---

## 4. 开发指南

### 4.1 添加新的翻译模式

**步骤 1**：在 `types.ts` 中添加模式类型

```typescript
export type ParallelTranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText' | 'newMode'
```

**步骤 2**：在 `POOL_CHAIN_CONFIGS` 中添加配置

```typescript
export const POOL_CHAIN_CONFIGS = {
  // ...existing modes
  newMode: ['detection', 'translate', 'render']  // 自定义池子链
}
```

**步骤 3**：如果需要特殊处理，在 `TranslatePool.ts` 的 `setMode` 方法中添加逻辑

```typescript
setMode(mode: ParallelTranslationMode, totalTasks: number, nextPool: TaskPool | null) {
  this.mode = mode
  this.totalTasks = totalTasks
  this.nextPool = nextPool
  
  if (mode === 'newMode') {
    // 特殊初始化逻辑
  }
}
```

### 4.2 添加新的处理池

**步骤 1**：创建池子类文件 `pools/NewPool.ts`

```typescript
import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelNew } from '@/api/parallelTranslate'

export class NewPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    // 参数：名称、图标、下一个池子、资源锁、进度追踪器、完成回调
    super('新池子', '🆕', nextPool, lock, progressTracker, onTaskComplete)
  }
  
  protected async process(task: PipelineTask): Promise<PipelineTask> {
    // 1. 调用后端API
    const response = await parallelNew({
      image: this.extractBase64(task.imageData.originalDataURL),
      // ...其他参数
    })
    
    if (!response.success) {
      throw new Error(response.error || '处理失败')
    }
    
    // 2. 存储结果到task
    task.newResult = response
    
    // 3. 设置状态为processing（表示本池子处理完成，准备传递给下一个池子）
    task.status = 'processing'
    
    return task
  }
  
  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }
}
```

**步骤 2**：在 `pools/index.ts` 中导出

```typescript
export { NewPool } from './NewPool'
```

**步骤 3**：在 `ParallelPipeline.ts` 中实例化并添加到映射

```typescript
import { NewPool } from './pools'

export class ParallelPipeline {
  private newPool: NewPool
  
  constructor(config: ParallelConfig) {
    // ...existing code
    // 注意：如果新池子需要GPU，传入lock；否则传入null
    this.newPool = new NewPool(null, this.lock, this.progressTracker)
  }
  
  private getPoolMap(): Record<string, TaskPool> {
    return {
      // ...existing pools
      new: this.newPool
    }
  }
  
  // 还需要在cancel()和reset()方法中添加对新池子的处理
  cancel(): void {
    // ...existing code
    this.newPool.cancel()
  }
  
  private reset(): void {
    // ...existing code
    this.newPool.reset()
  }
}
```

**步骤 4**：添加后端API `parallel_routes.py`

```python
@parallel_bp.route('/parallel/new', methods=['POST'])
def parallel_new():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        # 调用核心函数
        result = existing_core_function(...)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
```

**步骤 5**：添加前端API函数 `api/parallelTranslate.ts`

```typescript
import { apiClient } from './client'

export interface ParallelNewParams {
  image: string
  // ...其他参数
}

export interface ParallelNewResponse {
  success: boolean
  result?: any
  error?: string
}

export async function parallelNew(params: ParallelNewParams): Promise<ParallelNewResponse> {
  return apiClient.post<ParallelNewResponse>('/api/parallel/new', params)
}
```

### 4.3 修改现有流程

#### 修改流程步骤顺序

只需修改 `POOL_CHAIN_CONFIGS`：

```typescript
// 例：让消除文字模式跳过OCR和颜色提取
removeText: ['detection', 'inpaint', 'render']
```

#### 修改步骤内部逻辑

直接修改对应的核心模块代码（如 `src/core/inpainting.py`），并行模式会自动使用新逻辑。

### 4.4 批量处理模式

`TranslatePool` 支持批量处理（HQ翻译、AI校对）：

```typescript
// 任务状态说明
type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'buffered'

// 'buffered' 状态表示任务正在等待凑齐批次
// TaskPool.tryProcessNext 会跳过 buffered 状态的任务，不自动传递
```

批量处理流程：
1. 任务进入 `TranslatePool`
2. 如果批次未凑够，设置 `task.status = 'buffered'`，任务暂存
3. 批次凑够后，批量调用API
4. 处理完成后，设置 `task.status = 'processing'`，手动调用 `nextPool.enqueue(task)` 传递到下一个池子

---

## 5. API参考

### 5.1 后端API列表

| API | 方法 | 说明 |
|-----|------|------|
| `/api/parallel/detect` | POST | 文字检测 |
| `/api/parallel/ocr` | POST | 文字识别 |
| `/api/parallel/color` | POST | 颜色提取 |
| `/api/parallel/translate` | POST | 文字翻译 |
| `/api/parallel/inpaint` | POST | 背景修复 |
| `/api/parallel/render` | POST | 文字渲染 |

### 5.2 任务数据结构

```typescript
interface PipelineTask {
  id: string                    // 唯一标识
  imageIndex: number            // 图片索引
  imageData: ImageData          // 图片数据
  status: TaskStatus            // 任务状态
  error?: string                // 错误信息
  
  // 各阶段结果
  detectionResult?: {
    bubbleCoords: number[][]    // [[x1, y1, x2, y2], ...]
    bubbleAngles: number[]
    bubblePolygons: number[][][]
    autoDirections: string[]
    rawMask?: string
    textlinesPerBubble?: any[]
  }
  ocrResult?: {
    originalTexts: string[]
    textlinesPerBubble?: any[]
  }
  colorResult?: {
    colors: Array<{
      textColor: string
      bgColor: string
      autoFgColor?: [number, number, number] | null
      autoBgColor?: [number, number, number] | null
    }>
  }
  translateResult?: {
    translatedTexts: string[]
    textboxTexts: string[]
  }
  inpaintResult?: {
    cleanImage: string          // Base64编码的干净背景图
  }
  renderResult?: {
    finalImage: string          // Base64编码的最终图片
    bubbleStates: BubbleState[]
  }
}
```

### 5.3 进度数据结构

```typescript
interface ParallelProgress {
  pools: PoolStatus[]
  totalCompleted: number
  totalFailed: number
  totalPages: number
  estimatedTimeRemaining: number
}

interface PoolStatus {
  name: string
  icon: string
  waiting: number
  processing: boolean
  completed: number
  currentPage?: number
  isWaitingLock: boolean
}
```

---

## 6. 常见问题

### Q1: 如何调试池子处理问题？

在池子的 `process` 方法中添加日志：

```typescript
console.log(`[${this.name}] 处理图片 ${task.imageIndex + 1}`)
console.log(`[${this.name}] 输入数据:`, task.detectionResult)
```

### Q2: 为什么任务没有传递到下一个池子？

检查：
1. 任务状态是否为 `processing`（`buffered`、`completed` 和 `failed` 不会自动传递）
2. `nextPool` 是否正确设置
3. 查看 `TaskPool.tryProcessNext` 中的条件判断

### Q3: 如何添加新的设置参数到池子？

1. 在 `useSettingsStore` 中添加设置
2. 在池子的 `process` 方法中从 `settingsStore` 获取
3. 传递给后端API
4. 后端API传递给核心函数

### Q4: 资源锁如何工作？

```typescript
// 需要GPU的操作使用锁（在TaskPool基类中自动处理）
// 如果需要手动使用：
await this.lock.withLock(this.name, async () => {
  // GPU密集型操作
  const result = await callGpuApi(...)
  return result
})
```

### Q5: 如何处理批量API？

参考 `TranslatePool` 的 `handleHqTranslate` 和 `handleProofread` 方法：
1. 收集任务到缓冲区
2. 检查是否凑够批次
3. 未凑够设置 `buffered` 状态
4. 凑够后批量调用API
5. 分发结果到各任务
6. 手动调用 `nextPool.enqueue(task)`

---

## 附录：核心函数映射

| 池子 | 后端API | 核心函数 |
|------|---------|----------|
| DetectionPool | `/parallel/detect` | `get_bubble_detection_result_with_auto_directions()` |
| OcrPool | `/parallel/ocr` | `recognize_text_in_bubbles()` |
| ColorPool | `/parallel/color` | `extract_bubble_colors()` |
| TranslatePool | `/parallel/translate` | `translate_text_list()` |
| InpaintPool | `/parallel/inpaint` | `inpaint_bubbles()` |
| RenderPool | `/parallel/render` | `render_bubbles_unified()` |

---

## 7. 使用并行模式

### 7.1 启用并行模式

在设置界面中启用并行翻译：
1. 打开 **更多设置** → **并行翻译**
2. 开启 **启用并行模式**
3. 设置 **深度学习锁大小**（控制GPU任务并发数，默认1）

### 7.2 选择翻译模式

并行模式会根据设置自动选择翻译模式：

| 条件 | 选择的模式 |
|------|----------|
| 启用AI校对 | `proofread` |
| 配置高质量翻译API | `hq` |
| 其他情况 | `standard` |

### 7.3 进度显示

并行模式会在翻译进度条中显示：
- 总体进度百分比
- 各池子的处理状态（等待/处理中/完成）
- 完成/失败数量统计

---

## 8. 注意事项

### 8.1 任务状态说明

| 状态 | 说明 | 是否传递到下一个池子 |
|------|------|---------------------|
| `pending` | 等待处理 | - |
| `processing` | 本池子处理完成 | ✅ 自动传递 |
| `completed` | 所有流程完成 | ❌ 不传递 |
| `failed` | 处理失败 | ❌ 不传递 |
| `buffered` | 等待批量处理 | ❌ 不传递 |

### 8.2 资源锁使用原则

- **需要GPU的池子**：DetectionPool、OcrPool（部分引擎）、InpaintPool（LAMA模型）
- **不需要GPU的池子**：ColorPool、TranslatePool、RenderPool

### 8.3 错误处理

池子中的错误会被自动捕获，任务状态设为`failed`，错误信息存储在`task.error`中。失败的任务会被`ResultCollector`收集，最终汇总到执行结果中。

---

*最后更新：2026-01-07*
