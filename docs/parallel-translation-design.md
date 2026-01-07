# 并行翻译模式 - 完整设计方案

## 一、概述

### 1.1 设计目标
将原有的串行翻译流程改造为流水线并行模式，显著提升批量翻译速度。

### 1.2 核心思想
- **6个处理池**：检测池、OCR池、颜色池、翻译池、修复池、渲染池
- **流水线并行**：池子之间并行工作，任务像水流一样从检测池流向渲染池
- **池内串行**：每个池子内部一次只处理一页，保证资源不冲突
- **深度学习锁**：检测/OCR/颜色/修复共用一把锁，避免GPU过载
- **以页为单位**：任务以图片页为最小单位流动

### 1.3 默认状态
并行模式**默认关闭**，用户可在"设置 → 更多"中启用。

---

## 二、可扩展的系统架构

### 1.4 扩展性设计原则

为了方便以后添加新的处理流程，采用**插件化池子注册机制**：

1. **池子接口统一**：所有池子继承自 `TaskPool` 基类
2. **链式连接**：池子通过 `nextPool` 属性连接，可动态修改
3. **注册中心**：`PoolRegistry` 管理所有池子的创建和连接
4. **模式配置化**：不同模式的池子链通过配置定义

```typescript
// 池子注册中心 - 方便扩展新池子
class PoolRegistry {
  private pools: Map<string, TaskPool> = new Map();
  
  // 注册池子
  register(name: string, pool: TaskPool): void {
    this.pools.set(name, pool);
  }
  
  // 获取池子
  get(name: string): TaskPool | undefined {
    return this.pools.get(name);
  }
  
  // 根据模式配置连接池子链
  setupChain(config: PoolChainConfig): void {
    for (let i = 0; i < config.pools.length - 1; i++) {
      const current = this.pools.get(config.pools[i]);
      const next = this.pools.get(config.pools[i + 1]);
      if (current && next) {
        current.setNextPool(next);
      }
    }
  }
}

// 池子链配置示例
const poolChainConfigs: Record<TranslationMode, PoolChainConfig> = {
  standard: {
    pools: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render']
  },
  hq: {
    pools: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render']
  },
  proofread: {
    pools: ['translate', 'render']  // AI校对跳过前面的池子
  },
  removeText: {
    pools: ['detection', 'ocr', 'color', 'translate', 'inpaint']  // 无渲染
  }
};

// 添加新池子示例（如果以后需要添加“样式优化”池）
class StyleOptimizePool extends TaskPool {
  protected async process(task: PipelineTask): Promise<PipelineTask> {
    // 新池子的处理逻辑
    return task;
  }
}

// 注册新池子并插入到链中
registry.register('styleOptimize', new StyleOptimizePool(...));
const newConfig = {
  pools: ['detection', 'ocr', 'color', 'translate', 'styleOptimize', 'inpaint', 'render']
};
```

### 1.5 实时更新机制

渲染池每完成一张图片，立即更新到界面：

```typescript
// 渲染池完成回调
const onRenderComplete = (task: PipelineTask) => {
  // 1. 更新 imageStore
  imageStore.updateImageByIndex(task.imageIndex, {
    translatedDataURL: `data:image/png;base64,${task.renderResult!.finalImage}`,
    bubbleStates: task.renderResult!.bubbleStates,
    translationStatus: 'completed',
    hasUnsavedChanges: true
  });
  
  // 2. 如果是当前图片，同步更新 bubbleStore
  if (task.imageIndex === imageStore.currentImageIndex) {
    bubbleStore.setBubbles(task.renderResult!.bubbleStates);
  }
  
  // 3. 更新进度
  progressTracker.incrementCompleted();
};
```

### 2.1 完整流水线架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ParallelTranslationPipeline                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                      🔒 DeepLearningLock                             │    │
│   │              (检测/OCR/颜色/修复 四个池子共用)                         │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐│
│   │ 检测池  │──▶│ OCR池  │──▶│ 颜色池 │──▶│ 翻译池 │──▶│ 修复池 │──▶│ 渲染池 ││
│   │🔒深度锁 │   │🔒深度锁│   │🔒深度锁│   │  无锁  │   │🔒深度锁│   │  无锁  ││
│   └────────┘   └────────┘   └────────┘   └────────┘   └────────┘   └────────┘│
│        │            │            │            │            │            │     │
│        ▼            ▼            ▼            ▼            ▼            ▼     │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐│
│   │串行处理│   │串行处理│   │串行处理│   │模式分发│   │串行处理│   │串行处理││
│   │ 1页/次 │   │ 1页/次 │   │ 1页/次 │   │        │   │ 1页/次 │   │ 1页/次 ││
│   └────────┘   └────────┘   └────────┘   └────────┘   └────────┘   └────────┘│
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 流水线工作示意（以5页图片为例）

```
时间 →
     T1      T2      T3      T4      T5      T6      T7      T8      T9
     ─────────────────────────────────────────────────────────────────────
检测  页1     页2     页3     页4     页5
OCR          页1     页2     页3     页4     页5
颜色                 页1     页2     页3     页4     页5
翻译                         页1     页2     页3     页4     页5
修复                                 页1     页2     页3     页4     页5
渲染                                         页1     页2     页3     页4     页5
     ─────────────────────────────────────────────────────────────────────
                                                              ↑完成页1  ↑完成页5
```

**说明**：由于深度学习锁的存在，检测/OCR/颜色/修复不能同时处理不同页，但翻译和渲染可以与其他阶段并行。

---

## 三、各翻译模式的并行流程

### 3.1 标准翻译（翻译所有图片）

**完整流水线**：检测 → OCR → 颜色 → 翻译 → 修复 → 渲染

```
┌─────────────────────────────────────────────────────────────────┐
│ 标准翻译模式 - 完整6池流水线                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  检测池 ─▶ OCR池 ─▶ 颜色池 ─▶ 翻译池 ─▶ 修复池 ─▶ 渲染池 ─▶ 完成 │
│    🔒       🔒       🔒      (逐页)     🔒                       │
│                                                                  │
│  翻译池行为：逐页调用翻译API                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 高质量翻译

**完整流水线**：检测 → OCR → 颜色 → 翻译(批量+图片) → 修复 → 渲染

```
┌─────────────────────────────────────────────────────────────────┐
│ 高质量翻译模式 - 翻译池批量处理                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  检测池 ─▶ OCR池 ─▶ 颜色池 ─▶ 翻译池 ─▶ 修复池 ─▶ 渲染池 ─▶ 完成 │
│    🔒       🔒       🔒      (批量)     🔒                       │
│                                │                                 │
│                     ┌──────────┴──────────┐                      │
│                     │ 翻译池行为：         │                      │
│                     │ 1. 积累任务到批次大小 │                      │
│                     │ 2. 收集JSON数据      │                      │
│                     │ 3. 收集图片Base64    │                      │
│                     │ 4. 构建提示词        │                      │
│                     │ 5. 批量调用多模态AI  │                      │
│                     │ 6. 解析结果填充任务  │                      │
│                     └─────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**翻译池批量处理逻辑**：
```
任务到达 → 加入缓冲区 → 检查是否凑够批次
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
         未凑够，继续等待              凑够了或是最后一批
                                              │
                                              ▼
                              ┌──────────────────────────┐
                              │ 1. 收集批次内所有原文JSON │
                              │ 2. 收集批次内所有图片    │
                              │ 3. 构建多模态消息        │
                              │ 4. 调用 hqTranslateBatch │
                              │ 5. 解析返回的译文       │
                              │ 6. 填充到各任务         │
                              │ 7. 批量传给修复池       │
                              └──────────────────────────┘
```

### 3.3 AI校对

**简化流水线**：翻译(批量校对) → 渲染

AI校对是对**已翻译**的图片进行二次翻译，因此**跳过**检测、OCR、颜色、修复阶段。

```
┌─────────────────────────────────────────────────────────────────┐
│ AI校对模式 - 仅翻译+渲染                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  从已翻译图片提取数据 ─▶ 翻译池(校对) ─▶ 渲染池 ─▶ 完成          │
│                            (批量)                                │
│                              │                                   │
│               ┌──────────────┴──────────────┐                    │
│               │ 翻译池行为（校对模式）：      │                    │
│               │ 1. 积累任务到批次大小        │                    │
│               │ 2. 从已有bubbleStates提取数据│                    │
│               │ 3. 收集图片（优先翻译后图片）│                    │
│               │ 4. 构建校对提示词           │                    │
│               │ 5. 批量调用AI校对           │                    │
│               │ 6. 支持多轮校对             │                    │
│               │ 7. 填充校对结果到任务       │                    │
│               └─────────────────────────────┘                    │
│                                                                  │
│  ⚠️ 前提：图片必须已完成翻译                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 消除所有图片文字

**简化流水线**：检测 → OCR → 颜色 → 修复（跳过翻译和渲染）

```
┌─────────────────────────────────────────────────────────────────┐
│ 消除文字模式 - 无翻译无渲染                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  检测池 ─▶ OCR池 ─▶ 颜色池 ─▶ 翻译池 ─▶ 修复池 ─▶ 完成          │
│    🔒       🔒       🔒      (跳过)     🔒                       │
│                                                                  │
│  翻译池行为：直接传递给修复池，不翻译                             │
│  渲染池：跳过（修复后即为最终结果）                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 模式汇总表

| 模式 | 检测池 | OCR池 | 颜色池 | 翻译池 | 修复池 | 渲染池 |
|------|--------|-------|--------|--------|--------|--------|
| **翻译所有图片** | ✅ | ✅ | ✅ | 逐页翻译 | ✅ | ✅ |
| **高质量翻译** | ✅ | ✅ | ✅ | 批量+图片+提示词 | ✅ | ✅ |
| **AI校对** | ❌ | ❌ | ❌ | 批量校对+图片+提示词 | ❌ | ✅ |
| **消除所有文字** | ✅ | ✅ | ✅ | 跳过 | ✅ | ❌ |

---

## 四、核心数据结构

### 4.1 任务对象 (PipelineTask)

```typescript
interface PipelineTask {
  id: string;                      // 唯一ID (如 "task-0", "task-1")
  imageIndex: number;              // 图片索引（用于排序保序）
  imageData: ImageData;            // 图片数据引用
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  
  // 各阶段结果（逐步填充）
  detectionResult?: {
    bubbleCoords: number[][];      // [[x1,y1,x2,y2], ...]
    bubbleAngles: number[];        // 旋转角度
    bubblePolygons: number[][][];  // 多边形坐标
    autoDirections: string[];      // 自动排版方向 ['vertical', 'horizontal', ...]
    rawMask?: string;              // Base64 文字掩膜
  };
  
  ocrResult?: {
    originalTexts: string[];       // 原文列表
    textlinesPerBubble: any[];     // 每个气泡的文本行信息
  };
  
  colorResult?: {
    colors: Array<{
      textColor: string;           // 文字颜色
      bgColor: string;             // 背景颜色
    }>;
  };
  
  translateResult?: {
    translatedTexts: string[];     // 译文列表
    textboxTexts: string[];        // 文本框文本
  };
  
  inpaintResult?: {
    cleanImage: string;            // Base64 干净背景图
  };
  
  renderResult?: {
    finalImage: string;            // Base64 最终图片
    bubbleStates: BubbleState[];   // 气泡状态列表
  };
}
```

### 4.2 并行配置

```typescript
interface ParallelConfig {
  enabled: boolean;  // 是否启用并行模式，默认 false
}

// 在 Settings 中的位置
interface Settings {
  // ... 其他设置
  parallel: ParallelConfig;
}
```

### 4.3 池子状态

```typescript
interface PoolStatus {
  name: string;           // 池子名称
  waiting: number;        // 等待队列中的任务数
  processing: boolean;    // 是否正在处理
  currentPage?: number;   // 当前处理的页码
  completed: number;      // 已完成数
  isWaitingLock: boolean; // 是否在等待深度学习锁
}
```

---

## 五、核心类设计

### 5.1 深度学习锁 (DeepLearningLock)

```typescript
/**
 * 深度学习模型互斥锁
 * 确保检测、OCR、颜色提取、背景修复不会同时执行
 * 避免 GPU/CPU 资源竞争
 */
class DeepLearningLock {
  private locked = false;
  private waitQueue: Array<{ resolve: () => void; poolName: string }> = [];
  private currentHolder?: string;  // 当前持有锁的池子名称
  
  /**
   * 获取锁
   * @param poolName 请求锁的池子名称（用于调试和状态显示）
   */
  async acquire(poolName: string): Promise<void> {
    if (!this.locked) {
      this.locked = true;
      this.currentHolder = poolName;
      return;
    }
    
    // 排队等待
    return new Promise(resolve => {
      this.waitQueue.push({ resolve, poolName });
    });
  }
  
  /**
   * 释放锁
   */
  release(): void {
    if (this.waitQueue.length > 0) {
      const next = this.waitQueue.shift()!;
      this.currentHolder = next.poolName;
      next.resolve();
    } else {
      this.locked = false;
      this.currentHolder = undefined;
    }
  }
  
  /**
   * 带锁执行（自动获取和释放）
   */
  async withLock<T>(poolName: string, fn: () => Promise<T>): Promise<T> {
    await this.acquire(poolName);
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
  
  /**
   * 获取锁状态
   */
  getStatus(): { isLocked: boolean; holder?: string; waitingCount: number } {
    return {
      isLocked: this.locked,
      holder: this.currentHolder,
      waitingCount: this.waitQueue.length
    };
  }
}
```

### 5.2 通用任务池 (TaskPool)

```typescript
/**
 * 通用任务池基类
 * - 无容量限制，任务自然排队
 * - 串行处理（一次一页）
 * - 支持可选的深度学习锁
 */
abstract class TaskPool {
  protected queue: PipelineTask[] = [];
  protected currentTask: PipelineTask | null = null;
  protected isRunning = false;
  protected isCancelled = false;
  protected completedCount = 0;
  
  constructor(
    protected name: string,
    protected nextPool: TaskPool | null,
    protected lock: DeepLearningLock | null,
    protected progressTracker: ParallelProgressTracker,
    protected onTaskComplete?: (task: PipelineTask) => void
  ) {}
  
  /**
   * 添加任务到队列
   */
  enqueue(task: PipelineTask): void {
    if (this.isCancelled) return;
    this.queue.push(task);
    this.progressTracker.updatePool(this.name, { waiting: this.queue.length });
    this.tryProcessNext();
  }
  
  /**
   * 尝试处理下一个任务
   */
  private async tryProcessNext(): Promise<void> {
    if (this.isRunning || this.isCancelled || this.queue.length === 0) return;
    
    this.isRunning = true;
    this.currentTask = this.queue.shift()!;
    
    this.progressTracker.updatePool(this.name, {
      waiting: this.queue.length,
      isProcessing: true,
      currentPage: this.currentTask.imageIndex + 1,
      isWaitingLock: false
    });
    
    try {
      let result: PipelineTask;
      
      if (this.lock) {
        // 需要深度学习锁
        this.progressTracker.updatePool(this.name, { isWaitingLock: true });
        result = await this.lock.withLock(this.name, () => this.process(this.currentTask!));
      } else {
        result = await this.process(this.currentTask);
      }
      
      this.completedCount++;
      this.progressTracker.updatePool(this.name, { completed: this.completedCount });
      
      // 传递给下一个池子
      if (this.nextPool && result.status !== 'failed') {
        this.nextPool.enqueue(result);
      }
      
      this.onTaskComplete?.(result);
      
    } catch (error) {
      this.currentTask.status = 'failed';
      this.currentTask.error = (error as Error).message;
      this.onTaskComplete?.(this.currentTask);
    } finally {
      this.currentTask = null;
      this.isRunning = false;
      this.progressTracker.updatePool(this.name, { isProcessing: false, currentPage: undefined });
      this.tryProcessNext();
    }
  }
  
  /**
   * 子类实现具体处理逻辑
   */
  protected abstract process(task: PipelineTask): Promise<PipelineTask>;
  
  /**
   * 获取池子状态
   */
  getStatus(): PoolStatus {
    return {
      name: this.name,
      waiting: this.queue.length,
      processing: this.isRunning,
      currentPage: this.currentTask?.imageIndex,
      completed: this.completedCount,
      isWaitingLock: false  // 由 progressTracker 单独追踪
    };
  }
  
  /**
   * 取消所有任务
   */
  cancel(): void {
    this.isCancelled = true;
    this.queue = [];
  }
  
  /**
   * 重置池子
   */
  reset(): void {
    this.isCancelled = false;
    this.queue = [];
    this.currentTask = null;
    this.isRunning = false;
    this.completedCount = 0;
  }
}
```

### 5.3 翻译池 (TranslatePool) - 核心

```typescript
import { hqTranslateBatch } from '@/api/translate'
import type { HqTranslateParams } from '@/api/translate'
import type { TranslationJsonData } from '../core/types'

/**
 * 翻译池 - 根据模式不同采用不同处理策略
 * 
 * 注意：API 调用使用项目中已有的 hqTranslateBatch 函数
 * 参考：vue-frontend/src/composables/translation/steps/multimodalTranslate.ts
 */
class TranslatePool extends TaskPool {
  private mode: TranslationMode = 'standard';
  private batchBuffer: PipelineTask[] = [];
  private totalTasks = 0;
  private processedCount = 0;
  
  constructor(
    nextPool: InpaintPool | RenderPool,  // 根据模式可能直接连接渲染池
    lock: null,  // 翻译池不需要深度学习锁
    progressTracker: ParallelProgressTracker,
    private settingsStore: ReturnType<typeof useSettingsStore>,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('翻译', nextPool, lock, progressTracker, onTaskComplete);
  }
  
  /**
   * 设置翻译模式和下一个池子
   */
  setMode(mode: TranslationMode, totalTasks: number, nextPool: TaskPool | null): void {
    this.mode = mode;
    this.totalTasks = totalTasks;
    this.batchBuffer = [];
    this.processedCount = 0;
    this.nextPool = nextPool;  // AI校对模式直接连接渲染池
  }
  
  protected async process(task: PipelineTask): Promise<PipelineTask> {
    switch (this.mode) {
      case 'standard':
        return this.handleStandardTranslate(task);
      case 'hq':
        return this.handleHqTranslate(task);
      case 'proofread':
        return this.handleProofread(task);
      case 'removeText':
        return this.handleRemoveTextOnly(task);
      default:
        return task;
    }
  }
  
  // ==================== 普通翻译 ====================
  private async handleStandardTranslate(task: PipelineTask): Promise<PipelineTask> {
    const { translation, targetLanguage, sourceLanguage, translatePrompt } = this.settingsStore.settings;
    
    // 调用后端 /api/parallel/translate API
    const response = await parallelTranslateApi({
      original_texts: task.ocrResult!.originalTexts,
      target_language: targetLanguage,
      source_language: sourceLanguage,
      model_provider: translation.provider,
      model_name: translation.modelName,
      api_key: translation.apiKey,
      custom_base_url: translation.customBaseUrl,
      prompt_content: translatePrompt,
      rpm_limit: translation.rpmLimit,
      max_retries: translation.maxRetries,
    });
    
    task.translateResult = {
      translatedTexts: response.translated_texts,
      textboxTexts: response.textbox_texts || []
    };
    
    return task;
  }
  
  // ==================== 高质量翻译 ====================
  private async handleHqTranslate(task: PipelineTask): Promise<PipelineTask> {
    this.batchBuffer.push(task);
    this.processedCount++;
    
    const { hqTranslation } = this.settingsStore.settings;
    const batchSize = hqTranslation.batchSize || 3;
    const isLastBatch = this.processedCount >= this.totalTasks;
    const batchReady = this.batchBuffer.length >= batchSize || isLastBatch;
    
    if (!batchReady) {
      // 还没凑够批次，任务保持在缓冲区中，不传递给下一个池子
      // 返回 null 表示此任务暂不传递
      return task;
    }
    
    // 凑够批次，开始批量处理
    const batch = [...this.batchBuffer];
    this.batchBuffer = [];
    
    // 1. 收集 JSON 数据（参考 multimodalTranslate.ts 的 exportTextsToJson）
    const jsonData: TranslationJsonData[] = batch.map(t => ({
      imageIndex: t.imageIndex,
      bubbles: t.ocrResult!.originalTexts.map((text, idx) => ({
        bubbleIndex: idx,
        original: text,
        translated: '',
        textDirection: t.detectionResult?.autoDirections[idx] || 'vertical'
      }))
    }));
    
    // 2. 收集图片 Base64
    const imageBase64Array = batch.map(t => 
      this.extractBase64(t.imageData.originalDataURL)
    );
    
    // 3. 构建消息（参考 multimodalTranslate.ts 的 callMultimodalAI）
    const jsonString = JSON.stringify(jsonData, null, 2);
    type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
    const userContent: MessageContent[] = [
      {
        type: 'text',
        text: hqTranslation.prompt + '\n\n以下是JSON数据:\n```json\n' + jsonString + '\n```'
      }
    ];
    for (const imgBase64 of imageBase64Array) {
      userContent.push({
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${imgBase64}` }
      });
    }
    
    const messages: HqTranslateParams['messages'] = [
      { role: 'system', content: '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。' },
      { role: 'user', content: userContent }
    ];
    
    // 4. 调用多模态 AI API（使用项目已有的 hqTranslateBatch）
    const response = await hqTranslateBatch({
      provider: hqTranslation.provider,
      api_key: hqTranslation.apiKey,
      model_name: hqTranslation.modelName,
      custom_base_url: hqTranslation.customBaseUrl,
      messages: messages,
      low_reasoning: hqTranslation.lowReasoning,
      force_json_output: hqTranslation.forceJsonOutput,
      no_thinking_method: hqTranslation.noThinkingMethod,
      use_stream: hqTranslation.useStream
    });
    
    // 5. 解析结果（参考 multimodalTranslate.ts 的解析逻辑）
    const translatedData = this.parseHqResponse(response, hqTranslation.forceJsonOutput);
    
    // 6. 填充结果到各任务，并批量传递给下一个池子
    for (const t of batch) {
      const taskData = translatedData?.find(d => d.imageIndex === t.imageIndex);
      if (taskData) {
        t.translateResult = {
          translatedTexts: taskData.bubbles.map(b => b.translated),
          textboxTexts: []
        };
      }
      t.status = 'processing';
      // 批量传给修复池
      if (this.nextPool) {
        this.nextPool.enqueue(t);
      }
    }
    
    return task;  // 返回最后一个任务
  }
  
  // ==================== AI 校对 ====================
  private async handleProofread(task: PipelineTask): Promise<PipelineTask> {
    this.batchBuffer.push(task);
    this.processedCount++;
    
    const { proofreading, useTextboxPrompt } = this.settingsStore.settings;
    const batchSize = proofreading.rounds[0]?.batchSize || 3;
    const isLastBatch = this.processedCount >= this.totalTasks;
    const batchReady = this.batchBuffer.length >= batchSize || isLastBatch;
    
    if (!batchReady) {
      return task;
    }
    
    const batch = [...this.batchBuffer];
    this.batchBuffer = [];
    
    // 1. 收集 JSON 数据（参考 proofreadTranslate.ts 的 exportProofreadingTextsToJson）
    // 校对模式包含已有译文
    const jsonData: TranslationJsonData[] = batch.map(t => ({
      imageIndex: t.imageIndex,
      bubbles: (t.imageData.bubbleStates || []).map((state, idx) => ({
        bubbleIndex: idx,
        original: state.originalText || '',
        // 校对模式：导出已翻译的文本
        translated: useTextboxPrompt 
          ? (state.textboxText || state.translatedText || '')
          : (state.translatedText || ''),
        textDirection: state.textDirection !== 'auto' 
          ? state.textDirection 
          : (state.autoTextDirection !== 'auto' ? state.autoTextDirection : 'vertical')
      }))
    }));
    
    // 2. 收集图片（校对时优先使用翻译后的图片）
    const imageBase64Array = batch.map(t => {
      const dataUrl = t.imageData.translatedDataURL || t.imageData.originalDataURL;
      return this.extractBase64(dataUrl);
    });
    
    // 3. 遍历所有校对轮次（参考 proofreadTranslate.ts）
    let currentData = jsonData;
    for (const round of proofreading.rounds) {
      const jsonString = JSON.stringify(currentData, null, 2);
      type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
      const userContent: MessageContent[] = [
        {
          type: 'text',
          text: round.prompt + '\n\n以下是JSON数据:\n```json\n' + jsonString + '\n```'
        }
      ];
      for (const imgBase64 of imageBase64Array) {
        userContent.push({
          type: 'image_url',
          image_url: { url: `data:image/png;base64,${imgBase64}` }
        });
      }
      
      const messages: HqTranslateParams['messages'] = [
        { role: 'system', content: '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。' },
        { role: 'user', content: userContent }
      ];
      
      const response = await hqTranslateBatch({
        provider: round.provider,
        api_key: round.apiKey,
        model_name: round.modelName,
        custom_base_url: round.customBaseUrl,
        messages: messages,
        low_reasoning: round.lowReasoning,
        force_json_output: round.forceJsonOutput,
        no_thinking_method: round.noThinkingMethod,
        use_stream: false  // 校对不使用流式
      });
      
      const parsedResult = this.parseHqResponse(response, round.forceJsonOutput);
      if (parsedResult) {
        currentData = parsedResult;
      }
    }
    
    // 4. 填充校对结果并传递给渲染池
    for (const t of batch) {
      const taskData = currentData.find(d => d.imageIndex === t.imageIndex);
      if (taskData) {
        t.translateResult = {
          translatedTexts: taskData.bubbles.map(b => b.translated),
          textboxTexts: []
        };
      }
      t.status = 'processing';
      // 校对模式直接传给渲染池（跳过修复池）
      if (this.nextPool) {
        this.nextPool.enqueue(t);
      }
    }
    
    return task;
  }
  
  // ==================== 仅消除文字 ====================
  private async handleRemoveTextOnly(task: PipelineTask): Promise<PipelineTask> {
    // 不翻译，空结果
    task.translateResult = {
      translatedTexts: [],
      textboxTexts: []
    };
    return task;
  }
  
  // ==================== 辅助方法 ====================
  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || '';
    }
    return dataUrl;
  }
  
  /**
   * 解析高质量翻译/校对 API 响应
   * 参考：multimodalTranslate.ts 的解析逻辑
   */
  private parseHqResponse(
    response: { success: boolean; results?: any[]; content?: string; error?: string },
    forceJsonOutput: boolean
  ): TranslationJsonData[] | null {
    if (!response.success) {
      console.error('API调用失败:', response.error);
      return null;
    }
    
    // 优先使用后端已解析的 results
    if (response.results && response.results.length > 0) {
      const firstItem = response.results[0];
      if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
        return response.results as TranslationJsonData[];
      }
    }
    
    // 如果 results 不存在，尝试从 content 解析
    const content = (response as { content?: string }).content;
    if (content) {
      if (forceJsonOutput) {
        try {
          return JSON.parse(content);
        } catch (e) {
          console.error('解析AI强制JSON返回的内容失败:', e);
          return null;
        }
      } else {
        // 从 markdown 代码块中提取 JSON
        const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch?.[1]) {
          try {
            return JSON.parse(jsonMatch[1]);
          } catch (e) {
            console.error('解析AI返回的JSON失败:', e);
            return null;
          }
        }
      }
    }
    
    return null;
  }
}
```

### 5.4 渲染池 (RenderPool) - 实时更新

```typescript
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'

/**
 * 渲染池 - 最后一个池子，负责将翻译结果渲染到图片上
 * 
 * 关键特性：每完成一张图片的渲染，立即更新到界面
 * 
 * 注意：store 在 process 方法中获取，而不是在构造函数中
 * 参考：vue-frontend/src/composables/translation/steps/prepareStep.ts:34-35
 */
class RenderPool extends TaskPool {
  
  constructor(
    nextPool: null,  // 渲染池是最后一个，无下一个池子
    lock: null,      // 渲染不需要深度学习锁
    progressTracker: ParallelProgressTracker,
    private resultCollector: ResultCollector,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('渲染', nextPool, lock, progressTracker, onTaskComplete);
  }
  
  protected async process(task: PipelineTask): Promise<PipelineTask> {
    // ★ 在方法内部获取 store（不是在构造函数中）
    // 参考 prepareStep.ts 的写法
    const imageStore = useImageStore();
    const bubbleStore = useBubbleStore();
    
    // 构建渲染参数
    const bubbleStates = this.buildBubbleStates(task);
    
    // 调用后端 /api/parallel/render API
    const response = await parallelRenderApi({
      clean_image: task.inpaintResult!.cleanImage,
      bubble_states: bubbleStates,
      // ... 其他渲染参数
    });
    
    task.renderResult = {
      finalImage: response.final_image,
      bubbleStates: response.bubble_states
    };
    
    // ★★★ 关键：实时更新到界面 ★★★
    // 参考 prepareStep.ts:84-102 的写法
    this.updateImageToUI(task, imageStore, bubbleStore);
    
    // 收集结果（用于保序和统计）
    this.resultCollector.add(task);
    
    return task;
  }
  
  /**
   * 实时更新图片到界面
   * 每渲染完成一张图片，立即更新到 imageStore
   * 
   * 参考：vue-frontend/src/composables/translation/steps/prepareStep.ts:84-102
   */
  private updateImageToUI(
    task: PipelineTask, 
    imageStore: ReturnType<typeof useImageStore>,
    bubbleStore: ReturnType<typeof useBubbleStore>
  ): void {
    const imageIndex = task.imageIndex;
    
    // 1. 更新 imageStore（参考 prepareStep.ts:84-97）
    imageStore.updateImageByIndex(imageIndex, {
      translatedDataURL: `data:image/png;base64,${task.renderResult!.finalImage}`,
      // 注意：cleanImageData 在 prepareStep 中直接使用 response.clean_image
      // 但如果后端返回的是纯 base64，需要加前缀
      cleanImageData: task.inpaintResult?.cleanImage || null,
      bubbleStates: task.renderResult!.bubbleStates,
      bubbleCoords: task.detectionResult?.bubbleCoords || [],
      bubbleAngles: task.detectionResult?.bubbleAngles || [],
      originalTexts: task.ocrResult?.originalTexts || [],
      bubbleTexts: task.translateResult?.translatedTexts || [],
      textboxTexts: task.translateResult?.textboxTexts || [],
      translationStatus: 'completed',
      translationFailed: false,
      showOriginal: false,
      hasUnsavedChanges: true
    });
    
    // 2. 如果是当前显示的图片，同步更新 bubbleStore（参考 prepareStep.ts:100-102）
    if (imageIndex === imageStore.currentImageIndex && task.renderResult?.bubbleStates) {
      bubbleStore.setBubbles(task.renderResult.bubbleStates);
    }
    
    // 3. 更新进度显示
    this.progressTracker.updatePool('渲染', { 
      completed: this.completedCount + 1 
    });
    
    console.log(`✅ 图片 ${imageIndex + 1} 渲染完成并已更新到界面`);
  }
  
  /**
   * 构建 BubbleState 数组
   */
  private buildBubbleStates(task: PipelineTask): BubbleState[] {
    const coords = task.detectionResult?.bubbleCoords || [];
    const texts = task.translateResult?.translatedTexts || [];
    const originals = task.ocrResult?.originalTexts || [];
    const colors = task.colorResult?.colors || [];
    const angles = task.detectionResult?.bubbleAngles || [];
    const directions = task.detectionResult?.autoDirections || [];
    
    return coords.map((coord, idx) => ({
      originalText: originals[idx] || '',
      translatedText: texts[idx] || '',
      textboxText: '',
      coords: coord as [number, number, number, number],
      polygon: [],
      fontSize: 0,  // 使用自动字号
      fontFamily: '',  // 使用全局设置
      textDirection: 'auto' as TextDirection,
      autoTextDirection: (directions[idx] || 'vertical') as TextDirection,
      textColor: colors[idx]?.textColor || '',
      fillColor: colors[idx]?.bgColor || '',
      rotationAngle: angles[idx] || 0,
      position: { x: 0, y: 0 },
      strokeEnabled: false,
      strokeColor: '',
      strokeWidth: 0,
      inpaintMethod: 'solid' as InpaintMethod,
      autoFgColor: null,
      autoBgColor: null
    }));
  }
}
```

### 5.5 主控制器 (ParallelPipeline)

```typescript
class ParallelPipeline {
  private lock: DeepLearningLock;
  private progressTracker: ParallelProgressTracker;
  private resultCollector: ResultCollector;
  private poolRegistry: PoolRegistry;  // 池子注册中心
  
  private detectionPool: DetectionPool;
  private ocrPool: OcrPool;
  private colorPool: ColorPool;
  private translatePool: TranslatePool;
  private inpaintPool: InpaintPool;
  private renderPool: RenderPool;
  
  private isCancelled = false;
  
  constructor(private settingsStore: SettingsStore) {
    this.lock = new DeepLearningLock();
    this.progressTracker = new ParallelProgressTracker();
    this.resultCollector = new ResultCollector();
    
    // 初始化渲染池（最后一个）
    this.renderPool = new RenderPool(
      null,  // 无下一个池子
      null,  // 无锁
      this.progressTracker,
      (task) => this.resultCollector.add(task)
    );
    
    // 初始化修复池
    this.inpaintPool = new InpaintPool(
      this.renderPool,
      this.lock,
      this.progressTracker
    );
    
    // 初始化翻译池
    this.translatePool = new TranslatePool(
      this.inpaintPool,
      null,
      this.progressTracker,
      this.settingsStore
    );
    
    // 初始化颜色池
    this.colorPool = new ColorPool(
      this.translatePool,
      this.lock,
      this.progressTracker
    );
    
    // 初始化 OCR 池
    this.ocrPool = new OcrPool(
      this.colorPool,
      this.lock,
      this.progressTracker
    );
    
    // 初始化检测池（入口）
    this.detectionPool = new DetectionPool(
      this.ocrPool,
      this.lock,
      this.progressTracker
    );
  }
  
  /**
   * 执行并行翻译
   */
  async execute(
    images: ImageData[],
    mode: TranslationMode
  ): Promise<{ success: number; failed: number }> {
    this.reset();
    this.progressTracker.init(images.length);
    this.translatePool.setMode(mode, images.length);
    
    // 根据模式确定入口池
    if (mode === 'proofread') {
      // AI校对模式：跳过检测/OCR/颜色/修复，直接进入翻译池
      for (let i = 0; i < images.length; i++) {
        const task: PipelineTask = {
          id: `task-${i}`,
          imageIndex: i,
          imageData: images[i],
          status: 'pending'
        };
        this.translatePool.enqueue(task);
      }
    } else {
      // 其他模式：从检测池开始
      for (let i = 0; i < images.length; i++) {
        const task: PipelineTask = {
          id: `task-${i}`,
          imageIndex: i,
          imageData: images[i],
          status: 'pending'
        };
        this.detectionPool.enqueue(task);
      }
    }
    
    // 等待所有结果
    return this.resultCollector.waitForAll(images.length);
  }
  
  /**
   * 取消执行
   */
  cancel(): void {
    this.isCancelled = true;
    this.detectionPool.cancel();
    this.ocrPool.cancel();
    this.colorPool.cancel();
    this.translatePool.cancel();
    this.inpaintPool.cancel();
    this.renderPool.cancel();
  }
  
  /**
   * 重置所有池子
   */
  private reset(): void {
    this.isCancelled = false;
    this.detectionPool.reset();
    this.ocrPool.reset();
    this.colorPool.reset();
    this.translatePool.reset();
    this.inpaintPool.reset();
    this.renderPool.reset();
    this.resultCollector.reset();
  }
  
  /**
   * 获取实时进度
   */
  getProgress(): ParallelProgress {
    return this.progressTracker.getProgress();
  }
}
```

---

## 六、进度显示

### 6.1 进度数据结构

```typescript
interface PoolProgress {
  name: string;           // 池子名称
  icon: string;           // 图标
  completed: number;      // 已完成数
  total: number;          // 总数
  currentPage?: number;   // 当前处理页码
  isProcessing: boolean;  // 是否正在处理
  isWaitingLock: boolean; // 是否在等待锁
}

interface ParallelProgress {
  pools: PoolProgress[];     // 6个池子状态
  totalCompleted: number;    // 最终完成数
  totalFailed: number;       // 失败数
  totalPages: number;        // 总页数
  estimatedTimeRemaining: number; // 预计剩余时间（秒）
}
```

### 6.2 UI 显示效果

```
┌─────────────────────────────────────────────────────────────────────┐
│                        并行翻译进度                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📍 检测    [████████░░░░░░░░░░░░]  8/20   🔒处理: 页9               │
│  📖 OCR    [███████░░░░░░░░░░░░░]  7/20   ⏳等待锁                   │
│  🎨 颜色   [██████░░░░░░░░░░░░░░]  6/20   🔒处理: 页7               │
│  🌐 翻译   [█████░░░░░░░░░░░░░░░]  5/20   处理: 页6                 │
│  🖌️ 修复   [████░░░░░░░░░░░░░░░░]  4/20   ⏳等待锁                   │
│  ✨ 渲染   [███░░░░░░░░░░░░░░░░░]  3/20   处理: 页4                 │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│  ✅ 已完成: 3/20 页    ❌ 失败: 0 页    ⏱️ 剩余: 2分30秒             │
│                                                                      │
│                        [ 取消翻译 ]                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**状态图标**：
- 🔒 正在处理（持有深度学习锁）
- ⏳ 等待深度学习锁
- 无图标 = 普通处理中（翻译/渲染不需要锁）

---

## 七、文件结构

```
vue-frontend/src/
├── composables/translation/
│   ├── parallel/                        # 【新增】并行翻译模块
│   │   ├── index.ts                     # 模块入口，导出 ParallelPipeline
│   │   ├── types.ts                     # 类型定义
│   │   ├── ParallelPipeline.ts          # 主控制器
│   │   ├── TaskPool.ts                  # 通用任务池基类
│   │   ├── DeepLearningLock.ts          # 深度学习互斥锁
│   │   ├── pools/
│   │   │   ├── DetectionPool.ts         # 检测池
│   │   │   ├── OcrPool.ts               # OCR池
│   │   │   ├── ColorPool.ts             # 颜色池
│   │   │   ├── TranslatePool.ts         # 翻译池（含模式分发）
│   │   │   ├── InpaintPool.ts           # 修复池
│   │   │   └── RenderPool.ts            # 渲染池
│   │   ├── ParallelProgressTracker.ts   # 多进度条追踪器
│   │   └── ResultCollector.ts           # 结果收集器（保序）
│   │
│   ├── core/pipeline.ts                 # 【修改】添加并行模式判断分支
│   └── useTranslationPipeline.ts        # 【修改】检测并行开关
│
├── components/translation/
│   └── ParallelProgressBar.vue          # 【新增】多进度条组件
│
├── stores/
│   └── settingsStore.ts                 # 【修改】添加 parallel 配置
│
└── types/
    └── settings.ts                      # 【修改】添加 ParallelConfig 类型

src/app/api/translation/
└── parallel_routes.py                   # 【新增】6个独立步骤API
```

---

## 八、后端 API

### 8.1 新增的独立步骤 API

```python
# src/app/api/translation/parallel_routes.py

from flask import Blueprint, request, jsonify
from src.core import detection, ocr, color_extractor, translation, inpainting, rendering

parallel_bp = Blueprint('parallel', __name__)

@parallel_bp.route('/parallel/detect', methods=['POST'])
def parallel_detect():
    """仅执行检测步骤"""
    data = request.get_json()
    image_data = data.get('image')  # Base64
    
    # 解码图片
    img = decode_base64_image(image_data)
    
    # 执行检测
    result = detection.get_bubble_detection_result_with_auto_directions(
        img,
        conf_threshold=data.get('conf_threshold', 0.6),
        detector_type=data.get('detector_type'),
        # ... 其他参数
    )
    
    return jsonify({
        'bubble_coords': result['coords'],
        'bubble_angles': result['angles'],
        'bubble_polygons': result['polygons'],
        'auto_directions': result['auto_directions'],
        'raw_mask': encode_mask_to_base64(result['raw_mask']) if result['raw_mask'] else None,
        'textlines_per_bubble': result['textlines_per_bubble']
    })


@parallel_bp.route('/parallel/ocr', methods=['POST'])
def parallel_ocr():
    """仅执行OCR步骤"""
    data = request.get_json()
    image_data = data.get('image')
    bubble_coords = data.get('bubble_coords')
    
    img = decode_base64_image(image_data)
    
    original_texts = ocr.recognize_text_in_bubbles(
        img, bubble_coords,
        source_language=data.get('source_language', 'japan'),
        ocr_engine=data.get('ocr_engine', 'paddle_ocr'),
        # ... 其他参数
    )
    
    return jsonify({
        'original_texts': original_texts,
        'textlines_per_bubble': data.get('textlines_per_bubble', [])
    })


@parallel_bp.route('/parallel/color', methods=['POST'])
def parallel_color():
    """仅执行颜色提取步骤"""
    data = request.get_json()
    image_data = data.get('image')
    bubble_coords = data.get('bubble_coords')
    
    img = decode_base64_image(image_data)
    
    colors = color_extractor.extract_colors(
        img, bubble_coords,
        textlines_per_bubble=data.get('textlines_per_bubble')
    )
    
    return jsonify({
        'colors': colors
    })


@parallel_bp.route('/parallel/translate', methods=['POST'])
def parallel_translate():
    """仅执行翻译步骤（普通模式，逐条）"""
    data = request.get_json()
    original_texts = data.get('original_texts', [])
    
    translated_texts, textbox_texts = translation.translate_text_list(
        original_texts,
        target_language=data.get('target_language'),
        source_language=data.get('source_language'),
        api_key=data.get('api_key'),
        model_name=data.get('model_name'),
        model_provider=data.get('model_provider'),
        prompt_content=data.get('prompt_content'),
        # ... 其他参数
    )
    
    return jsonify({
        'translated_texts': translated_texts,
        'textbox_texts': textbox_texts
    })


@parallel_bp.route('/parallel/inpaint', methods=['POST'])
def parallel_inpaint():
    """仅执行修复步骤"""
    data = request.get_json()
    image_data = data.get('image')
    bubble_coords = data.get('bubble_coords')
    
    img = decode_base64_image(image_data)
    
    clean_image = inpainting.inpaint_bubbles(
        img, bubble_coords,
        method=data.get('method', 'solid'),
        fill_color=data.get('fill_color'),
        bubble_polygons=data.get('bubble_polygons'),
        precise_mask=decode_mask_from_base64(data.get('raw_mask')),
        # ... 其他参数
    )
    
    return jsonify({
        'clean_image': encode_image_to_base64(clean_image)
    })


@parallel_bp.route('/parallel/render', methods=['POST'])
def parallel_render():
    """仅执行渲染步骤"""
    data = request.get_json()
    clean_image_data = data.get('clean_image')
    bubble_states = data.get('bubble_states', [])
    
    img = decode_base64_image(clean_image_data)
    
    final_image, updated_states = rendering.render_bubbles_unified(
        img, bubble_states,
        # ... 其他参数
    )
    
    return jsonify({
        'final_image': encode_image_to_base64(final_image),
        'bubble_states': updated_states
    })
```

---

## 九、设置界面

**位置**：设置 → 更多

```
┌─────────────────────────────────────────────────┐
│ 并行翻译                                         │
├─────────────────────────────────────────────────┤
│                                                  │
│ [ ] 启用并行翻译模式                              │
│                                                  │
│ 💡 启用后，批量翻译时多张图片将以流水线方式        │
│    并行处理，可显著提升翻译速度。                  │
│                                                  │
│ ⚠️ 注意：                                        │
│ • 检测/OCR/颜色提取/背景修复共用GPU锁，避免       │
│   显存溢出                                       │
│ • 翻译和渲染不受GPU锁限制                        │
│ • AI校对模式会跳过检测/OCR/颜色/修复阶段         │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 十、预期性能提升

### 10.1 各阶段耗时假设
| 阶段 | 平均耗时 |
|------|---------|
| 检测 | 0.5s |
| OCR | 0.3s |
| 颜色 | 0.1s |
| 翻译 | 1.0s |
| 修复 | 0.8s |
| 渲染 | 0.3s |
| **总计** | **3.0s/页** |

### 10.2 串行 vs 并行（20张图片）

| 模式 | 计算方式 | 总耗时 |
|------|---------|--------|
| **串行** | 20 × 3.0s | **60秒** |
| **并行** | 首页3.0s + 后续19页×1.0s（瓶颈为翻译） | **22秒** |

**加速比：约 2.7 倍** 🚀

### 10.3 深度学习锁的影响

由于检测/OCR/颜色/修复共用一把锁，实际并行度会受限：
- 这4个阶段实际上是串行的
- 但翻译和渲染可以与它们并行
- 仍然能获得显著加速

---

## 十一、实现清单

### 前端
- [ ] 新增 `parallel/` 目录及所有文件
- [ ] 实现 `DeepLearningLock` 类
- [ ] 实现 `TaskPool` 基类
- [ ] 实现 6 个具体池子类
- [ ] 实现 `ParallelPipeline` 主控制器
- [ ] 实现 `ParallelProgressTracker` 进度追踪
- [ ] 实现 `ResultCollector` 结果收集
- [ ] 新增 `ParallelProgressBar.vue` 组件
- [ ] 修改 `settingsStore` 添加 `parallel` 配置
- [ ] 修改 `pipeline.ts` 添加并行分支
- [ ] 修改设置界面添加并行开关

### 后端
- [ ] 新增 `parallel_routes.py`
- [ ] 实现 6 个独立步骤 API
- [ ] 注册路由到 Flask 应用

---

**方案完成，可以开始实现。**
