/**
 * 并行翻译管线主控制器
 * 
 * 协调6个池子的工作，管理任务流动
 */

import type { ImageData } from '@/types/image'
import type { PipelineTask, ParallelTranslationMode, ParallelExecutionResult, ParallelConfig } from './types'
import { DeepLearningLock } from './DeepLearningLock'
import { ParallelProgressTracker } from './ParallelProgressTracker'
import { ResultCollector } from './ResultCollector'
import { TaskPool } from './TaskPool'
import {
  DetectionPool,
  OcrPool,
  ColorPool,
  TranslatePool,
  InpaintPool,
  RenderPool
} from './pools'
import { useSettingsStore } from '@/stores/settingsStore'

/**
 * 池子链配置
 * 
 * 扩展性设计：
 * - 添加新模式：在此添加新的池子链配置
 * - 添加新池子：在ParallelPipeline中创建池子实例，并在配置中使用
 * - 修改流程：直接修改对应模式的池子链数组
 */
export const POOL_CHAIN_CONFIGS: Record<ParallelTranslationMode, string[]> = {
  standard: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  hq: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  proofread: ['translate', 'render'],  // AI校对跳过检测/OCR/颜色/修复
  removeText: ['detection', 'inpaint', 'render']  // 仅消除文字：跳过OCR/颜色/翻译，只检测+修复+更新UI
}

export class ParallelPipeline {
  private lock: DeepLearningLock
  private progressTracker: ParallelProgressTracker
  private resultCollector: ResultCollector

  private detectionPool: DetectionPool
  private ocrPool: OcrPool
  private colorPool: ColorPool
  private translatePool: TranslatePool
  private inpaintPool: InpaintPool
  private renderPool: RenderPool

  private isCancelled = false
  private config: ParallelConfig

  constructor(config: ParallelConfig) {
    this.config = config
    this.lock = new DeepLearningLock(config.deepLearningLockSize)
    this.progressTracker = new ParallelProgressTracker()
    this.resultCollector = new ResultCollector()

    // 初始化渲染池（最后一个）
    this.renderPool = new RenderPool(
      this.progressTracker,
      this.resultCollector
    )

    // 初始化修复池
    this.inpaintPool = new InpaintPool(
      this.renderPool,
      this.lock,
      this.progressTracker
    )

    // 初始化翻译池（无锁）
    this.translatePool = new TranslatePool(
      this.inpaintPool,
      this.progressTracker
    )

    // 初始化颜色池
    this.colorPool = new ColorPool(
      this.translatePool,
      this.lock,
      this.progressTracker
    )

    // 初始化 OCR 池
    this.ocrPool = new OcrPool(
      this.colorPool,
      this.lock,
      this.progressTracker
    )

    // 初始化检测池（入口）
    this.detectionPool = new DetectionPool(
      this.ocrPool,
      this.lock,
      this.progressTracker
    )
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<ParallelConfig>): void {
    if (config.deepLearningLockSize !== undefined) {
      this.lock.setSize(config.deepLearningLockSize)
      this.config.deepLearningLockSize = config.deepLearningLockSize
    }
    if (config.enabled !== undefined) {
      this.config.enabled = config.enabled
    }
  }

  /**
   * 执行并行翻译
   * @param images 要处理的图片数组
   * @param mode 翻译模式
   * @param startIndex 起始索引（用于范围翻译时保持原始索引）
   */
  async execute(
    images: ImageData[],
    mode: ParallelTranslationMode,
    startIndex: number = 0
  ): Promise<ParallelExecutionResult> {
    this.reset()
    this.progressTracker.init(images.length)
    this.resultCollector.init(images.length)

    // 根据模式配置池子链
    this.setupPoolChain(mode, images.length)

    // 创建任务 - 使用原始索引（startIndex + localIndex）
    const tasks: PipelineTask[] = images.map((imageData, localIndex) => ({
      id: `task-${startIndex + localIndex}`,
      imageIndex: startIndex + localIndex,  // 使用原始索引
      imageData,
      status: 'pending'
    }))

    // 根据模式确定入口池
    const entryPool = this.getEntryPool(mode)

    // 将任务添加到入口池
    for (const task of tasks) {
      entryPool.enqueue(task)
    }

    // 等待所有结果
    const result = await this.resultCollector.waitForAll(images.length)

    return {
      success: result.success,
      failed: result.failed,
      errors: this.resultCollector.getFailed().map(t => t.error || '未知错误')
    }
  }

  /**
   * 根据模式配置池子链
   * 
   * 扩展性：通过POOL_CHAIN_CONFIGS配置不同模式的池子链
   */
  private setupPoolChain(mode: ParallelTranslationMode, totalTasks: number): void {
    // 动态生成池子链配置
    let chainConfig = [...POOL_CHAIN_CONFIGS[mode]]

    // 消除文字模式：根据设置决定是否包含 OCR 步骤
    if (mode === 'removeText') {
      try {
        const settingsStore = useSettingsStore()
        if (settingsStore.settings.removeTextWithOcr) {
          // 在 detection 后插入 ocr 步骤
          const detectionIdx = chainConfig.indexOf('detection')
          if (detectionIdx !== -1 && !chainConfig.includes('ocr')) {
            chainConfig.splice(detectionIdx + 1, 0, 'ocr')
          }
        }
      } catch {
        // 忽略错误，使用默认配置
      }
    }

    const poolMap = this.getPoolMap()

    // 根据配置动态设置池子链
    for (let i = 0; i < chainConfig.length - 1; i++) {
      const currentPoolName = chainConfig[i] as string
      const nextPoolName = chainConfig[i + 1] as string
      const currentPool = poolMap[currentPoolName]
      const nextPool = poolMap[nextPoolName]

      if (currentPool && nextPool) {
        currentPool.setNextPool(nextPool)
      }
    }

    // 最后一个池子不连接下一个
    const lastPoolName = chainConfig[chainConfig.length - 1] as string
    const lastPool = poolMap[lastPoolName]
    if (lastPool) {
      lastPool.setNextPool(null)
    }

    // 设置翻译池的模式（批量处理需要知道总任务数）
    const translateIndex = chainConfig.indexOf('translate')
    if (translateIndex >= 0 && translateIndex < chainConfig.length - 1) {
      const nextPoolName = chainConfig[translateIndex + 1] as string
      this.translatePool.setMode(mode, totalTasks, poolMap[nextPoolName] || null)
    } else if (translateIndex === chainConfig.length - 1) {
      this.translatePool.setMode(mode, totalTasks, null)
    }
  }

  /**
   * 获取池子名称到实例的映射
   * 
   * 扩展性：添加新池子时，在此添加映射
   */
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

  /**
   * 获取入口池
   * 
   * 扩展性：根据配置的第一个池子确定入口
   */
  private getEntryPool(mode: ParallelTranslationMode): TaskPool {
    const chainConfig = POOL_CHAIN_CONFIGS[mode]
    const firstPoolName = chainConfig[0] as string
    const poolMap = this.getPoolMap()
    return poolMap[firstPoolName] || this.detectionPool
  }

  /**
   * 取消执行
   */
  cancel(): void {
    this.isCancelled = true
    this.detectionPool.cancel()
    this.ocrPool.cancel()
    this.colorPool.cancel()
    this.translatePool.cancel()
    this.inpaintPool.cancel()
    this.renderPool.cancel()
    this.lock.reset()
  }

  /**
   * 重置所有池子
   */
  private reset(): void {
    this.isCancelled = false
    this.detectionPool.reset()
    this.ocrPool.reset()
    this.colorPool.reset()
    this.translatePool.reset()
    this.inpaintPool.reset()
    this.renderPool.reset()
    this.resultCollector.reset()
    this.progressTracker.reset()
  }

  /**
   * 获取实时进度（响应式）
   */
  get progress() {
    return this.progressTracker.progress
  }

  /**
   * 获取进度追踪器
   */
  getProgressTracker(): ParallelProgressTracker {
    return this.progressTracker
  }

  /**
   * 获取锁状态
   */
  getLockStatus() {
    return this.lock.getStatus()
  }

  /**
   * 是否已取消
   */
  get cancelled(): boolean {
    return this.isCancelled
  }
}

/**
 * 创建并行管线的工厂函数
 */
export function createParallelPipeline(config: ParallelConfig): ParallelPipeline {
  return new ParallelPipeline(config)
}
