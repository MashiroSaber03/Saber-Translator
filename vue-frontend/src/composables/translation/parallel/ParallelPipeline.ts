/**
 * 并行翻译管线主控制器
 *
 * 协调各个调度池，但具体业务逻辑全部交给共享 atomic steps。
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
  RenderPool,
  SavePool,
} from './pools'
import { useImageStore } from '@/stores/imageStore'
import { createPipelineRuntime, hydrateTaskContextFromImage, type PipelineRuntime } from '@/composables/translation/core/runtime'

export const POOL_CHAIN_CONFIGS: Record<ParallelTranslationMode, string[]> = {
  standard: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  hq: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  proofread: ['translate', 'render'],
  removeText: ['detection', 'inpaint', 'render'],
}

export class ParallelPipeline {
  private lock: DeepLearningLock
  private progressTracker: ParallelProgressTracker
  private resultCollector: ResultCollector
  private activeTaskIndices: number[] = []

  private detectionPool: DetectionPool
  private ocrPool: OcrPool
  private colorPool: ColorPool
  private translatePool: TranslatePool
  private inpaintPool: InpaintPool
  private renderPool: RenderPool
  private savePool: SavePool

  private isCancelled = false
  private config: ParallelConfig
  constructor(config: ParallelConfig) {
    this.config = config
    this.lock = new DeepLearningLock(config.deepLearningLockSize)
    this.progressTracker = new ParallelProgressTracker()
    this.resultCollector = new ResultCollector()
    const imageStore = useImageStore()

    const handleTaskComplete = (task: PipelineTask) => {
      if (this.isCancelled || task.status !== 'failed') {
        return
      }

      imageStore.setTranslationStatus(task.imageIndex, 'failed', task.error || '未知错误')
      this.resultCollector.add(task)
      this.progressTracker.incrementFailed()
    }

    this.savePool = new SavePool(
      this.progressTracker,
      this.resultCollector,
      handleTaskComplete,
    )

    this.renderPool = new RenderPool(
      null,
      this.progressTracker,
      this.resultCollector,
      handleTaskComplete,
    )

    this.inpaintPool = new InpaintPool(
      this.renderPool,
      this.lock,
      this.progressTracker,
      handleTaskComplete,
    )

    this.translatePool = new TranslatePool(
      this.inpaintPool,
      this.progressTracker,
      handleTaskComplete,
    )

    this.colorPool = new ColorPool(
      this.translatePool,
      this.lock,
      this.progressTracker,
      handleTaskComplete,
    )

    this.ocrPool = new OcrPool(
      this.colorPool,
      this.lock,
      this.progressTracker,
      handleTaskComplete,
    )

    this.detectionPool = new DetectionPool(
      this.ocrPool,
      this.lock,
      this.progressTracker,
      handleTaskComplete,
    )
  }

  updateConfig(config: Partial<ParallelConfig>): void {
    if (config.deepLearningLockSize !== undefined) {
      this.lock.setSize(config.deepLearningLockSize)
      this.config.deepLearningLockSize = config.deepLearningLockSize
    }
    if (config.enabled !== undefined) {
      this.config.enabled = config.enabled
    }
  }

  async execute(
    images: ImageData[],
    mode: ParallelTranslationMode,
    startIndex: number = 0
  ): Promise<ParallelExecutionResult> {
    const imageStore = useImageStore()
    this.reset()
    this.progressTracker.init(images.length)
    this.resultCollector.init(images.length)

    const runtime = createPipelineRuntime(mode)
    this.setupPoolChain(mode, images.length, runtime)

    const tasks: PipelineTask[] = images.map((imageData, localIndex) =>
      hydrateTaskContextFromImage(startIndex + localIndex, imageData, mode, runtime),
    )
    this.activeTaskIndices = tasks.map((task) => task.imageIndex)

    for (const task of tasks) {
      imageStore.setTranslationStatus(task.imageIndex, 'processing')
    }

    const entryPool = this.getEntryPool(mode)
    for (const task of tasks) {
      entryPool.enqueue(task)
    }

    const result = await this.resultCollector.waitForAll(images.length)
    this.activeTaskIndices = []

    return {
      success: result.success,
      failed: result.failed,
      errors: this.resultCollector.getFailed().map((task) => task.error || '未知错误'),
    }
  }

  private setupPoolChain(mode: ParallelTranslationMode, totalTasks: number, runtime: PipelineRuntime): void {
    const chainConfig = [...POOL_CHAIN_CONFIGS[mode]]

    if (mode === 'removeText' && runtime.settingsSnapshot.removeTextWithOcr) {
      const detectionIdx = chainConfig.indexOf('detection')
      if (detectionIdx !== -1 && !chainConfig.includes('ocr')) {
        chainConfig.splice(detectionIdx + 1, 0, 'ocr')
      }
    }

    if (runtime.autoSaveEnabled) {
      chainConfig.push('save')
    }

    const poolMap = this.getPoolMap()

    for (let i = 0; i < chainConfig.length - 1; i++) {
      const currentPoolName = chainConfig[i] as string
      const nextPoolName = chainConfig[i + 1] as string
      const currentPool = poolMap[currentPoolName]
      const nextPool = poolMap[nextPoolName]
      if (currentPool && nextPool) {
        currentPool.setNextPool(nextPool)
      }
    }

    const lastPoolName = chainConfig[chainConfig.length - 1] as string
    const lastPool = poolMap[lastPoolName]
    if (lastPool) {
      lastPool.setNextPool(null)
    }

    const translateIndex = chainConfig.indexOf('translate')
    if (translateIndex >= 0 && translateIndex < chainConfig.length - 1) {
      const nextPoolName = chainConfig[translateIndex + 1] as string
      this.translatePool.setMode(mode, totalTasks, poolMap[nextPoolName] || null)
    } else if (translateIndex === chainConfig.length - 1) {
      this.translatePool.setMode(mode, totalTasks, null)
    }
  }

  private getPoolMap(): Record<string, TaskPool> {
    return {
      detection: this.detectionPool,
      ocr: this.ocrPool,
      color: this.colorPool,
      translate: this.translatePool,
      inpaint: this.inpaintPool,
      render: this.renderPool,
      save: this.savePool,
    }
  }

  private getEntryPool(mode: ParallelTranslationMode): TaskPool {
    const chainConfig = POOL_CHAIN_CONFIGS[mode]
    const firstPoolName = chainConfig[0] as string
    const poolMap = this.getPoolMap()
    return poolMap[firstPoolName] || this.detectionPool
  }

  cancel(): void {
    const imageStore = useImageStore()
    this.isCancelled = true
    this.detectionPool.cancel()
    this.ocrPool.cancel()
    this.colorPool.cancel()
    this.translatePool.cancel()
    this.inpaintPool.cancel()
    this.renderPool.cancel()
    this.savePool.cancel()
    this.lock.reset()

    for (const imageIndex of this.activeTaskIndices) {
      const image = imageStore.images[imageIndex]
      if (image?.translationStatus === 'processing') {
        imageStore.setTranslationStatus(imageIndex, 'pending')
      }
    }

    this.activeTaskIndices = []
    this.resultCollector.finishEarly()
  }

  private reset(): void {
    this.isCancelled = false
    this.activeTaskIndices = []
    this.detectionPool.reset()
    this.ocrPool.reset()
    this.colorPool.reset()
    this.translatePool.reset()
    this.inpaintPool.reset()
    this.renderPool.reset()
    this.savePool.reset()
    this.resultCollector.reset()
    this.progressTracker.reset()
  }

  get progress() {
    return this.progressTracker.progress
  }

  getProgressTracker(): ParallelProgressTracker {
    return this.progressTracker
  }

  getLockStatus() {
    return this.lock.getStatus()
  }

  get cancelled(): boolean {
    return this.isCancelled
  }
}

export function createParallelPipeline(config: ParallelConfig): ParallelPipeline {
  return new ParallelPipeline(config)
}
