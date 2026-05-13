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
  AutoGlossaryPool,
  TranslatePool,
  InpaintPool,
  RenderPool,
  SavePool,
} from './pools'
import { useImageStore } from '@/stores/imageStore'
import { createPipelineRuntime, hydrateTaskContextFromImage } from '@/composables/translation/core/runtime'
import { resolveParallelPoolChain, type ParallelPoolStepName } from '@/composables/translation/core/pipelineRegistry'

export class ParallelPipeline {
  private lock: DeepLearningLock
  private progressTracker: ParallelProgressTracker
  private resultCollector: ResultCollector
  private activeTaskIndices: number[] = []

  private detectionPool: DetectionPool
  private ocrPool: OcrPool
  private colorPool: ColorPool
  private autoGlossaryPool: AutoGlossaryPool
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

    this.autoGlossaryPool = new AutoGlossaryPool(
      this.translatePool,
      this.progressTracker,
      handleTaskComplete,
    )

    this.colorPool = new ColorPool(
      this.autoGlossaryPool,
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
    const chainConfig = resolveParallelPoolChain(mode, {
      removeTextWithOcr: runtime.settingsSnapshot.removeTextWithOcr,
      autoSaveEnabled: runtime.autoSaveEnabled,
    })
    this.setupPoolChain(mode, images.length, chainConfig)

    const tasks: PipelineTask[] = images.map((imageData, localIndex) =>
      hydrateTaskContextFromImage(startIndex + localIndex, imageData, mode, runtime),
    )
    this.activeTaskIndices = tasks.map((task) => task.imageIndex)

    for (const task of tasks) {
      imageStore.setTranslationStatus(task.imageIndex, 'processing')
    }

    const entryPool = this.getEntryPool(chainConfig)
    for (const task of tasks) {
      entryPool.enqueue(task)
    }

    const result = await this.resultCollector.waitForAll(images.length)
    this.activeTaskIndices = []

    return {
      success: result.success,
      failed: result.failed,
      errors: this.resultCollector.getFailed().map((task) => task.error || '未知错误'),
      autoGlossaryStats: this.resultCollector.getAll().reduce((total, task) => ({
        added: total.added + (task.autoGlossaryStats?.added || 0),
        duplicates: total.duplicates + (task.autoGlossaryStats?.duplicates || 0),
        failedPages: total.failedPages + (task.autoGlossaryStats?.failedPages || 0),
      }), {
        added: 0,
        duplicates: 0,
        failedPages: 0,
      }),
    }
  }

  private setupPoolChain(
    mode: ParallelTranslationMode,
    totalTasks: number,
    chainConfig: ParallelPoolStepName[],
  ): void {
    const poolMap = this.getPoolMap()

    for (let i = 0; i < chainConfig.length - 1; i++) {
      const currentPoolName = chainConfig[i]!
      const nextPoolName = chainConfig[i + 1]!
      const currentPool = poolMap[currentPoolName]
      const nextPool = poolMap[nextPoolName]
      if (currentPool && nextPool) {
        currentPool.setNextPool(nextPool)
      }
    }

    const lastPoolName = chainConfig[chainConfig.length - 1]!
    const lastPool = poolMap[lastPoolName]
    if (lastPool) {
      lastPool.setNextPool(null)
    }

    const translateIndex = chainConfig.indexOf('translate')
    if (translateIndex >= 0 && translateIndex < chainConfig.length - 1) {
      const nextPoolName = chainConfig[translateIndex + 1]!
      this.translatePool.setMode(mode, totalTasks, poolMap[nextPoolName] || null)
    } else if (translateIndex === chainConfig.length - 1) {
      this.translatePool.setMode(mode, totalTasks, null)
    }
  }

  private getPoolMap(): Record<ParallelPoolStepName, TaskPool> {
    return {
      detection: this.detectionPool,
      ocr: this.ocrPool,
      color: this.colorPool,
      autoGlossary: this.autoGlossaryPool,
      translate: this.translatePool,
      inpaint: this.inpaintPool,
      render: this.renderPool,
      save: this.savePool,
    }
  }

  private getEntryPool(chainConfig: ParallelPoolStepName[]): TaskPool {
    const firstPoolName = chainConfig[0]!
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
    this.autoGlossaryPool.cancel()
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
    this.autoGlossaryPool.reset()
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
