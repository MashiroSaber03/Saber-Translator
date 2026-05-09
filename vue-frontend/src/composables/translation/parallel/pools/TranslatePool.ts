/**
 * 翻译池
 *
 * 负责：
 * - standard: 调度共享单页 translate 原子步骤
 * - hq/proofread: 组批后调度共享 aiTranslate 批量原子步骤
 * - removeText: 跳过翻译
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask, ParallelTranslationMode } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep, executeBatchAtomicStep } from '@/composables/translation/core/atomicSteps'

export class TranslatePool extends TaskPool {
  private mode: ParallelTranslationMode = 'standard'
  private batchBuffer: PipelineTask[] = []
  private totalTasks = 0
  private processedCount = 0

  constructor(
    nextPool: TaskPool | null,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('翻译', '🌐', nextPool, null, progressTracker, onTaskComplete)
  }

  setMode(mode: ParallelTranslationMode, totalTasks: number, nextPool: TaskPool | null): void {
    this.mode = mode
    this.totalTasks = totalTasks
    this.batchBuffer = []
    this.processedCount = 0
    this.nextPool = nextPool
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    switch (this.mode) {
      case 'standard':
        return this.handleStandardTranslate(task)
      case 'hq':
      case 'proofread':
        return this.handleBatchTranslate(task)
      default:
        return {
          ...task,
          status: 'processing',
        }
    }
  }

  private async handleStandardTranslate(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('翻译步骤缺少运行时上下文')
    }

    if (task.originalTexts.length === 0) {
      return {
        ...task,
        status: 'processing',
        translatedTexts: [],
        textboxTexts: [],
        warnings: [],
      }
    }

    return await executeAtomicStep('translate', task, runtime)
  }

  private async handleBatchTranslate(task: PipelineTask): Promise<PipelineTask> {
    this.batchBuffer.push(task)
    this.processedCount++

    const runtime = task.runtime
    if (!runtime) {
      throw new Error('AI 翻译步骤缺少运行时上下文')
    }
    const batchSize = this.mode === 'hq'
      ? (runtime.settingsSnapshot.hqTranslation.batchSize || 3)
      : (runtime.settingsSnapshot.proofreading.rounds[0]?.batchSize || 3)
    const isLastBatch = this.processedCount >= this.totalTasks
    const batchReady = this.batchBuffer.length >= batchSize || isLastBatch

    if (!batchReady) {
      return {
        ...task,
        status: 'buffered',
      }
    }

    const batch = [...this.batchBuffer]
    this.batchBuffer = []

    const translatedBatch = await executeBatchAtomicStep('aiTranslate', batch, runtime)

    for (const translatedTask of translatedBatch) {
      if (this.nextPool) {
        this.nextPool.enqueue(translatedTask)
      }
    }

    return {
      ...task,
      status: 'buffered',
    }
  }
}
