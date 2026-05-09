/**
 * OCR池
 * 
 * 负责调用后端OCR API，识别气泡中的文字
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'

export class OcrPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('OCR', '📖', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('OCR 步骤缺少运行时上下文')
    }

    if (task.bubbleCoords.length === 0) {
      return {
        ...task,
        status: 'processing',
        originalTexts: [],
        ocrResults: [],
      }
    }

    return await executeAtomicStep('ocr', task, runtime)
  }
}
