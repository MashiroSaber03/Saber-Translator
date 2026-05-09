/**
 * 颜色提取池
 * 
 * 负责调用后端颜色提取API，识别文字和背景颜色
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'

export class ColorPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('颜色', '🎨', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('颜色步骤缺少运行时上下文')
    }

    if (task.bubbleCoords.length === 0) {
      return {
        ...task,
        status: 'processing',
        colors: [],
      }
    }

    return await executeAtomicStep('color', task, runtime)
  }
}
