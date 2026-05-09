/**
 * 修复池
 * 
 * 负责调用后端修复API，生成干净的背景图
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'

export class InpaintPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('修复', '🖌️', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('修复步骤缺少运行时上下文')
    }

    return await executeAtomicStep('inpaint', task, runtime)
  }
}
