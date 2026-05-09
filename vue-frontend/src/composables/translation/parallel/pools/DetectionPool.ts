/**
 * 检测池
 * 
 * 负责调用后端检测API，获取气泡坐标、角度、多边形等信息
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'

export class DetectionPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('检测', '📍', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('检测步骤缺少运行时上下文')
    }

    return await executeAtomicStep('detection', task, runtime)
  }
}
