/**
 * 自动术语提取池
 *
 * 负责在正式翻译前根据 OCR 结果自动提取当前页术语并写回书籍术语表。
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'

export class AutoGlossaryPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('术语', '📚', nextPool, null, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('自动术语提取步骤缺少运行时上下文')
    }

    return await executeAtomicStep('autoGlossary', task, runtime)
  }
}
