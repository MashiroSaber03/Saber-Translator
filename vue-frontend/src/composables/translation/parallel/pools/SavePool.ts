import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import type { ResultCollector } from '../ResultCollector'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'
import { projectTaskContext } from '@/composables/translation/core/taskProjector'

export class SavePool extends TaskPool {
  private resultCollector: ResultCollector

  constructor(
    progressTracker: ParallelProgressTracker,
    resultCollector: ResultCollector,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('保存', 'save', null, null, progressTracker, onTaskComplete)
    this.resultCollector = resultCollector
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('保存步骤缺少运行时上下文')
    }

    const persistedTask = await executeAtomicStep('save', task, runtime)
    const completedTask: PipelineTask = {
      ...persistedTask,
      status: 'completed',
      persisted: true,
    }

    projectTaskContext(completedTask, runtime)

    this.progressTracker.incrementSaveCompleted()
    this.progressTracker.incrementCompleted()
    this.resultCollector.add(completedTask)
    return completedTask
  }
}
