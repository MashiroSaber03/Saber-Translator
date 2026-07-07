import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import type { ResultCollector } from '../ResultCollector'
import { executeAtomicStep } from '@/composables/translation/core/atomicSteps'
import { projectTaskContext } from '@/composables/translation/core/taskProjector'

export class RenderPool extends TaskPool {
  private resultCollector: ResultCollector

  constructor(
    nextPool: TaskPool | null,
    progressTracker: ParallelProgressTracker,
    resultCollector: ResultCollector,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('渲染', 'sparkles', nextPool, null, progressTracker, onTaskComplete)
    this.resultCollector = resultCollector
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const runtime = task.runtime
    if (!runtime) {
      throw new Error('渲染步骤缺少运行时上下文')
    }

    const renderedTask = await executeAtomicStep('render', task, runtime)
    const projectedTask: PipelineTask = {
      ...renderedTask,
      status: this.nextPool ? 'processing' : 'completed',
    }

    if (this.isCancelled) {
      return projectedTask
    }

    projectTaskContext(projectedTask, runtime)

    if (!this.nextPool) {
      this.progressTracker.incrementCompleted()
      this.resultCollector.add(projectedTask)
    }

    return projectedTask
  }
}
