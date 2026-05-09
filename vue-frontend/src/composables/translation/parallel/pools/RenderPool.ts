/**
 * 渲染池
 *
 * 使用共享 render 原子步骤生成页面结果，并通过共享 projector 同步到界面。
 * 当自动保存关闭时，render 就是最终完成步骤；
 * 当自动保存开启时，render 只负责预览投影，真正完成由 SavePool 决定。
 */

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
    super('渲染', '✨', nextPool, null, progressTracker, onTaskComplete)
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
