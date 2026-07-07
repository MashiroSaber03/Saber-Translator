import type { PipelineTask, PoolStatus } from './types'
import type { DeepLearningLock } from './DeepLearningLock'
import type { ParallelProgressTracker } from './ParallelProgressTracker'

export abstract class TaskPool {
  protected queue: PipelineTask[] = []
  protected currentTask: PipelineTask | null = null
  protected isRunning = false
  protected isCancelled = false
  protected completedCount = 0
  protected nextPool: TaskPool | null
  protected name: string
  protected icon: PoolStatus['icon']
  protected lock: DeepLearningLock | null
  protected progressTracker: ParallelProgressTracker
  protected onTaskComplete?: (task: PipelineTask) => void

  constructor(
    name: string,
    icon: PoolStatus['icon'],
    nextPool: TaskPool | null,
    lock: DeepLearningLock | null,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    this.name = name
    this.icon = icon
    this.nextPool = nextPool
    this.lock = lock
    this.progressTracker = progressTracker
    this.onTaskComplete = onTaskComplete
  }

  getName(): string {
    return this.name
  }

  getIcon(): PoolStatus['icon'] {
    return this.icon
  }

  setNextPool(pool: TaskPool | null): void {
    this.nextPool = pool
  }

  enqueue(task: PipelineTask): void {
    if (this.isCancelled) return
    this.queue.push(task)
    this.progressTracker.updatePool(this.name, { waiting: this.queue.length })
    this.tryProcessNext()
  }

  enqueueBatch(tasks: PipelineTask[]): void {
    if (this.isCancelled) return
    for (const task of tasks) {
      this.queue.push(task)
    }
    this.progressTracker.updatePool(this.name, { waiting: this.queue.length })
    this.tryProcessNext()
  }

  private async tryProcessNext(): Promise<void> {
    if (this.isRunning || this.isCancelled || this.queue.length === 0) return

    this.isRunning = true
    this.currentTask = this.queue.shift()!

    this.progressTracker.updatePool(this.name, {
      waiting: this.queue.length,
      isProcessing: true,
      currentPage: this.currentTask.imageIndex + 1,
      isWaitingLock: false
    })

    try {
      let result: PipelineTask | null

      if (this.lock) {
        this.progressTracker.updatePool(this.name, { isWaitingLock: true })
        result = await this.lock.withLock(this.name, async () => {
          this.progressTracker.updatePool(this.name, { isWaitingLock: false })
          if (this.isCancelled || !this.currentTask) {
            return null
          }
          return await this.process(this.currentTask!)
        })
      } else {
        result = await this.process(this.currentTask)
      }

      if (!result) {
        return
      }

      this.completedCount++
      this.progressTracker.updatePool(this.name, { completed: this.completedCount })

      if (this.nextPool && result.status !== 'failed' && result.status !== 'buffered') {
        this.nextPool.enqueue(result)
      }

      this.onTaskComplete?.(result)

    } catch (error) {
      this.currentTask.status = 'failed'
      this.currentTask.error = (error as Error).message
      this.onTaskComplete?.(this.currentTask)
    } finally {
      this.currentTask = null
      this.isRunning = false
      this.progressTracker.updatePool(this.name, {
        isProcessing: false,
        currentPage: undefined
      })
      this.tryProcessNext()
    }
  }

  protected abstract process(task: PipelineTask): Promise<PipelineTask>

  getStatus(): PoolStatus {
    return {
      name: this.name,
      icon: this.icon,
      waiting: this.queue.length,
      processing: this.isRunning,
      currentPage: this.currentTask ? this.currentTask.imageIndex + 1 : undefined,
      completed: this.completedCount,
      isWaitingLock: this.lock?.isWaiting(this.name) ?? false,
    }
  }

  cancel(): void {
    this.isCancelled = true
    this.queue = []
  }

  reset(): void {
    this.isCancelled = false
    this.queue = []
    this.currentTask = null
    this.isRunning = false
    this.completedCount = 0
  }

  getCompletedCount(): number {
    return this.completedCount
  }

  getWaitingCount(): number {
    return this.queue.length
  }

  isProcessing(): boolean {
    return this.isRunning
  }
}
