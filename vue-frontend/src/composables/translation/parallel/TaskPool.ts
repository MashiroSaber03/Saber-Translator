/**
 * 任务池基类
 * 
 * 所有具体的池子（检测池、OCR池等）都继承自此类
 * - 无容量限制，任务自然排队
 * - 串行处理（一次处理一个任务）
 * - 支持可选的深度学习锁
 */

import type { PipelineTask } from './types'
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
  protected icon: string
  protected lock: DeepLearningLock | null
  protected progressTracker: ParallelProgressTracker
  protected onTaskComplete?: (task: PipelineTask) => void

  constructor(
    name: string,
    icon: string,
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

  /**
   * 获取池子名称
   */
  getName(): string {
    return this.name
  }

  /**
   * 获取池子图标
   */
  getIcon(): string {
    return this.icon
  }

  /**
   * 设置下一个池子
   */
  setNextPool(pool: TaskPool | null): void {
    this.nextPool = pool
  }

  /**
   * 添加任务到队列
   */
  enqueue(task: PipelineTask): void {
    if (this.isCancelled) return
    this.queue.push(task)
    this.progressTracker.updatePool(this.name, { waiting: this.queue.length })
    this.tryProcessNext()
  }

  /**
   * 批量添加任务
   */
  enqueueBatch(tasks: PipelineTask[]): void {
    if (this.isCancelled) return
    for (const task of tasks) {
      this.queue.push(task)
    }
    this.progressTracker.updatePool(this.name, { waiting: this.queue.length })
    this.tryProcessNext()
  }

  /**
   * 尝试处理下一个任务
   */
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
      let result: PipelineTask

      if (this.lock) {
        // 需要深度学习锁
        this.progressTracker.updatePool(this.name, { isWaitingLock: true })
        result = await this.lock.withLock(this.name, async () => {
          this.progressTracker.updatePool(this.name, { isWaitingLock: false })
          return await this.process(this.currentTask!)
        })
      } else {
        result = await this.process(this.currentTask)
      }

      this.completedCount++
      this.progressTracker.updatePool(this.name, { completed: this.completedCount })

      // 传递给下一个池子
      // 注意：status为'buffered'表示任务被缓冲等待批量处理，不自动传递
      if (this.nextPool && result.status !== 'failed' && result.status !== 'buffered') {
        this.nextPool.enqueue(result)
      }

      this.onTaskComplete?.(result)

    } catch (error) {
      this.currentTask.status = 'failed'
      this.currentTask.error = (error as Error).message
      console.error(`[${this.name}] 处理任务失败:`, error)
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

  /**
   * 子类实现具体处理逻辑
   */
  protected abstract process(task: PipelineTask): Promise<PipelineTask>

  /**
   * 获取池子状态
   */
  getStatus(): {
    name: string
    icon: string
    waiting: number
    processing: boolean
    currentPage?: number
    completed: number
    isWaitingLock: boolean
  } {
    return {
      name: this.name,
      icon: this.icon,
      waiting: this.queue.length,
      processing: this.isRunning,
      currentPage: this.currentTask?.imageIndex,
      completed: this.completedCount,
      isWaitingLock: this.lock?.isWaiting(this.name) ?? false
    }
  }

  /**
   * 取消所有任务
   */
  cancel(): void {
    this.isCancelled = true
    this.queue = []
  }

  /**
   * 重置池子
   */
  reset(): void {
    this.isCancelled = false
    this.queue = []
    this.currentTask = null
    this.isRunning = false
    this.completedCount = 0
  }

  /**
   * 获取已完成数量
   */
  getCompletedCount(): number {
    return this.completedCount
  }

  /**
   * 获取等待数量
   */
  getWaitingCount(): number {
    return this.queue.length
  }

  /**
   * 是否正在处理
   */
  isProcessing(): boolean {
    return this.isRunning
  }
}
