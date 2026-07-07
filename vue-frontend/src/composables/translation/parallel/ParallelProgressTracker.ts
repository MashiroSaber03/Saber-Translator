import { reactive } from 'vue'
import { createInitialParallelProgress, createParallelPoolStatuses } from './progressDefaults'
import type { PoolStatus, ParallelProgress, PoolProgressUpdate } from './types'

export class ParallelProgressTracker {
  private poolStatuses: Map<string, PoolStatus> = new Map()
  private totalPages = 0
  private startTime = 0

  public readonly progress = reactive<ParallelProgress>(createInitialParallelProgress())

  constructor() {
    this.initPools()
  }

  private initPools(): void {
    this.poolStatuses.clear()
    for (const status of createParallelPoolStatuses()) {
      this.poolStatuses.set(status.name, status)
    }
    this.syncToReactive()
  }

  init(totalPages: number): void {
    this.totalPages = totalPages
    this.startTime = Date.now()

    for (const status of this.poolStatuses.values()) {
      status.waiting = 0
      status.processing = false
      status.currentPage = undefined
      status.completed = 0
      status.isWaitingLock = false
    }

    this.progress.totalCompleted = 0
    this.progress.totalFailed = 0
    this.progress.totalPages = totalPages
    this.progress.estimatedTimeRemaining = 0
    this.clearSaveProgress()

    this.syncToReactive()
  }

  updatePool(poolName: string, update: PoolProgressUpdate): void {
    const status = this.poolStatuses.get(poolName)
    if (!status) return

    if (update.waiting !== undefined) status.waiting = update.waiting
    if (update.isProcessing !== undefined) status.processing = update.isProcessing
    if (update.currentPage !== undefined) status.currentPage = update.currentPage
    if (update.completed !== undefined) status.completed = update.completed
    if (update.isWaitingLock !== undefined) status.isWaitingLock = update.isWaitingLock

    this.syncToReactive()
    this.updateEstimatedTime()
  }

  incrementCompleted(): void {
    this.progress.totalCompleted++
    this.updateEstimatedTime()
  }

  startSaveProgress(total: number): void {
    this.progress.save = {
      completed: 0,
      total,
    }
  }

  incrementSaveCompleted(): void {
    if (!this.progress.save) return
    this.progress.save.completed = (this.progress.save.completed || 0) + 1
  }

  clearSaveProgress(): void {
    this.progress.save = undefined
  }

  incrementFailed(): void {
    this.progress.totalFailed++
  }

  private updateEstimatedTime(): void {
    if (this.progress.totalCompleted === 0) {
      this.progress.estimatedTimeRemaining = 0
      return
    }

    const elapsed = (Date.now() - this.startTime) / 1000
    const avgTimePerPage = elapsed / this.progress.totalCompleted
    const remaining = this.totalPages - this.progress.totalCompleted - this.progress.totalFailed
    this.progress.estimatedTimeRemaining = Math.ceil(avgTimePerPage * remaining)
  }

  private syncToReactive(): void {
    this.progress.pools = Array.from(this.poolStatuses.values()).map(s => ({ ...s }))
  }

  getPoolStatus(poolName: string): PoolStatus | undefined {
    return this.poolStatuses.get(poolName)
  }

  getAllPoolStatuses(): PoolStatus[] {
    return Array.from(this.poolStatuses.values())
  }

  getProgress(): ParallelProgress {
    return { ...this.progress }
  }

  reset(): void {
    this.totalPages = 0
    this.startTime = 0
    this.clearSaveProgress()
    this.initPools()
  }

  formatRemainingTime(): string {
    const seconds = this.progress.estimatedTimeRemaining
    if (seconds <= 0) return '--'

    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60

    if (minutes > 0) {
      return `${minutes}分${secs}秒`
    }
    return `${secs}秒`
  }
}

export function useParallelProgressTracker() {
  const tracker = new ParallelProgressTracker()
  return {
    tracker,
    progress: tracker.progress,
  }
}
