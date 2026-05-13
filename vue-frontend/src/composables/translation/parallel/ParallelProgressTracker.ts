/**
 * 并行进度追踪器
 * 
 * 管理并行流水线各个池子的进度状态，用于多进度条显示
 */

import { reactive } from 'vue'
import type { PoolStatus, ParallelProgress, PoolProgressUpdate } from './types'

/**
 * 默认池子配置
 */
const DEFAULT_POOLS: Array<{ name: string; icon: string }> = [
  { name: '检测', icon: '📍' },
  { name: 'OCR', icon: '📖' },
  { name: '颜色', icon: '🎨' },
  { name: '术语', icon: '📚' },
  { name: '翻译', icon: '🌐' },
  { name: '修复', icon: '🖌️' },
  { name: '渲染', icon: '✨' }
]

export class ParallelProgressTracker {
  private poolStatuses: Map<string, PoolStatus> = new Map()
  private totalPages = 0
  private startTime = 0

  // 响应式状态（用于Vue组件）
  public readonly progress = reactive<ParallelProgress>({
    pools: [],
    totalCompleted: 0,
    totalFailed: 0,
    totalPages: 0,
    estimatedTimeRemaining: 0
    // 注意：preSave 和 save 字段由 pipeline.ts 直接在 globalProgress 上管理，
    // 不在这里初始化
  })

  constructor() {
    this.initPools()
  }

  /**
   * 初始化池子状态
   */
  private initPools(): void {
    for (const pool of DEFAULT_POOLS) {
      const status: PoolStatus = {
        name: pool.name,
        icon: pool.icon,
        waiting: 0,
        processing: false,
        completed: 0,
        isWaitingLock: false
      }
      this.poolStatuses.set(pool.name, status)
    }
    this.syncToReactive()
  }

  /**
   * 初始化进度（开始新的翻译任务时调用）
   */
  init(totalPages: number): void {
    this.totalPages = totalPages
    this.startTime = Date.now()

    // 重置所有池子状态
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

    this.syncToReactive()
  }

  /**
   * 更新池子状态
   */
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

  /**
   * 增加完成数
   */
  incrementCompleted(): void {
    this.progress.totalCompleted++
    this.updateEstimatedTime()
  }

  /**
   * 增加失败数
   */
  incrementFailed(): void {
    this.progress.totalFailed++
  }

  /**
   * 更新预计剩余时间
   */
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

  /**
   * 同步到响应式对象
   */
  private syncToReactive(): void {
    this.progress.pools = Array.from(this.poolStatuses.values()).map(s => ({ ...s }))
  }

  /**
   * 获取指定池子状态
   */
  getPoolStatus(poolName: string): PoolStatus | undefined {
    return this.poolStatuses.get(poolName)
  }

  /**
   * 获取所有池子状态
   */
  getAllPoolStatuses(): PoolStatus[] {
    return Array.from(this.poolStatuses.values())
  }

  /**
   * 获取当前进度
   */
  getProgress(): ParallelProgress {
    return { ...this.progress }
  }

  /**
   * 重置
   */
  reset(): void {
    this.totalPages = 0
    this.startTime = 0
    this.initPools()
  }

  /**
   * 格式化剩余时间
   */
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

/**
 * 创建进度追踪器的组合式函数
 */
export function useParallelProgressTracker() {
  const tracker = new ParallelProgressTracker()
  return {
    tracker,
    progress: tracker.progress
  }
}
