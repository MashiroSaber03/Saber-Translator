/**
 * 结果收集器
 * 
 * 收集渲染池完成的任务，确保按顺序输出结果
 */

import type { PipelineTask } from './types'

export class ResultCollector {
  private results: Map<number, PipelineTask> = new Map()
  private totalExpected = 0
  private completedCount = 0
  private failedCount = 0
  private resolveWaitAll: ((value: { success: number; failed: number }) => void) | null = null

  /**
   * 初始化收集器
   */
  init(totalExpected: number): void {
    this.results.clear()
    this.totalExpected = totalExpected
    this.completedCount = 0
    this.failedCount = 0
    this.resolveWaitAll = null
  }

  /**
   * 添加完成的任务
   */
  add(task: PipelineTask): void {
    this.results.set(task.imageIndex, task)
    
    if (task.status === 'completed') {
      this.completedCount++
    } else if (task.status === 'failed') {
      this.failedCount++
    }

    // 检查是否全部完成
    if (this.completedCount + this.failedCount >= this.totalExpected) {
      if (this.resolveWaitAll) {
        this.resolveWaitAll({
          success: this.completedCount,
          failed: this.failedCount
        })
        this.resolveWaitAll = null
      }
    }
  }

  /**
   * 等待所有结果
   */
  waitForAll(totalExpected: number): Promise<{ success: number; failed: number }> {
    this.totalExpected = totalExpected
    
    // 如果已经全部完成，直接返回
    if (this.completedCount + this.failedCount >= totalExpected) {
      return Promise.resolve({
        success: this.completedCount,
        failed: this.failedCount
      })
    }

    // 否则等待
    return new Promise(resolve => {
      this.resolveWaitAll = resolve
    })
  }

  /**
   * 获取指定索引的结果
   */
  get(imageIndex: number): PipelineTask | undefined {
    return this.results.get(imageIndex)
  }

  /**
   * 获取所有结果（按索引排序）
   */
  getAll(): PipelineTask[] {
    return Array.from(this.results.values())
      .sort((a, b) => a.imageIndex - b.imageIndex)
  }

  /**
   * 获取成功的结果
   */
  getSuccessful(): PipelineTask[] {
    return this.getAll().filter(t => t.status === 'completed')
  }

  /**
   * 获取失败的结果
   */
  getFailed(): PipelineTask[] {
    return this.getAll().filter(t => t.status === 'failed')
  }

  /**
   * 获取统计
   */
  getStats(): { total: number; completed: number; failed: number; pending: number } {
    return {
      total: this.totalExpected,
      completed: this.completedCount,
      failed: this.failedCount,
      pending: this.totalExpected - this.completedCount - this.failedCount
    }
  }

  /**
   * 重置
   */
  reset(): void {
    this.results.clear()
    this.totalExpected = 0
    this.completedCount = 0
    this.failedCount = 0
    this.resolveWaitAll = null
  }
}
