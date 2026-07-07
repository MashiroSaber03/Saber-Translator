import type { PipelineTask } from './types'

export class ResultCollector {
  private results: Map<number, PipelineTask> = new Map()
  private totalExpected = 0
  private completedCount = 0
  private failedCount = 0
  private resolveWaitAll: ((value: { success: number; failed: number }) => void) | null = null
  private isClosed = false

  init(totalExpected: number): void {
    this.results.clear()
    this.totalExpected = totalExpected
    this.completedCount = 0
    this.failedCount = 0
    this.resolveWaitAll = null
    this.isClosed = false
  }

  add(task: PipelineTask): void {
    if (this.isClosed) {
      return
    }

    if (this.results.has(task.imageIndex)) {
      return
    }

    this.results.set(task.imageIndex, task)

    if (task.status === 'completed') {
      this.completedCount++
    } else if (task.status === 'failed') {
      this.failedCount++
    }

    if (this.completedCount + this.failedCount >= this.totalExpected) {
      this.isClosed = true
      if (this.resolveWaitAll) {
        this.resolveWaitAll({
          success: this.completedCount,
          failed: this.failedCount,
        })
        this.resolveWaitAll = null
      }
    }
  }

  waitForAll(totalExpected: number): Promise<{ success: number; failed: number }> {
    this.totalExpected = totalExpected

    if (this.completedCount + this.failedCount >= totalExpected) {
      return Promise.resolve({
        success: this.completedCount,
        failed: this.failedCount,
      })
    }

    return new Promise(resolve => {
      this.resolveWaitAll = resolve
    })
  }

  finishEarly(): { success: number; failed: number } {
    this.isClosed = true
    const summary = {
      success: this.completedCount,
      failed: this.failedCount,
    }

    if (this.resolveWaitAll) {
      this.resolveWaitAll(summary)
      this.resolveWaitAll = null
    }

    return summary
  }

  get(imageIndex: number): PipelineTask | undefined {
    return this.results.get(imageIndex)
  }

  getAll(): PipelineTask[] {
    return Array.from(this.results.values())
      .sort((a, b) => a.imageIndex - b.imageIndex)
  }

  getSuccessful(): PipelineTask[] {
    return this.getAll().filter(t => t.status === 'completed')
  }

  getFailed(): PipelineTask[] {
    return this.getAll().filter(t => t.status === 'failed')
  }

  getStats(): { total: number; completed: number; failed: number; pending: number } {
    return {
      total: this.totalExpected,
      completed: this.completedCount,
      failed: this.failedCount,
      pending: this.totalExpected - this.completedCount - this.failedCount,
    }
  }

  reset(): void {
    this.results.clear()
    this.totalExpected = 0
    this.completedCount = 0
    this.failedCount = 0
    this.resolveWaitAll = null
    this.isClosed = false
  }
}
