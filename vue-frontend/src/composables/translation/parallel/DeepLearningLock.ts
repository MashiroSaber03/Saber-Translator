interface WaitingTask {
  resolve: () => void
  poolName: string
}

export class DeepLearningLock {
  private currentCount = 0
  private maxCount: number
  private waitQueue: WaitingTask[] = []
  private holders: string[] = []

  constructor(maxCount: number = 1) {
    this.maxCount = Math.max(1, maxCount)
  }

  setSize(size: number): void {
    this.maxCount = Math.max(1, size)
    this.tryWakeUp()
  }

  getSize(): number {
    return this.maxCount
  }

  async acquire(poolName: string): Promise<void> {
    if (this.currentCount < this.maxCount) {
      this.currentCount++
      this.holders.push(poolName)
      return
    }

    return new Promise<void>(resolve => {
      this.waitQueue.push({ resolve, poolName })
    })
  }

  release(poolName: string): void {
    const index = this.holders.indexOf(poolName)
    if (index >= 0) {
      this.holders.splice(index, 1)
      this.currentCount--
    }

    this.tryWakeUp()
  }

  private tryWakeUp(): void {
    while (this.currentCount < this.maxCount && this.waitQueue.length > 0) {
      const next = this.waitQueue.shift()!
      this.currentCount++
      this.holders.push(next.poolName)
      next.resolve()
    }
  }

  async withLock<T>(poolName: string, fn: () => Promise<T>): Promise<T> {
    await this.acquire(poolName)
    try {
      return await fn()
    } finally {
      this.release(poolName)
    }
  }

  getStatus(): {
    isLocked: boolean
    currentCount: number
    maxCount: number
    holders: string[]
    waitingCount: number
    waitingPools: string[]
  } {
    return {
      isLocked: this.currentCount >= this.maxCount,
      currentCount: this.currentCount,
      maxCount: this.maxCount,
      holders: [...this.holders],
      waitingCount: this.waitQueue.length,
      waitingPools: this.waitQueue.map(w => w.poolName),
    }
  }

  isWaiting(poolName: string): boolean {
    return this.waitQueue.some(w => w.poolName === poolName)
  }

  isHolding(poolName: string): boolean {
    return this.holders.includes(poolName)
  }

  reset(): void {
    this.currentCount = 0
    this.holders = []
    for (const task of this.waitQueue) {
      task.resolve()
    }
    this.waitQueue = []
  }
}
