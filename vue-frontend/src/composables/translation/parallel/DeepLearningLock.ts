/**
 * 深度学习模型互斥锁
 * 
 * 用于控制检测、OCR、颜色提取、背景修复这些深度学习操作的并发数
 * 避免GPU/CPU资源竞争导致内存溢出
 * 
 * 锁的大小可配置（默认1，即互斥锁）
 */

interface WaitingTask {
  resolve: () => void
  poolName: string
}

export class DeepLearningLock {
  private currentCount = 0
  private maxCount: number
  private waitQueue: WaitingTask[] = []
  private holders: string[] = []

  /**
   * 创建深度学习锁
   * @param maxCount 最大并发数（默认1，即互斥锁）
   */
  constructor(maxCount: number = 1) {
    this.maxCount = Math.max(1, maxCount)
  }

  /**
   * 设置锁的大小
   * @param size 新的并发大小
   */
  setSize(size: number): void {
    this.maxCount = Math.max(1, size)
    // 如果新大小大于当前占用数，尝试唤醒等待者
    this.tryWakeUp()
  }

  /**
   * 获取当前锁大小
   */
  getSize(): number {
    return this.maxCount
  }

  /**
   * 获取锁
   * @param poolName 请求锁的池子名称
   */
  async acquire(poolName: string): Promise<void> {
    if (this.currentCount < this.maxCount) {
      this.currentCount++
      this.holders.push(poolName)
      return
    }

    // 排队等待
    return new Promise<void>(resolve => {
      this.waitQueue.push({ resolve, poolName })
    })
  }

  /**
   * 释放锁
   * @param poolName 释放锁的池子名称
   */
  release(poolName: string): void {
    const index = this.holders.indexOf(poolName)
    if (index >= 0) {
      this.holders.splice(index, 1)
      this.currentCount--
    }

    this.tryWakeUp()
  }

  /**
   * 尝试唤醒等待者
   */
  private tryWakeUp(): void {
    while (this.currentCount < this.maxCount && this.waitQueue.length > 0) {
      const next = this.waitQueue.shift()!
      this.currentCount++
      this.holders.push(next.poolName)
      next.resolve()
    }
  }

  /**
   * 带锁执行（自动获取和释放）
   * @param poolName 池子名称
   * @param fn 要执行的异步函数
   */
  async withLock<T>(poolName: string, fn: () => Promise<T>): Promise<T> {
    await this.acquire(poolName)
    try {
      return await fn()
    } finally {
      this.release(poolName)
    }
  }

  /**
   * 获取锁状态
   */
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
      waitingPools: this.waitQueue.map(w => w.poolName)
    }
  }

  /**
   * 检查指定池子是否在等待锁
   */
  isWaiting(poolName: string): boolean {
    return this.waitQueue.some(w => w.poolName === poolName)
  }

  /**
   * 检查指定池子是否持有锁
   */
  isHolding(poolName: string): boolean {
    return this.holders.includes(poolName)
  }

  /**
   * 重置锁状态
   */
  reset(): void {
    this.currentCount = 0
    this.holders = []
    // 唤醒所有等待者（让它们能够正常退出）
    for (const task of this.waitQueue) {
      task.resolve()
    }
    this.waitQueue = []
  }
}
