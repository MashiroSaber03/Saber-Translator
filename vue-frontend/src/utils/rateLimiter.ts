/**
 * RPM 限速器工具函数
 * 用于翻译服务和 AI 视觉 OCR 的请求限速
 */

/**
 * 限速器接口
 */
export interface RateLimiter {
  /** 等待直到可以执行下一个请求 */
  acquire(): Promise<void>
  /** 重置限速器状态 */
  reset(): void
  /** 获取当前 RPM 限制 */
  getRpm(): number
  /** 设置新的 RPM 限制 */
  setRpm(rpm: number): void
}

/**
 * 创建 RPM 限速器
 * @param rpm - 每分钟请求数限制，0 表示无限制
 * @returns 限速器实例
 */
export function createRateLimiter(rpm: number): RateLimiter {
  let currentRpm = rpm
  let lastRequestTime = 0
  let requestCount = 0
  let windowStart = Date.now()

  // 计算请求间隔（毫秒）
  const getInterval = (): number => {
    if (currentRpm <= 0) return 0
    return Math.ceil(60000 / currentRpm)
  }

  return {
    async acquire(): Promise<void> {
      // 如果 RPM 为 0，表示无限制，直接返回
      if (currentRpm <= 0) {
        return
      }

      const now = Date.now()
      const interval = getInterval()

      // 检查是否需要重置窗口
      if (now - windowStart >= 60000) {
        windowStart = now
        requestCount = 0
      }

      // 检查是否达到限制
      if (requestCount >= currentRpm) {
        // 等待到下一个窗口
        const waitTime = 60000 - (now - windowStart)
        if (waitTime > 0) {
          await sleep(waitTime)
        }
        windowStart = Date.now()
        requestCount = 0
      }

      // 检查距离上次请求的时间
      const timeSinceLastRequest = now - lastRequestTime
      if (timeSinceLastRequest < interval) {
        await sleep(interval - timeSinceLastRequest)
      }

      lastRequestTime = Date.now()
      requestCount++
    },

    reset(): void {
      lastRequestTime = 0
      requestCount = 0
      windowStart = Date.now()
    },

    getRpm(): number {
      return currentRpm
    },

    setRpm(rpm: number): void {
      currentRpm = rpm
      // 重置状态以应用新的限制
      this.reset()
    }
  }
}

/**
 * 睡眠函数
 * @param ms - 毫秒数
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * 创建带重试的限速执行器
 * @param limiter - 限速器实例
 * @param maxRetries - 最大重试次数
 * @returns 执行函数
 */
export function createRateLimitedExecutor<T>(
  limiter: RateLimiter,
  maxRetries: number = 3
): (fn: () => Promise<T>) => Promise<T> {
  return async (fn: () => Promise<T>): Promise<T> => {
    let lastError: Error | null = null

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        // 等待限速器许可
        await limiter.acquire()
        // 执行函数
        return await fn()
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error))

        // 如果是最后一次尝试，抛出错误
        if (attempt === maxRetries) {
          throw lastError
        }

        // 等待一段时间后重试（指数退避）
        const backoffTime = Math.min(1000 * Math.pow(2, attempt), 10000)
        await sleep(backoffTime)
      }
    }

    // 理论上不会到达这里，但 TypeScript 需要返回值
    throw lastError || new Error('未知错误')
  }
}

/**
 * 批量执行带限速的请求
 * @param limiter - 限速器实例
 * @param tasks - 任务数组
 * @param onProgress - 进度回调
 * @returns 执行结果数组
 */
export async function executeBatchWithRateLimit<T, R>(
  limiter: RateLimiter,
  tasks: T[],
  executor: (task: T, index: number) => Promise<R>,
  onProgress?: (completed: number, total: number) => void
): Promise<Array<{ success: boolean; result?: R; error?: Error }>> {
  const results: Array<{ success: boolean; result?: R; error?: Error }> = []

  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i] as T
    try {
      await limiter.acquire()
      const result = await executor(task, i)
      results.push({ success: true, result })
    } catch (error) {
      results.push({
        success: false,
        error: error instanceof Error ? error : new Error(String(error))
      })
    }

    onProgress?.(i + 1, tasks.length)
  }

  return results
}
