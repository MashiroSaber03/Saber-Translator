export interface RateLimiter {
  acquire: () => Promise<void>
  getRpm: () => number
  setRpm: (rpm: number) => void
  reset: () => void
}

export interface BatchResult<T> {
  success: boolean
  result?: T
  error?: Error
}

export function createRateLimiter(initialRpm: number): RateLimiter {
  let rpm = initialRpm
  let timestamps: number[] = []

  function reset(): void {
    timestamps = []
  }

  async function acquire(): Promise<void> {
    if (rpm <= 0) {
      return
    }

    const now = Date.now()
    timestamps = timestamps.filter((time) => now - time < 60000)

    if (timestamps.length < rpm) {
      timestamps.push(now)
      return
    }

    const oldest = timestamps[0] ?? now
    const waitMs = Math.max(0, 60000 - (now - oldest))

    await new Promise<void>((resolve) => {
      setTimeout(resolve, waitMs)
    })

    return acquire()
  }

  return {
    acquire,
    getRpm: () => rpm,
    setRpm: (nextRpm: number) => {
      rpm = nextRpm
      reset()
    },
    reset,
  }
}

export function createRateLimitedExecutor<T>(
  limiter: RateLimiter,
  maxRetries = 0
): (fn: () => Promise<T>) => Promise<T> {
  return async (fn: () => Promise<T>) => {
    let attempt = 0

    while (true) {
      await limiter.acquire()

      try {
        return await fn()
      } catch (error) {
        if (attempt >= maxRetries) {
          throw error
        }
        attempt += 1
      }
    }
  }
}

export async function executeBatchWithRateLimit<TTask, TResult>(
  limiter: RateLimiter,
  tasks: TTask[],
  worker: (task: TTask, index: number) => Promise<TResult>,
  onProgress?: (completed: number, total: number) => void
): Promise<Array<BatchResult<TResult>>> {
  const results: Array<BatchResult<TResult>> = []

  for (let index = 0; index < tasks.length; index++) {
    const task = tasks[index] as TTask

    try {
      await limiter.acquire()
      results.push({
        success: true,
        result: await worker(task, index),
      })
    } catch (error) {
      results.push({
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
      })
    }

    onProgress?.(index + 1, tasks.length)
  }

  return results
}
