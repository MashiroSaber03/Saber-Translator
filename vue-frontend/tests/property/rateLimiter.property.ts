import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import {
  createRateLimitedExecutor,
  createRateLimiter,
  executeBatchWithRateLimit,
} from '@/utils/rateLimiter'

const rpmArb = fc.integer({ min: 0, max: 1000 })
const positiveWindowRpmArb = fc.integer({ min: 1, max: 5 })
const smallTaskArrayArb = fc.array(fc.integer({ min: -1000, max: 1000 }), { minLength: 0, maxLength: 12 })

async function letMicrotasksSettle(): Promise<void> {
  await vi.advanceTimersByTimeAsync(0)
  await Promise.resolve()
}

describe('rate limiter property contracts', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('keeps non-positive RPM unlimited', async () => {
    await fc.assert(
      fc.asyncProperty(fc.integer({ min: -100, max: 0 }), fc.integer({ min: 1, max: 50 }), async (rpm, calls) => {
        const limiter = createRateLimiter(rpm)
        const startedAt = Date.now()

        for (let i = 0; i < calls; i++) {
          await limiter.acquire()
        }

        expect(Date.now()).toBe(startedAt)
        expect(limiter.getRpm()).toBe(rpm)
      }),
      { numRuns: 100 },
    )
  })

  it('updates RPM and clears the window for future acquisitions', async () => {
    await fc.assert(
      fc.asyncProperty(positiveWindowRpmArb, rpmArb, async (initialRpm, nextRpm) => {
        const limiter = createRateLimiter(initialRpm)

        for (let i = 0; i < initialRpm; i++) {
          await limiter.acquire()
        }

        limiter.setRpm(nextRpm)

        expect(limiter.getRpm()).toBe(nextRpm)
        await limiter.acquire()
        expect(Date.now()).toBe(0)
      }),
      { numRuns: 100 },
    )
  })

  it('delays the first request beyond the current minute window', async () => {
    await fc.assert(
      fc.asyncProperty(positiveWindowRpmArb, async (rpm) => {
        const limiter = createRateLimiter(rpm)

        for (let i = 0; i < rpm; i++) {
          await limiter.acquire()
        }

        let settled = false
        const pending = limiter.acquire().then(() => {
          settled = true
        })

        await letMicrotasksSettle()
        expect(settled).toBe(false)

        await vi.advanceTimersByTimeAsync(59999)
        expect(settled).toBe(false)

        await vi.advanceTimersByTimeAsync(1)
        await pending
        expect(settled).toBe(true)
      }),
      { numRuns: 20 },
    )
  })

  it('retries executor work through the limiter until the retry budget is exhausted', async () => {
    const limiter = createRateLimiter(0)
    const executor = createRateLimitedExecutor<number>(limiter, 2)
    let attempts = 0

    const result = await executor(async () => {
      attempts += 1
      if (attempts < 3) {
        throw new Error('transient failure')
      }
      return 42
    })

    expect(result).toBe(42)
    expect(attempts).toBe(3)

    const noRetryExecutor = createRateLimitedExecutor<number>(limiter, 0)
    await expect(noRetryExecutor(async () => {
      throw new Error('permanent failure')
    })).rejects.toThrow('permanent failure')
  })

  it('returns batch results in task order while preserving values and errors', async () => {
    await fc.assert(
      fc.asyncProperty(smallTaskArrayArb, fc.integer({ min: 0, max: 11 }), async (tasks, failingIndex) => {
        const limiter = createRateLimiter(0)
        const progressCalls: Array<[number, number]> = []
        const results = await executeBatchWithRateLimit(
          limiter,
          tasks,
          async (task, index) => {
            if (index === failingIndex && tasks.length > 0) {
              throw new Error(`task ${index} failed`)
            }
            return task * 2
          },
          (completed, total) => {
            progressCalls.push([completed, total])
          },
        )

        expect(results).toHaveLength(tasks.length)
        expect(progressCalls).toEqual(tasks.map((_, index) => [index + 1, tasks.length]))

        for (let index = 0; index < tasks.length; index++) {
          if (index === failingIndex) {
            expect(results[index]).toEqual({
              success: false,
              error: expect.objectContaining({ message: `task ${index} failed` }),
            })
          } else {
            expect(results[index]).toEqual({
              success: true,
              result: tasks[index]! * 2,
            })
          }
        }
      }),
      { numRuns: 100 },
    )
  })

  it('wraps non-Error batch failures as Error instances', async () => {
    const limiter = createRateLimiter(0)

    const [result] = await executeBatchWithRateLimit(limiter, ['task'], async () => {
      throw 'string failure'
    })

    expect(result).toEqual({
      success: false,
      error: expect.objectContaining({ message: 'string failure' }),
    })
  })
})
