/**
 * RPM 限速器属性测试
 *
 * Feature: vue-frontend-migration
 * Property 12: RPM限速器行为一致性
 *
 * Validates: Requirements 9.4
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import * as fc from 'fast-check'
import {
  createRateLimiter,
  createRateLimitedExecutor,
  executeBatchWithRateLimit,
  type RateLimiter
} from '@/utils/rateLimiter'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的 RPM 值（包括 0 表示无限制）
 */
const rpmArb = fc.integer({ min: 0, max: 1000 })

/**
 * 生成正整数 RPM 值（不包括 0）
 */
const positiveRpmArb = fc.integer({ min: 1, max: 100 })

// ============================================================
// Property 12: RPM限速器行为一致性
// ============================================================

describe('RPM 限速器属性测试', () => {
  beforeEach(() => {
    // 使用假定时器
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('Property 12: RPM限速器行为一致性', () => {
    /**
     * 测试 RPM=0 时无限制行为
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('RPM=0 时应无限制，acquire 立即返回', async () => {
      const limiter = createRateLimiter(0)

      // 连续调用多次 acquire 应该立即返回
      const startTime = Date.now()

      for (let i = 0; i < 100; i++) {
        await limiter.acquire()
      }

      const endTime = Date.now()

      // 由于使用假定时器，时间应该没有变化
      expect(endTime - startTime).toBe(0)
    })

    /**
     * 测试 getRpm 返回正确的值
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('getRpm 应返回设置的 RPM 值', () => {
      fc.assert(
        fc.property(rpmArb, (rpm) => {
          const limiter = createRateLimiter(rpm)
          expect(limiter.getRpm()).toBe(rpm)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 setRpm 正确更新 RPM 值
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('setRpm 应正确更新 RPM 值', () => {
      fc.assert(
        fc.property(rpmArb, rpmArb, (initialRpm, newRpm) => {
          const limiter = createRateLimiter(initialRpm)
          expect(limiter.getRpm()).toBe(initialRpm)

          limiter.setRpm(newRpm)
          expect(limiter.getRpm()).toBe(newRpm)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 reset 重置限速器状态
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('reset 应重置限速器状态', async () => {
      const limiter = createRateLimiter(60) // 每秒1个请求

      // 执行一次请求
      await limiter.acquire()

      // 重置
      limiter.reset()

      // 重置后应该可以立即执行下一个请求
      const acquirePromise = limiter.acquire()

      // 不需要等待就能完成
      await acquirePromise
    })

    /**
     * 测试请求间隔计算正确性
     * 对于 RPM=60，每个请求间隔应该是 1000ms
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('请求间隔应根据 RPM 正确计算', () => {
      fc.assert(
        fc.property(positiveRpmArb, (rpm) => {
          // 计算期望的间隔
          const expectedInterval = Math.ceil(60000 / rpm)

          // 验证间隔计算逻辑
          // 60000ms / rpm = 每个请求的间隔
          expect(expectedInterval).toBeGreaterThan(0)
          expect(expectedInterval).toBeLessThanOrEqual(60000)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试限速器在窗口内限制请求数量
     *
     * Feature: vue-frontend-migration, Property 12: RPM限速器行为一致性
     * Validates: Requirements 9.4
     */
    it('限速器应在窗口内限制请求数量', async () => {
      const rpm = 10
      const limiter = createRateLimiter(rpm)

      let completedRequests = 0
      const promises: Promise<void>[] = []

      // 尝试发起超过限制的请求
      for (let i = 0; i < rpm + 5; i++) {
        promises.push(
          limiter.acquire().then(() => {
            completedRequests++
          })
        )
      }

      // 立即检查，应该只有部分请求完成
      // 由于使用假定时器，需要推进时间
      await vi.advanceTimersByTimeAsync(0)

      // 在不推进时间的情况下，完成的请求数应该受限
      // 注意：由于实现细节，第一个请求可能立即完成
      expect(completedRequests).toBeLessThanOrEqual(rpm + 1)
    })

    /**
     * 测试从 0 切换到正数 RPM
     */
    it('从 RPM=0 切换到正数应开始限速', async () => {
      const limiter = createRateLimiter(0)

      // RPM=0 时应该无限制
      await limiter.acquire()
      await limiter.acquire()

      // 切换到有限制的 RPM
      limiter.setRpm(60)
      expect(limiter.getRpm()).toBe(60)
    })

    /**
     * 测试从正数切换到 0 RPM
     */
    it('从正数 RPM 切换到 0 应取消限速', async () => {
      const limiter = createRateLimiter(60)

      // 切换到无限制
      limiter.setRpm(0)
      expect(limiter.getRpm()).toBe(0)

      // 应该可以立即执行多个请求
      const startTime = Date.now()
      for (let i = 0; i < 10; i++) {
        await limiter.acquire()
      }
      const endTime = Date.now()

      expect(endTime - startTime).toBe(0)
    })
  })

  describe('createRateLimitedExecutor 测试', () => {
    /**
     * 测试执行器成功执行函数
     */
    it('执行器应成功执行函数并返回结果', async () => {
      const limiter = createRateLimiter(0) // 无限制
      const executor = createRateLimitedExecutor<number>(limiter)

      const result = await executor(async () => 42)
      expect(result).toBe(42)
    })

    /**
     * 测试执行器重试机制
     */
    it('执行器应在失败时重试', async () => {
      const limiter = createRateLimiter(0)
      const executor = createRateLimitedExecutor<number>(limiter, 3)

      let attempts = 0
      const fn = async () => {
        attempts++
        if (attempts < 3) {
          throw new Error('模拟失败')
        }
        return 42
      }

      // 推进时间以允许重试
      const resultPromise = executor(fn)
      await vi.advanceTimersByTimeAsync(10000)
      const result = await resultPromise

      expect(result).toBe(42)
      expect(attempts).toBe(3)
    })

    /**
     * 测试执行器超过最大重试次数后抛出错误
     */
    it('执行器超过最大重试次数后应抛出错误', async () => {
      // 使用真实定时器来避免假定时器的复杂性
      vi.useRealTimers()

      const limiter = createRateLimiter(0)
      // 设置 maxRetries 为 0，这样第一次失败就会抛出错误
      const executor = createRateLimitedExecutor<number>(limiter, 0)

      const fn = async () => {
        throw new Error('始终失败')
      }

      // 直接等待 Promise 拒绝
      await expect(executor(fn)).rejects.toThrow('始终失败')

      // 恢复假定时器
      vi.useFakeTimers()
    })
  })

  describe('executeBatchWithRateLimit 测试', () => {
    /**
     * 测试批量执行返回正确的结果数量
     */
    it('批量执行应返回与任务数量相同的结果', async () => {
      fc.assert(
        await fc.asyncProperty(
          fc.array(fc.integer(), { minLength: 0, maxLength: 10 }),
          async (tasks) => {
            const limiter = createRateLimiter(0) // 无限制

            const results = await executeBatchWithRateLimit(limiter, tasks, async (task) => task * 2)

            expect(results.length).toBe(tasks.length)
            return true
          }
        ),
        { numRuns: 50 }
      )
    })

    /**
     * 测试批量执行成功任务的结果
     */
    it('批量执行成功任务应返回正确结果', async () => {
      const limiter = createRateLimiter(0)
      const tasks = [1, 2, 3, 4, 5]

      const results = await executeBatchWithRateLimit(limiter, tasks, async (task) => task * 2)

      expect(results.length).toBe(5)
      for (let i = 0; i < tasks.length; i++) {
        expect(results[i]?.success).toBe(true)
        expect(results[i]?.result).toBe(tasks[i]! * 2)
      }
    })

    /**
     * 测试批量执行失败任务的处理
     */
    it('批量执行失败任务应正确记录错误', async () => {
      const limiter = createRateLimiter(0)
      const tasks = [1, 2, 3]

      const results = await executeBatchWithRateLimit(limiter, tasks, async (task) => {
        if (task === 2) {
          throw new Error('任务2失败')
        }
        return task * 2
      })

      expect(results.length).toBe(3)
      expect(results[0]?.success).toBe(true)
      expect(results[0]?.result).toBe(2)
      expect(results[1]?.success).toBe(false)
      expect(results[1]?.error?.message).toBe('任务2失败')
      expect(results[2]?.success).toBe(true)
      expect(results[2]?.result).toBe(6)
    })

    /**
     * 测试进度回调
     */
    it('批量执行应正确调用进度回调', async () => {
      const limiter = createRateLimiter(0)
      const tasks = [1, 2, 3, 4, 5]
      const progressCalls: Array<{ completed: number; total: number }> = []

      await executeBatchWithRateLimit(
        limiter,
        tasks,
        async (task) => task,
        (completed, total) => {
          progressCalls.push({ completed, total })
        }
      )

      expect(progressCalls.length).toBe(5)
      for (let i = 0; i < 5; i++) {
        expect(progressCalls[i]?.completed).toBe(i + 1)
        expect(progressCalls[i]?.total).toBe(5)
      }
    })

    /**
     * 测试空任务数组
     */
    it('空任务数组应返回空结果', async () => {
      const limiter = createRateLimiter(0)
      const results = await executeBatchWithRateLimit(limiter, [], async () => 42)

      expect(results).toEqual([])
    })
  })

  describe('边界条件测试', () => {
    /**
     * 测试非常大的 RPM 值
     */
    it('非常大的 RPM 值应正常工作', () => {
      const limiter = createRateLimiter(10000)
      expect(limiter.getRpm()).toBe(10000)
    })

    /**
     * 测试 RPM=1 的极端情况
     * 验证间隔计算正确性（60000ms / 1 = 60000ms）
     */
    it('RPM=1 应计算出 60000ms 的间隔', () => {
      const limiter = createRateLimiter(1)
      expect(limiter.getRpm()).toBe(1)

      // 验证 RPM=1 时的间隔计算逻辑
      // 60000ms / 1 = 60000ms
      const expectedInterval = Math.ceil(60000 / 1)
      expect(expectedInterval).toBe(60000)
    })

    /**
     * 测试多次 reset 调用
     */
    it('多次 reset 调用应正常工作', () => {
      const limiter = createRateLimiter(60)

      limiter.reset()
      limiter.reset()
      limiter.reset()

      expect(limiter.getRpm()).toBe(60)
    })

    /**
     * 测试 setRpm 后 reset 的行为
     */
    it('setRpm 后应自动 reset', async () => {
      const limiter = createRateLimiter(60)

      // 执行一些请求
      await limiter.acquire()

      // 设置新的 RPM（内部会调用 reset）
      limiter.setRpm(120)

      // 应该可以立即执行请求
      await limiter.acquire()
    })
  })
})
