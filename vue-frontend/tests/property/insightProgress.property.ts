/**
 * 漫画分析进度状态属性测试
 * 
 * **Feature: vue-frontend-migration, Property 31: 分析进度状态一致性**
 * **Validates: Requirements 6.2**
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type AnalysisStatus } from '@/stores/insightStore'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的分析状态
 */
const analysisStatusArb = fc.constantFrom<AnalysisStatus>('idle', 'running', 'paused', 'completed', 'failed')

/**
 * 生成有效的进度值
 */
const progressArb = fc.record({
  current: fc.integer({ min: 0, max: 1000 }),
  total: fc.integer({ min: 0, max: 1000 })
}).filter(p => p.current <= p.total || p.total === 0)

/**
 * 生成有效的书籍ID
 */
const bookIdArb = fc.stringOf(fc.constantFrom(...'abcdef0123456789'.split('')), { minLength: 8, maxLength: 8 })

// ============================================================
// 属性测试
// ============================================================

describe('漫画分析进度状态属性测试', () => {
  beforeEach(() => {
    // 创建新的 Pinia 实例
    setActivePinia(createPinia())
  })

  /**
   * Property 31: 进度百分比计算正确性
   * 
   * *对于任意* 有效的进度值（current, total），计算的百分比应当在 0-100 范围内
   * 且等于 Math.round((current / total) * 100)
   */
  it('进度百分比计算应当正确', () => {
    fc.assert(
      fc.property(
        progressArb,
        ({ current, total }) => {
          const store = useInsightStore()
          
          // 更新进度
          store.updateProgress(current, total)
          
          // 计算预期百分比
          const expectedPercent = total === 0 ? 0 : Math.round((current / total) * 100)
          
          // 验证计算结果
          expect(store.progressPercent).toBe(expectedPercent)
          
          // 验证百分比在有效范围内
          expect(store.progressPercent).toBeGreaterThanOrEqual(0)
          expect(store.progressPercent).toBeLessThanOrEqual(100)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 31: 进度百分比边界情况
   * 
   * *对于任意* total 为 0 的情况，百分比应当为 0
   */
  it('total 为 0 时百分比应当为 0', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 1000 }),
        (current) => {
          const store = useInsightStore()
          
          // 更新进度，total 为 0
          store.updateProgress(current, 0)
          
          // 验证百分比为 0
          expect(store.progressPercent).toBe(0)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 31: 状态切换正确性
   * 
   * *对于任意* 状态切换序列，store 的状态应当正确反映最后设置的状态
   */
  it('状态切换应当正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(analysisStatusArb, { minLength: 1, maxLength: 10 }),
        (statusSequence) => {
          const store = useInsightStore()
          
          // 依次设置状态
          for (const status of statusSequence) {
            store.setAnalysisStatus(status)
          }
          
          // 验证最终状态
          const lastStatus = statusSequence[statusSequence.length - 1]
          expect(store.analysisStatus).toBe(lastStatus)
          
          // 验证计算属性
          expect(store.isAnalyzing).toBe(lastStatus === 'running')
          expect(store.isAnalysisCompleted).toBe(lastStatus === 'completed')
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 31: 暂停/继续状态切换
   * 
   * *对于任意* 暂停/继续操作序列，状态应当正确切换
   */
  it('暂停/继续状态切换应当正确', () => {
    fc.assert(
      fc.property(
        fc.array(fc.boolean(), { minLength: 1, maxLength: 20 }),
        (pauseSequence) => {
          const store = useInsightStore()
          
          // 初始状态设为运行中
          store.setAnalysisStatus('running')
          
          // 依次执行暂停/继续操作
          for (const shouldPause of pauseSequence) {
            if (shouldPause) {
              store.setAnalysisStatus('paused')
            } else {
              store.setAnalysisStatus('running')
            }
          }
          
          // 验证最终状态
          const lastAction = pauseSequence[pauseSequence.length - 1]
          const expectedStatus = lastAction ? 'paused' : 'running'
          expect(store.analysisStatus).toBe(expectedStatus)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 31: 进度更新不影响状态
   * 
   * *对于任意* 进度更新，分析状态应当保持不变
   */
  it('进度更新不应影响分析状态', () => {
    fc.assert(
      fc.property(
        analysisStatusArb,
        fc.array(progressArb, { minLength: 1, maxLength: 10 }),
        (initialStatus, progressUpdates) => {
          const store = useInsightStore()
          
          // 设置初始状态
          store.setAnalysisStatus(initialStatus)
          
          // 执行多次进度更新
          for (const { current, total } of progressUpdates) {
            store.updateProgress(current, total)
          }
          
          // 验证状态未改变
          expect(store.analysisStatus).toBe(initialStatus)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 31: 重置分析状态
   * 
   * *对于任意* 初始状态和进度，重置后应当恢复到初始状态
   */
  it('重置分析状态应当恢复到初始状态', () => {
    fc.assert(
      fc.property(
        analysisStatusArb,
        progressArb,
        bookIdArb,
        (status, progress, bookId) => {
          const store = useInsightStore()
          
          // 设置各种状态
          store.setCurrentBook(bookId)
          store.setAnalysisStatus(status)
          store.updateProgress(progress.current, progress.total)
          
          // 重置分析状态
          store.resetAnalysis()
          
          // 验证状态已重置
          expect(store.analysisStatus).toBe('idle')
          expect(store.progress.current).toBe(0)
          expect(store.progress.total).toBe(0)
          expect(store.progressPercent).toBe(0)
          
          // 书籍ID 应当保持不变
          expect(store.currentBookId).toBe(bookId)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
