import { describe, expect, it } from 'vitest'
import type { V2Job } from '@/api/v2/jobs'
import {
  batchProgressCounts,
  batchStatusCounts,
  currentStepLabel,
  describeJobTarget,
  groupJobsByBatch,
  poolProgress,
  progressCounts,
  progressPercent,
} from '@/stores/taskCenterProjection'

function job(overrides: Partial<V2Job>): V2Job {
  return {
    jobId: crypto.randomUUID(),
    kind: 'translation',
    retryOfJobId: null,
    retryMode: null,
    status: 'queued',
    queueRank: 1,
    progress: {},
    target: {},
    createdAt: null,
    ...overrides,
  }
}

describe('task center projection', () => {
  it('groups member jobs without hiding standalone jobs', () => {
    const grouped = groupJobsByBatch([
      job({ jobId: 'a', batchId: 'batch', batchDisplayName: 'Batch' }),
      job({ jobId: 'b', batchId: 'batch', batchDisplayName: 'Batch' }),
      job({ jobId: 'c', batchId: null }),
    ])
    expect(grouped).toHaveLength(2)
    expect(grouped[0]?.jobs.map(item => item.jobId)).toEqual(['a', 'b'])
    expect(grouped[1]?.jobs.map(item => item.jobId)).toEqual(['c'])
  })

  it('derives bounded progress from durable item counts', () => {
    expect(progressPercent(job({
      progress: {
        totalItems: 10,
        completedItems: 4,
        failedItems: 1,
        skippedItems: 2,
        cancelledItems: 1,
      },
    }))).toBe(80)
    expect(progressPercent(job({ progress: {} }))).toBe(0)
  })

  it('counts only page items for Insight analysis jobs', () => {
    const running = job({
      kind: 'insight_analysis',
      target: { book: 'Book', pageCount: 18 },
      progress: {
        totalItems: 19,
        completedItems: 6,
        failedItems: 1,
        pools: [
          {
            kind: 'insight_analyze_page',
            total: 18,
            completed: 5,
            failed: 1,
            skipped: 0,
          },
          {
            kind: 'insight_publish_run',
            total: 1,
            completed: 0,
            failed: 0,
            skipped: 0,
          },
        ],
      },
    })
    const queued = job({
      kind: 'insight_analysis',
      target: { book: 'Book', pageCount: 4 },
      progress: { totalItems: 5, completedItems: 0 },
    })

    expect(progressCounts(running)).toEqual({ completed: 6, total: 18 })
    expect(progressCounts(queued)).toEqual({ completed: 0, total: 4 })
  })

  it('aggregates batch progress and status distribution from backend snapshots', () => {
    const jobs = [
      job({
        status: 'completed',
        progress: { totalItems: 4, completedItems: 4 },
      }),
      job({
        status: 'completed_with_errors',
        progress: { totalItems: 3, completedItems: 1, failedItems: 1 },
      }),
    ]

    expect(batchProgressCounts(jobs)).toEqual({ completed: 6, total: 7 })
    expect(batchStatusCounts(jobs)).toEqual([
      ['completed', 1],
      ['completed_with_errors', 1],
    ])
  })

  it('projects current step and parallel pool state without rebuilding progress client-side', () => {
    const running = job({
      status: 'running',
      progress: {
        executionMode: 'parallel',
        currentStep: { itemOrdinal: 3, kind: 'translate' },
        pools: [
          {
            kind: 'translate',
            waiting: 2,
            processing: 1,
            completed: 4,
            lockWaiting: true,
          },
        ],
      },
    })

    expect(currentStepLabel(running)).toBe('第 3 项 · 文本翻译')
    expect(poolProgress(running)).toEqual([
      {
        kind: 'translate',
        waiting: 2,
        processing: 1,
        completed: 4,
        lockWaiting: true,
      },
    ])
  })

  it('labels numeric book and chapter names without looking like task counts', () => {
    const numericTarget = job({
      batchId: 'batch',
      batchDisplayName: '1 / 3',
      target: { book: '1', chapter: '3', pageCount: 1 },
    })

    expect(groupJobsByBatch([numericTarget])[0]?.displayName).toBe(
      '书籍：1 · 章节：3',
    )
    expect(describeJobTarget(numericTarget)).toBe('章节：3 · 1 页')
  })
})
