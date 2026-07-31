import { describe, expect, it } from 'vitest'
import type { V2Job } from '@/api/v2/jobs'
import {
  batchProgressCounts,
  batchStatusCounts,
  currentStepLabel,
  describeJobTarget,
  groupJobsByBatch,
  poolProgress,
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

    expect(currentStepLabel(running)).toBe('第 3 项 · translate')
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
