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

function progress(overrides: Partial<V2Job['progress']> = {}): V2Job['progress'] {
  return {
    executionMode: 'sequential',
    jobStatus: 'queued',
    totalItems: 0,
    completedItems: 0,
    failedItems: 0,
    skippedItems: 0,
    cancelledItems: 0,
    pools: [],
    ...overrides,
  }
}

function job(overrides: Partial<V2Job>): V2Job {
  return {
    jobId: crypto.randomUUID(),
    kind: 'translation',
    retryOfJobId: null,
    retryMode: null,
    status: 'queued',
    queueRank: 1,
    progress: progress(),
    target: {},
    createdAt: '2026-08-23T04:00:00Z',
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
    expect(
      progressPercent(
        job({
          progress: progress({
            totalItems: 10,
            completedItems: 4,
            failedItems: 1,
            skippedItems: 2,
            cancelledItems: 1,
          }),
        })
      )
    ).toBe(80)
    expect(progressPercent(job({ progress: progress() }))).toBe(0)
  })

  it('keeps small non-zero progress visible for large jobs', () => {
    const percent = progressPercent(
      job({
        progress: progress({
          totalItems: 2702,
          completedItems: 8,
        }),
      })
    )

    expect(percent).toBeCloseTo((8 / 2702) * 100)
    expect(percent).toBeGreaterThan(0)
  })

  it('counts only page items for Insight analysis jobs', () => {
    const running = job({
      kind: 'insight_analysis',
      target: { book: 'Book', pageCount: 18 },
      progress: progress({
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
            cancelled: 0,
            waiting: 12,
            processing: 0,
            lockWaiting: false,
            current: [],
          },
          {
            kind: 'insight_publish_run',
            total: 1,
            completed: 0,
            failed: 0,
            skipped: 0,
            cancelled: 0,
            waiting: 1,
            processing: 0,
            lockWaiting: false,
            current: [],
          },
        ],
      }),
    })
    const queued = job({
      kind: 'insight_analysis',
      target: { book: 'Book', pageCount: 4 },
      progress: progress({
        totalItems: 5,
      }),
    })

    expect(progressCounts(running)).toEqual({ completed: 6, total: 18 })
    expect(progressCounts(queued)).toEqual({ completed: 0, total: 5 })
  })

  it('aggregates batch progress and status distribution from backend snapshots', () => {
    const jobs = [
      job({
        status: 'completed',
        progress: progress({ totalItems: 4, completedItems: 4 }),
      }),
      job({
        status: 'completed_with_errors',
        progress: progress({ totalItems: 3, completedItems: 1, failedItems: 1 }),
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
      progress: progress({
        executionMode: 'parallel',
        currentStep: {
          itemId: 'item-3',
          pageId: 'page-3',
          itemOrdinal: 3,
          stepId: 'step-3',
          stepOrdinal: 4,
          kind: 'translate',
        },
        pools: [
          {
            kind: 'translate',
            total: 8,
            waiting: 2,
            processing: 1,
            completed: 4,
            failed: 1,
            skipped: 0,
            cancelled: 0,
            lockWaiting: true,
            current: [{
              itemId: 'item-3',
              pageId: 'page-3',
              itemOrdinal: 3,
              stepId: 'step-3',
              stepOrdinal: 4,
            }],
          },
        ],
      }),
    })

    expect(currentStepLabel(running)).toBe('第 3 项 · 文本翻译')
    expect(poolProgress(running)).toEqual([
      {
        kind: 'translate',
        total: 8,
        waiting: 2,
        processing: 1,
        completed: 4,
        failed: 1,
        skipped: 0,
        cancelled: 0,
        lockWaiting: true,
      },
    ])
  })

  it('does not coerce string target counts into current page totals', () => {
    const target = job({ target: { chapter: 'Chapter', pageCount: '18' } })

    expect(describeJobTarget(target)).toBe('章节：Chapter')
  })

  it('labels numeric book and chapter names without looking like task counts', () => {
    const numericTarget = job({
      batchId: 'batch',
      batchDisplayName: '1 / 3',
      target: { book: '1', chapter: '3', pageCount: 1 },
    })

    expect(groupJobsByBatch([numericTarget])[0]?.displayName).toBe('书籍：1 · 章节：3')
    expect(describeJobTarget(numericTarget)).toBe('章节：3 · 1 页')
  })
})
