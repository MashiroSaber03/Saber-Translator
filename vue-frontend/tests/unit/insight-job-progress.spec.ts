import { describe, expect, it } from 'vitest'
import type { V2Job } from '@/api/v2/jobs'
import {
  projectInsightPageProgress,
  projectTerminalInsightPageProgress,
} from '@/utils/insightJobProgress'

function progress(): V2Job['progress'] {
  return {
    executionMode: 'sequential',
    jobStatus: 'running',
    totalItems: 2,
    completedItems: 0,
    failedItems: 0,
    skippedItems: 0,
    cancelledItems: 0,
    pools: [
      {
        kind: 'insight_analyze_page',
        total: 1,
        completed: 0,
        failed: 0,
        skipped: 0,
        waiting: 0,
        processing: 1,
        lockWaiting: false,
        current: [],
      },
      {
        kind: 'insight_publish_run',
        total: 1,
        completed: 0,
        failed: 0,
        skipped: 0,
        waiting: 1,
        processing: 0,
        lockWaiting: false,
        current: [],
      },
    ],
    currentStep: {
      itemId: 'item-1',
      pageId: 'page-1',
      itemOrdinal: 0,
      stepId: 'step-1',
      stepOrdinal: 0,
      kind: 'insight_analyze_page',
    },
  }
}

describe('projectInsightPageProgress', () => {
  it('counts analyzed pages without treating the publish step as another page', () => {
    expect(projectInsightPageProgress(progress())).toEqual({
      current: 0,
      total: 1,
      currentStepKind: 'insight_analyze_page',
    })
  })

  it('counts terminal page outcomes as processed progress', () => {
    const value = progress()
    const pagePool = value.pools[0]
    if (!pagePool) throw new Error('missing page pool')
    pagePool.processing = 0
    pagePool.completed = 1

    expect(projectInsightPageProgress(value).current).toBe(1)
  })

  it('closes finished progress even when the terminal event precedes the refreshed job snapshot', () => {
    const stale = progress()

    expect(projectTerminalInsightPageProgress(stale, 'job_finished')).toEqual({
      current: 1,
      total: 1,
      currentStepKind: '',
    })
    expect(projectTerminalInsightPageProgress(stale, 'job_failed')).toEqual({
      current: 0,
      total: 1,
      currentStepKind: '',
    })
  })
})
