import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { V2Job, V2JobDetail } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  list: vi.fn(),
  snapshot: vi.fn(),
  reorder: vi.fn(),
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    get: mocks.get,
    list: mocks.list,
    snapshot: mocks.snapshot,
    reorder: mocks.reorder,
  },
}))

class FakeEventSource {
  static latest: FakeEventSource | null = null

  onopen: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(readonly url: string) {
    FakeEventSource.latest = this
  }

  listeners = new Map<string, (event: MessageEvent<string>) => void>()

  addEventListener(type: string, listener: EventListener) {
    this.listeners.set(type, listener as (event: MessageEvent<string>) => void)
  }
  close() {}

  emit(type: string, payload: object) {
    this.listeners.get(type)?.({ data: JSON.stringify(payload) } as MessageEvent<string>)
  }
}

function makeJob(overrides: Partial<V2Job> & Pick<V2Job, 'jobId' | 'status'>): V2Job {
  const status = overrides.status
  return {
    jobId: overrides.jobId,
    kind: 'translation',
    retryOfJobId: null,
    retryMode: null,
    status,
    queueRank: null,
    progress: {
      executionMode: 'sequential',
      jobStatus: status,
      totalItems: 0,
      completedItems: 0,
      failedItems: 0,
      skippedItems: 0,
      cancelledItems: 0,
      pools: [],
    },
    target: {},
    createdAt: null,
    ...overrides,
  }
}

function makeDetail(
  overrides: Partial<V2JobDetail> & Pick<V2JobDetail, 'jobId' | 'status'>,
): V2JobDetail {
  const job = makeJob(overrides)
  return {
    ...job,
    counts: {
      total: job.progress.totalItems,
      pending: 0,
      running: 0,
      completed: job.progress.completedItems,
      failed: job.progress.failedItems,
      skipped: job.progress.skippedItems,
      cancelled: job.progress.cancelledItems,
    },
    durationMs: null,
    error: null,
    configSummary: {},
    items: [],
    failedItems: [],
    artifacts: [],
    resources: [],
    recentEvents: [],
    ...overrides,
  }
}

describe('taskCenterStore snapshot reconciliation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    FakeEventSource.latest = null
    vi.stubGlobal('EventSource', FakeEventSource)
    mocks.list.mockResolvedValue({ items: [], queueRevision: 1 })
    mocks.snapshot.mockResolvedValue({ items: [], queueRevision: 1 })
    mocks.reorder.mockResolvedValue({ queueRevision: 2 })
  })

  it('refreshes the durable snapshot whenever the drawer opens', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    store.open()

    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(2))
    expect(mocks.list).toHaveBeenCalledWith('queue')
    expect(mocks.list).toHaveBeenCalledWith('history')
  })

  it('refreshes the durable snapshot after the event stream reconnects', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    FakeEventSource.latest?.onopen?.()
    mocks.list.mockClear()

    FakeEventSource.latest?.onopen?.()

    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(2))
    expect(store.connected).toBe(true)
  })

  it('coalesces event bursts and never overlaps targeted projection requests', async () => {
    vi.useFakeTimers()
    let releaseSnapshot: (() => void) | undefined
    const pendingSnapshot = new Promise<void>(resolve => { releaseSnapshot = resolve })
    let activeRequests = 0
    let peakRequests = 0
    let snapshotCalls = 0
    mocks.snapshot.mockImplementation(async () => {
      snapshotCalls += 1
      activeRequests += 1
      peakRequests = Math.max(peakRequests, activeRequests)
      if (snapshotCalls === 1) await pendingSnapshot
      activeRequests -= 1
      return { items: [], queueRevision: 1 }
    })
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    for (let index = 1; index <= 10; index += 1) {
      FakeEventSource.latest?.emit('page_completed', {
        eventId: index,
        type: 'page_completed',
        jobId: 'job-1',
        payload: {},
      })
    }
    await vi.advanceTimersByTimeAsync(100)
    expect(mocks.snapshot).toHaveBeenCalledTimes(1)

    for (let index = 11; index <= 20; index += 1) {
      FakeEventSource.latest?.emit('page_completed', {
        eventId: index,
        type: 'page_completed',
        jobId: 'job-1',
        payload: {},
      })
    }
    releaseSnapshot?.()
    await vi.runAllTimersAsync()

    expect(mocks.snapshot).toHaveBeenCalledTimes(2)
    expect(mocks.snapshot).toHaveBeenNthCalledWith(1, ['job-1'])
    expect(mocks.snapshot).toHaveBeenNthCalledWith(2, ['job-1'])
    expect(mocks.list).not.toHaveBeenCalled()
    expect(peakRequests).toBe(1)
    store.disconnect()
    vi.useRealTimers()
  })

  it('drains targeted projection bursts larger than the 200-job API limit', async () => {
    vi.useFakeTimers()
    mocks.snapshot.mockImplementation(async (jobIds: string[]) => ({
      items: jobIds.map(jobId => makeJob({ jobId, status: 'running' })),
      queueRevision: 2,
    }))
    const store = useTaskCenterStore()
    await store.initialize()

    for (let index = 1; index <= 205; index += 1) {
      FakeEventSource.latest?.emit('page_completed', {
        eventId: index,
        type: 'page_completed',
        jobId: `job-${index}`,
        payload: {},
      })
    }
    await vi.runAllTimersAsync()

    expect(mocks.snapshot).toHaveBeenCalledTimes(2)
    expect(mocks.snapshot.mock.calls[0]?.[0]).toHaveLength(200)
    expect(mocks.snapshot.mock.calls[1]?.[0]).toHaveLength(5)
    expect(new Set(mocks.snapshot.mock.calls.flatMap(call => call[0])).size).toBe(205)
    expect(store.queue).toHaveLength(205)
    store.disconnect()
    vi.useRealTimers()
  })

  it('projects complete SSE job snapshots without reloading queue and history', async () => {
    vi.useFakeTimers()
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'queue' ? [makeJob({ jobId: 'job-1', status: 'queued', queueRank: 1 })] : [],
      queueRevision: 1,
    }))
    mocks.snapshot
      .mockResolvedValueOnce({
        items: [makeJob({
          jobId: 'job-1',
          status: 'running',
          queueRank: 1,
          progress: {
            executionMode: 'sequential',
            jobStatus: 'running',
            totalItems: 1,
            completedItems: 1,
            failedItems: 0,
            skippedItems: 0,
            cancelledItems: 0,
            pools: [],
          },
        })],
        queueRevision: 2,
      })
      .mockResolvedValueOnce({
        items: [makeJob({
          jobId: 'job-1',
          status: 'completed',
          finishedAt: '2026-08-04T12:00:00Z',
        })],
        queueRevision: 3,
      })
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    FakeEventSource.latest?.emit('job_started', {
      eventId: 1,
      jobId: 'job-1',
      type: 'job_started',
      payload: {},
      createdAt: null,
    })
    FakeEventSource.latest?.emit('page_completed', {
      eventId: 2,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {},
      createdAt: null,
    })
    await vi.advanceTimersByTimeAsync(100)

    expect(mocks.list).not.toHaveBeenCalled()
    expect(mocks.snapshot).toHaveBeenCalledWith(['job-1'])
    expect(store.queue[0]?.status).toBe('running')
    expect(store.queue[0]?.progress.completedItems).toBe(1)
    expect(store.queueRevision).toBe(2)

    FakeEventSource.latest?.emit('job_finished', {
      eventId: 3,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: null,
    })
    await vi.advanceTimersByTimeAsync(100)

    expect(store.queue).toEqual([])
    expect(store.history.map(job => job.jobId)).toEqual(['job-1'])
    expect(mocks.list).not.toHaveBeenCalled()
    store.disconnect()
    vi.useRealTimers()
  })

  it('refreshes an expanded detail when its projected job reaches a new state', async () => {
    vi.useFakeTimers()
    const paused = makeDetail({
      jobId: 'job-1',
      status: 'paused',
      progress: {
        executionMode: 'sequential',
        jobStatus: 'paused',
        totalItems: 2,
        completedItems: 0,
        failedItems: 0,
        skippedItems: 0,
        cancelledItems: 0,
        pools: [],
      },
      recentEvents: [{
        eventId: 1,
        jobId: 'job-1',
        type: 'job_paused',
        payload: {},
        createdAt: null,
      }],
    })
    const completed = makeDetail({
      jobId: 'job-1',
      status: 'completed',
      finishedAt: '2026-08-08T04:45:00Z',
      durationMs: 30_000,
      progress: {
        executionMode: 'sequential',
        jobStatus: 'completed',
        totalItems: 2,
        completedItems: 2,
        failedItems: 0,
        skippedItems: 0,
        cancelledItems: 0,
        pools: [],
      },
      counts: {
        total: 2,
        pending: 0,
        running: 0,
        completed: 2,
        failed: 0,
        skipped: 0,
        cancelled: 0,
      },
      recentEvents: [{
        eventId: 2,
        jobId: 'job-1',
        type: 'job_finished',
        payload: {},
        createdAt: null,
      }],
    })
    mocks.get.mockResolvedValueOnce(paused).mockResolvedValueOnce(completed)
    mocks.snapshot.mockResolvedValue({
      items: [makeJob({
        jobId: 'job-1',
        status: 'completed',
        finishedAt: completed.finishedAt,
        progress: completed.progress,
      })],
      queueRevision: 2,
    })
    const store = useTaskCenterStore()
    await store.initialize()
    store.drawerOpen = true
    await store.loadDetail('job-1')

    FakeEventSource.latest?.emit('job_finished', {
      eventId: 2,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: null,
    })
    await vi.advanceTimersByTimeAsync(200)

    expect(store.selectedDetail?.status).toBe('completed')
    expect(store.selectedDetail?.counts.completed).toBe(2)
    expect(store.selectedDetail?.durationMs).toBe(30_000)
    expect(store.selectedDetail?.recentEvents.map(event => event.eventId)).toEqual([1, 2])
    store.disconnect()
    vi.useRealTimers()
  })

  it('falls back to one durable refresh when the event cursor has a gap', async () => {
    vi.useFakeTimers()
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    FakeEventSource.latest?.emit('job_started', {
      eventId: 1,
      jobId: 'job-1',
      type: 'job_started',
      payload: {},
      createdAt: null,
    })
    FakeEventSource.latest?.emit('page_completed', {
      eventId: 3,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {},
      createdAt: null,
    })
    await vi.advanceTimersByTimeAsync(250)
    await vi.runAllTimersAsync()

    expect(mocks.list).toHaveBeenCalledTimes(2)
    expect(mocks.snapshot).toHaveBeenCalledTimes(1)
    store.disconnect()
    vi.useRealTimers()
  })

  it('counts paused work as active and interrupted work only as needing attention', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'queue'
        ? [
            makeJob({ jobId: 'paused', status: 'paused' }),
            makeJob({ jobId: 'queued', status: 'queued' }),
          ]
        : [makeJob({ jobId: 'interrupted', status: 'interrupted' })],
      queueRevision: 1,
    }))
    const store = useTaskCenterStore()

    await store.refresh()

    expect(store.activeCount).toBe(1)
    expect(store.queuedCount).toBe(1)
    expect(store.interruptedCount).toBe(1)
  })

  it('keeps the durable history snapshot complete while deriving filters locally', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'history'
        ? [
            makeJob({ jobId: 'failed-1', status: 'failed', kind: 'translation', bookId: 'book-1' }),
            makeJob({ jobId: 'completed-1', status: 'completed', kind: 'insight_analysis', bookId: 'book-2' }),
          ]
        : [],
      queueRevision: 1,
    }))
    const store = useTaskCenterStore()
    store.statusFilter = 'failed'
    store.kindFilter = 'translation'
    store.bookFilter = 'book-1'

    await store.refresh()

    expect(mocks.list).toHaveBeenCalledWith('history')
    expect(store.history.map(job => job.jobId)).toEqual(['failed-1', 'completed-1'])
    expect(store.historyBatches.flatMap(batch => batch.jobs.map(job => job.jobId))).toEqual([
      'failed-1',
    ])
  })

  it('waits for a backend job through the shared durable snapshot', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'history'
        ? [makeJob({
            jobId: 'completed-1',
            status: 'completed',
            progress: {
              executionMode: 'sequential',
              jobStatus: 'completed',
              totalItems: 1,
              completedItems: 1,
              failedItems: 0,
              skippedItems: 0,
              cancelledItems: 0,
              pools: [],
            },
          })]
        : [],
      queueRevision: 1,
    }))
    const detail = {
      jobId: 'completed-1',
      status: 'completed',
      progress: { totalItems: 1 },
      recentEvents: [],
    }
    mocks.get.mockResolvedValue(detail)
    const store = useTaskCenterStore()

    await expect(store.waitForJob('completed-1')).resolves.toBe(detail)

    expect(mocks.get).toHaveBeenCalledWith('completed-1')
  })

  it('separates the current task from waiting batches and prioritizes only sortable queued jobs', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'queue'
        ? [
            makeJob({ jobId: 'running', status: 'running', batchId: 'batch-a' }),
            makeJob({ jobId: 'retained', status: 'queued', blockedReason: 'retained_chapter_lock' }),
            makeJob({ jobId: 'first', status: 'queued', blockedReason: null }),
            makeJob({ jobId: 'target', status: 'queued', blockedReason: null }),
          ]
        : [],
      queueRevision: 7,
    }))
    const store = useTaskCenterStore()
    await store.refresh()

    expect(store.currentJobs.map(job => job.jobId)).toEqual(['running'])
    expect(store.waitingBatches.flatMap(batch => batch.jobs.map(job => job.jobId))).toEqual([
      'retained',
      'first',
      'target',
    ])

    await store.prioritizeQueued('target')

    expect(mocks.reorder).toHaveBeenCalledWith(['target', 'first'], 7)
  })
})
