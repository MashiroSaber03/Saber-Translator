import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { V2Job, V2JobDetail } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  events: vi.fn(),
  list: vi.fn(),
  snapshot: vi.fn(),
  reorder: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  pauseQueue: vi.fn(),
  resumeQueue: vi.fn(),
}))

vi.mock('@/api/v2/jobs', () => ({
  CURRENT_JOB_STATUSES: new Set(['running', 'paused']),
  HISTORY_JOB_STATUSES: new Set([
    'cancelled',
    'completed',
    'completed_with_errors',
    'failed',
    'interrupted',
  ]),
  NONTERMINAL_JOB_STATUSES: new Set([
    'queued',
    'running',
    'paused',
    'interrupted',
  ]),
  jobsApi: {
    get: mocks.get,
    events: mocks.events,
    list: mocks.list,
    snapshot: mocks.snapshot,
    reorder: mocks.reorder,
    retry: mocks.retry,
    cancel: mocks.cancel,
    pauseQueue: mocks.pauseQueue,
    resumeQueue: mocks.resumeQueue,
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

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolvePromise => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

function makeJob(overrides: Partial<V2Job> & Pick<V2Job, 'jobId' | 'status'>): V2Job {
  const status = overrides.status
  return {
    jobId: overrides.jobId,
    batchId: null,
    batchDisplayName: null,
    kind: 'translation',
    retryOfJobId: null,
    retryMode: null,
    status,
    queueRank: null,
    bookId: null,
    chapterId: null,
    pageId: null,
    blockedReason: null,
    blockedByJobId: null,
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
    createdAt: '2026-08-23T04:00:00Z',
    startedAt: null,
    finishedAt: null,
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
    mocks.list.mockResolvedValue({
      items: [],
      queuePaused: false,
      eventCursor: 0,
      workerOnline: true,
      executorBusy: false,
      waitingReason: null,
    })
    mocks.snapshot.mockResolvedValue({
      items: [],
      queuePaused: false,
      workerOnline: true,
      executorBusy: false,
      waitingReason: null,
    })
    mocks.events.mockResolvedValue({ items: [] })
    mocks.reorder.mockResolvedValue({ status: 'reordered' })
    mocks.retry.mockResolvedValue({
      batchId: 'batch-retry',
      jobIds: ['retry-1', 'retry-2'],
      status: 'queued',
      sourceJobId: 'source-job',
      retryMode: 'current',
      failedOnly: false,
    })
    mocks.cancel.mockResolvedValue({ jobId: 'job-1', status: 'cancelled' })
    mocks.pauseQueue.mockResolvedValue({ queuePaused: true })
    mocks.resumeQueue.mockResolvedValue({ queuePaused: false })
  })

  it('refreshes the durable snapshot whenever the drawer opens', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    store.open()

    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledOnce())
    expect(mocks.list).toHaveBeenCalledWith('all')
  })

  it('refreshes the durable snapshot after the event stream reconnects', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    FakeEventSource.latest?.onopen?.()
    mocks.list.mockClear()

    FakeEventSource.latest?.onopen?.()
    await store.refresh()
    expect(mocks.list).toHaveBeenCalledOnce()
    expect(store.connected).toBe(true)
  })

  it('reconciles active jobs when the event stream stops delivering updates', async () => {
    vi.useFakeTimers()
    const running = makeJob({
      jobId: 'job-1',
      status: 'running',
      chapterId: 'chapter-1',
    })
    const completed = makeJob({
      ...running,
      status: 'completed',
      finishedAt: '2026-08-23T04:01:42Z',
    })
    mocks.list.mockResolvedValue({
      items: [running],
    })
    mocks.snapshot.mockResolvedValue({
      items: [completed],
    })
    const store = useTaskCenterStore()
    await store.initialize()
    FakeEventSource.latest?.onopen?.()
    FakeEventSource.latest?.onerror?.()
    mocks.snapshot.mockClear()

    await vi.advanceTimersByTimeAsync(15_100)

    expect(mocks.snapshot).toHaveBeenCalledOnce()
    expect(mocks.snapshot).toHaveBeenCalledWith(['job-1'])
    expect(store.queue).toEqual([])
    expect(store.history[0]?.status).toBe('completed')
    store.disconnect()
    vi.useRealTimers()
  })

  it('connects the event stream and recovers when the initial snapshot fails', async () => {
    mocks.list.mockRejectedValueOnce(new Error('temporary snapshot failure'))
    const store = useTaskCenterStore()

    await expect(store.initialize()).resolves.toBeUndefined()
    expect(FakeEventSource.latest).not.toBeNull()

    mocks.list.mockClear()
    FakeEventSource.latest?.onopen?.()
    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledOnce())
    expect(store.connected).toBe(true)
  })

  it('does not reconnect or publish an old snapshot after disconnect', async () => {
    const request = deferred<Awaited<ReturnType<typeof mocks.list>>>()
    mocks.list.mockReturnValueOnce(request.promise)
    const store = useTaskCenterStore()

    const initialization = store.initialize()
    store.disconnect()
    request.resolve({
      items: [makeJob({ jobId: 'old-user-job', status: 'queued' })],
      queuePaused: false,
      eventCursor: 8,
      workerOnline: true,
      executorBusy: false,
      waitingReason: null,
    })
    await initialization

    expect(FakeEventSource.latest).toBeNull()
    expect(store.queue).toEqual([])
    expect(store.snapshotLoaded).toBe(false)
    expect(store.workerOnline).toBe(false)
  })

  it('rejects job waiters when their user lifecycle is reset', async () => {
    mocks.list.mockResolvedValueOnce({
      items: [makeJob({ jobId: 'job-1', status: 'queued' })],
      queuePaused: false,
      eventCursor: 1,
      workerOnline: true,
      executorBusy: false,
      waitingReason: 'executor_busy',
    })
    const store = useTaskCenterStore()
    const waiting = store.waitForJob('job-1')
    await vi.waitFor(() => expect(store.queue).toHaveLength(1))

    store.disconnect()

    await expect(waiting).rejects.toThrow('任务上下文已切换')
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
      return { items: [], queuePaused: false, eventCursor: 0, workerOnline: true, executorBusy: false, waitingReason: null }
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
        createdAt: '2026-08-23T04:00:00Z',
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
        createdAt: '2026-08-23T04:00:00Z',
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
    }))
    const store = useTaskCenterStore()
    await store.initialize()

    for (let index = 1; index <= 205; index += 1) {
      FakeEventSource.latest?.emit('page_completed', {
        eventId: index,
        type: 'page_completed',
        jobId: `job-${index}`,
        payload: {},
        createdAt: '2026-08-23T04:00:00Z',
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
    mocks.list.mockResolvedValue({
      items: [makeJob({ jobId: 'job-1', status: 'queued', queueRank: 1 })],
    })
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
      })
      .mockResolvedValueOnce({
        items: [makeJob({
          jobId: 'job-1',
          status: 'completed',
          finishedAt: '2026-08-04T12:00:00Z',
        })],
      })
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    FakeEventSource.latest?.emit('job_started', {
      eventId: 1,
      jobId: 'job-1',
      type: 'job_started',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    FakeEventSource.latest?.emit('page_completed', {
      eventId: 2,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    await vi.advanceTimersByTimeAsync(100)

    expect(mocks.list).not.toHaveBeenCalled()
    expect(mocks.snapshot).toHaveBeenCalledWith(['job-1'])
    expect(store.queue[0]?.status).toBe('running')
    expect(store.queue[0]?.progress.completedItems).toBe(1)

    FakeEventSource.latest?.emit('job_finished', {
      eventId: 3,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
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
        createdAt: '2026-08-23T04:00:00Z',
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
        createdAt: '2026-08-23T04:00:00Z',
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
      createdAt: '2026-08-23T04:00:00Z',
    })
    await vi.advanceTimersByTimeAsync(200)

    expect(store.selectedDetail?.status).toBe('completed')
    expect(store.selectedDetail?.counts.completed).toBe(2)
    expect(store.selectedDetail?.durationMs).toBe(30_000)
    expect(store.selectedDetail?.recentEvents.map(event => event.eventId)).toEqual([1, 2])
    store.disconnect()
    vi.useRealTimers()
  })

  it('keeps the latest detail selection when requests finish out of order', async () => {
    let resolveFirst: ((detail: V2JobDetail) => void) | undefined
    let resolveSecond: ((detail: V2JobDetail) => void) | undefined
    mocks.get.mockImplementation((jobId: string) => new Promise<V2JobDetail>(resolve => {
      if (jobId === 'job-1') resolveFirst = resolve
      else resolveSecond = resolve
    }))
    const store = useTaskCenterStore()

    const firstRequest = store.loadDetail('job-1')
    const secondRequest = store.loadDetail('job-2')
    resolveSecond?.(makeDetail({ jobId: 'job-2', status: 'running' }))
    await secondRequest
    resolveFirst?.(makeDetail({ jobId: 'job-1', status: 'completed' }))
    await firstRequest

    expect(store.selectedDetailJobId).toBe('job-2')
    expect(store.selectedDetail?.jobId).toBe('job-2')
    expect(store.detailLoading).toBe(false)
  })

  it('accepts gaps in the owner-filtered global event cursor', async () => {
    vi.useFakeTimers()
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    FakeEventSource.latest?.emit('job_started', {
      eventId: 1,
      jobId: 'job-1',
      type: 'job_started',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    FakeEventSource.latest?.emit('page_completed', {
      eventId: 3,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    await vi.advanceTimersByTimeAsync(250)
    await vi.runAllTimersAsync()

    expect(mocks.list).not.toHaveBeenCalled()
    expect(mocks.snapshot).toHaveBeenCalledTimes(1)
    store.disconnect()
    vi.useRealTimers()
  })

  it('rejects malformed or mislabeled SSE events and reloads the durable snapshot', async () => {
    vi.useFakeTimers()
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()
    mocks.snapshot.mockClear()

    FakeEventSource.latest?.emit('page_completed', {
      eventId: 1,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    FakeEventSource.latest?.emit('page_completed', {
      eventId: 2,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {},
    })
    await vi.advanceTimersByTimeAsync(250)
    await vi.runAllTimersAsync()

    expect(mocks.snapshot).not.toHaveBeenCalled()
    expect(mocks.list).toHaveBeenCalledOnce()
    store.disconnect()
    vi.useRealTimers()
  })

  it('never trims recoverable interrupted batches from the local history projection', async () => {
    vi.useFakeTimers()
    const interrupted = makeJob({
      jobId: 'interrupted-old',
      batchId: 'batch-interrupted',
      status: 'interrupted',
    })
    const terminal = Array.from({ length: 200 }, (_, index) => makeJob({
      jobId: `completed-${index}`,
      batchId: `batch-completed-${index}`,
      status: 'completed',
    }))
    mocks.list.mockResolvedValue({
      items: [...terminal, interrupted],
    })
    mocks.snapshot.mockResolvedValue({
      items: [makeJob({
        jobId: 'completed-new',
        batchId: 'batch-completed-new',
        status: 'completed',
      })],
    })
    const store = useTaskCenterStore()
    await store.initialize()

    FakeEventSource.latest?.emit('job_finished', {
      eventId: 1,
      jobId: 'completed-new',
      type: 'job_finished',
      payload: {},
      createdAt: '2026-08-23T04:00:00Z',
    })
    await vi.advanceTimersByTimeAsync(100)
    await vi.runAllTimersAsync()

    expect(store.history.some(job => job.jobId === 'interrupted-old')).toBe(true)
    expect(new Set(store.history.map(job => job.batchId)).size).toBe(201)
    store.disconnect()
    vi.useRealTimers()
  })

  it('counts paused work as active and interrupted work only as needing attention', async () => {
    mocks.list.mockResolvedValue({
      items: [
        makeJob({ jobId: 'paused', status: 'paused' }),
        makeJob({ jobId: 'queued', status: 'queued' }),
        makeJob({ jobId: 'interrupted', status: 'interrupted' }),
      ],
    })
    const store = useTaskCenterStore()

    await store.refresh()

    expect(store.activeCount).toBe(1)
    expect(store.queuedCount).toBe(1)
    expect(store.interruptedCount).toBe(1)
  })

  it('keeps the durable history snapshot complete while deriving filters locally', async () => {
    mocks.list.mockResolvedValue({
      items: [
        makeJob({ jobId: 'failed-1', status: 'failed', kind: 'translation', bookId: 'book-1' }),
        makeJob({ jobId: 'completed-1', status: 'completed', kind: 'insight_analysis', bookId: 'book-2' }),
      ],
    })
    const store = useTaskCenterStore()
    store.statusFilter = 'failed'
    store.kindFilter = 'translation'
    store.bookFilter = 'book-1'

    await store.refresh()

    expect(mocks.list).toHaveBeenCalledWith('all')
    expect(store.history.map(job => job.jobId)).toEqual(['failed-1', 'completed-1'])
    expect(store.historyBatches.flatMap(batch => batch.jobs.map(job => job.jobId))).toEqual([
      'failed-1',
    ])
  })

  it('waits for a backend job through the shared durable snapshot', async () => {
    mocks.list.mockResolvedValue({
      items: [makeJob({
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
          })],
    })
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
    mocks.list.mockResolvedValue({
      items: [
            makeJob({ jobId: 'running', status: 'running', batchId: 'batch-a' }),
            makeJob({ jobId: 'retained', status: 'queued', blockedReason: 'retained_chapter_lock' }),
            makeJob({ jobId: 'first', status: 'queued', blockedReason: null }),
            makeJob({ jobId: 'target', status: 'queued', blockedReason: null }),
          ],
    })
    const store = useTaskCenterStore()
    await store.refresh()

    expect(store.currentJobs.map(job => job.jobId)).toEqual(['running'])
    expect(store.waitingBatches.flatMap(batch => batch.jobs.map(job => job.jobId))).toEqual([
      'retained',
      'first',
      'target',
    ])

    await store.prioritizeQueued('target')

    expect(mocks.reorder).toHaveBeenCalledWith(['target', 'first'])
  })

  it('does not turn an accepted command into a failure when the follow-up snapshot fails', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockRejectedValue(new Error('snapshot unavailable'))

    await expect(store.cancel('job-1')).resolves.toEqual({
      jobId: 'job-1',
      status: 'cancelled',
    })

    expect(mocks.cancel).toHaveBeenCalledWith('job-1')
  })

  it('projects every replacement job from a retry without reloading all history', async () => {
    vi.useFakeTimers()
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    await store.retry('source-job')
    await vi.runAllTimersAsync()

    expect(mocks.retry).toHaveBeenCalledWith('source-job', 'current')
    expect(mocks.snapshot).toHaveBeenCalledWith(['retry-1', 'retry-2'])
    expect(mocks.list).not.toHaveBeenCalled()
    store.disconnect()
    vi.useRealTimers()
  })

  it('persists the queue admission gate returned by the backend', async () => {
    const store = useTaskCenterStore()
    await store.initialize()

    await store.pauseQueue()
    expect(store.queuePaused).toBe(true)

    await store.resumeQueue()
    expect(store.queuePaused).toBe(false)
  })
})
