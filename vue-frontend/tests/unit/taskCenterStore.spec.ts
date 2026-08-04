import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { V2Job } from '@/api/v2/jobs'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  list: vi.fn(),
  reorder: vi.fn(),
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    get: mocks.get,
    list: mocks.list,
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

describe('taskCenterStore snapshot reconciliation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    FakeEventSource.latest = null
    vi.stubGlobal('EventSource', FakeEventSource)
    mocks.list.mockResolvedValue({ items: [], queueRevision: 1 })
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
    mocks.list.mockClear()

    FakeEventSource.latest?.onopen?.()

    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(2))
    expect(store.connected).toBe(true)
  })

  it('coalesces event bursts and never overlaps durable snapshot requests', async () => {
    vi.useFakeTimers()
    let release: (() => void) | undefined
    const pending = new Promise<void>(resolve => { release = resolve })
    let activeRequests = 0
    let peakRequests = 0
    mocks.list.mockImplementation(async () => {
      activeRequests += 1
      peakRequests = Math.max(peakRequests, activeRequests)
      await pending
      activeRequests -= 1
      return { items: [], queueRevision: 1 }
    })
    const store = useTaskCenterStore()
    const initializing = store.initialize()
    release?.()
    await initializing
    mocks.list.mockClear()

    for (let index = 1; index <= 20; index += 1) {
      FakeEventSource.latest?.emit('page_completed', {
        eventId: index,
        eventType: 'page_completed',
        jobId: 'job-1',
        payload: {},
      })
    }
    await vi.advanceTimersByTimeAsync(250)
    await vi.runAllTimersAsync()

    expect(mocks.list).toHaveBeenCalledTimes(2)
    expect(peakRequests).toBeLessThanOrEqual(2)
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
