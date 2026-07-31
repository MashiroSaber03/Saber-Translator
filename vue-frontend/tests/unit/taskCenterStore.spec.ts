import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  reorder: vi.fn(),
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
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

  addEventListener() {}
  close() {}
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
    expect(mocks.list).toHaveBeenCalledWith('history', {})
  })

  it('refreshes the durable snapshot after the event stream reconnects', async () => {
    const store = useTaskCenterStore()
    await store.initialize()
    mocks.list.mockClear()

    FakeEventSource.latest?.onopen?.()

    await vi.waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(2))
    expect(store.connected).toBe(true)
  })

  it('counts paused work as active and interrupted work only as needing attention', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'queue'
        ? [
            { jobId: 'paused', status: 'paused' },
            { jobId: 'queued', status: 'queued' },
          ]
        : [{ jobId: 'interrupted', status: 'interrupted' }],
      queueRevision: 1,
    }))
    const store = useTaskCenterStore()

    await store.refresh()

    expect(store.activeCount).toBe(1)
    expect(store.queuedCount).toBe(1)
    expect(store.interruptedCount).toBe(1)
  })

  it('sends history filters to the backend instead of filtering the durable snapshot locally', async () => {
    const store = useTaskCenterStore()
    store.statusFilter = 'failed'
    store.kindFilter = 'translation'
    store.bookFilter = 'book-1'

    await store.refresh()

    expect(mocks.list).toHaveBeenCalledWith('history', {
      status: 'failed',
      type: 'translation',
      bookId: 'book-1',
    })
  })

  it('separates the current task from waiting batches and prioritizes only sortable queued jobs', async () => {
    mocks.list.mockImplementation(async (scope: 'queue' | 'history') => ({
      items: scope === 'queue'
        ? [
            { jobId: 'running', status: 'running', batchId: 'batch-a', target: {} },
            { jobId: 'retained', status: 'queued', blockedReason: 'retained_chapter_lock', target: {} },
            { jobId: 'first', status: 'queued', blockedReason: null, target: {} },
            { jobId: 'target', status: 'queued', blockedReason: null, target: {} },
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
