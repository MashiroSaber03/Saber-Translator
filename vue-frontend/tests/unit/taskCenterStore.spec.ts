import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    list: mocks.list,
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
})
