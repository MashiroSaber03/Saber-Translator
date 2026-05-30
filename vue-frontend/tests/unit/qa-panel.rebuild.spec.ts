import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const {
  rebuildEmbeddingsMock,
  getRebuildEmbeddingsStatusMock,
} = vi.hoisted(() => ({
  rebuildEmbeddingsMock: vi.fn(),
  getRebuildEmbeddingsStatusMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  sendChat: vi.fn(),
  rebuildEmbeddings: rebuildEmbeddingsMock,
  getRebuildEmbeddingsStatus: getRebuildEmbeddingsStatusMock,
}))

import QAPanel from '@/components/insight/QAPanel.vue'

describe('QAPanel rebuild embeddings polling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setLoading(false)

    rebuildEmbeddingsMock.mockReset()
    getRebuildEmbeddingsStatusMock.mockReset()

    vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.spyOn(window, 'alert').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('stops polling and recovers UI when rebuild task cannot be found', async () => {
    rebuildEmbeddingsMock.mockResolvedValue({
      success: true,
      task_id: 'task-1',
    })
    getRebuildEmbeddingsStatusMock.mockResolvedValue({
      success: true,
      task: null,
    })

    const wrapper = mount(QAPanel, {
      global: {
        plugins: [createPinia()],
      },
    })
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    await wrapper.find('button[title="重建向量索引"]').trigger('click')
    await flushPromises()
    await vi.advanceTimersByTimeAsync(3000)
    await flushPromises()

    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('未找到向量重建任务状态'))
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(store.isLoading).toBe(false)
  })

  it('stops polling after repeated status request failures', async () => {
    rebuildEmbeddingsMock.mockResolvedValue({
      success: true,
      task_id: 'task-2',
    })
    getRebuildEmbeddingsStatusMock.mockRejectedValue(new Error('network down'))

    const wrapper = mount(QAPanel, {
      global: {
        plugins: [createPinia()],
      },
    })
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    await wrapper.find('button[title="重建向量索引"]').trigger('click')
    await flushPromises()

    await vi.advanceTimersByTimeAsync(9000)
    await flushPromises()

    expect(getRebuildEmbeddingsStatusMock).toHaveBeenCalledTimes(3)
    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('无法获取任务状态'))
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(store.isLoading).toBe(false)
  })
})
