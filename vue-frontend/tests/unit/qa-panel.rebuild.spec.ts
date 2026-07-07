import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const {
  rebuildEmbeddingsMock,
  getRebuildEmbeddingsStatusMock,
  showToastMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  rebuildEmbeddingsMock: vi.fn(),
  getRebuildEmbeddingsStatusMock: vi.fn(),
  showToastMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  sendChat: vi.fn(),
  rebuildEmbeddings: rebuildEmbeddingsMock,
  getRebuildEmbeddingsStatus: getRebuildEmbeddingsStatusMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
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
    showToastMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)

    vi.spyOn(window, 'confirm').mockReturnValue(true)
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

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '重建向量索引',
      message: '确定要重建向量索引吗？这将删除现有的向量数据并重新构建，可能需要一些时间。',
      confirmText: '重建',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(window.confirm).not.toHaveBeenCalled()
    expect(showToastMock).toHaveBeenCalledWith(
      expect.stringContaining('未找到向量重建任务状态'),
      'error'
    )
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
    expect(showToastMock).toHaveBeenCalledWith(
      expect.stringContaining('无法获取任务状态'),
      'error'
    )
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(store.isLoading).toBe(false)
  })

  it('stops rebuild polling instead of applying an old task id to a newly selected book', async () => {
    rebuildEmbeddingsMock.mockResolvedValue({
      success: true,
      task_id: 'task-from-book-1',
    })
    getRebuildEmbeddingsStatusMock.mockResolvedValue({
      success: true,
      task: {
        status: 'running',
        progress: {
          current_phase: '向量化',
          analyzed_pages: 1,
          total_pages: 10,
        },
      },
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

    store.currentBookId = 'book-2'
    await vi.advanceTimersByTimeAsync(3000)
    await flushPromises()

    expect(getRebuildEmbeddingsStatusMock).not.toHaveBeenCalled()
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(store.isLoading).toBe(false)
  })
})
