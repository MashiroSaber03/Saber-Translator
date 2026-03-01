import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const {
  reanalyzeChapterMock,
  getThumbnailUrlMock,
} = vi.hoisted(() => ({
  reanalyzeChapterMock: vi.fn(),
  getThumbnailUrlMock: vi.fn((bookId: string, pageNum: number) => `/thumb/${bookId}/${pageNum}`),
}))

vi.mock('@/api/insight', () => ({
  reanalyzeChapter: reanalyzeChapterMock,
  getThumbnailUrl: getThumbnailUrlMock,
}))

import PagesTree from '@/components/insight/PagesTree.vue'

describe('PagesTree', () => {
  let alertSpy: ReturnType<typeof vi.spyOn>
  let confirmSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setCurrentTaskId(null)
    store.setBookTotalPages(2)
    store.setChapters([
      { id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false },
    ])

    reanalyzeChapterMock.mockReset()
    reanalyzeChapterMock.mockResolvedValue({
      success: true,
      task_id: 'task-chapter-1',
    })
    getThumbnailUrlMock.mockClear()

    ;(globalThis as any).fetch = vi.fn().mockResolvedValue({
      json: async () => ({ success: true, pages: [] }),
    })

    confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
  })

  afterEach(() => {
    confirmSpy.mockRestore()
    alertSpy.mockRestore()
    vi.clearAllMocks()
  })

  it('starts chapter reanalyze via API and writes task state to store', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setCurrentTaskId(null)
    store.setBookTotalPages(2)
    store.setChapters([
      { id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false },
    ])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const reanalyzeButton = wrapper.find('.btn-reanalyze-chapter')
    expect(reanalyzeButton.exists()).toBe(true)

    await reanalyzeButton.trigger('click')
    await flushPromises()

    expect(confirmSpy).toHaveBeenCalled()
    expect(reanalyzeChapterMock).toHaveBeenCalledWith('book-1', 'ch-1')
    expect(store.currentTaskId).toBe('task-chapter-1')
    expect(store.analysisStatus).toBe('running')
    expect(alertSpy).toHaveBeenCalledWith('章节分析已启动')
  })
})
