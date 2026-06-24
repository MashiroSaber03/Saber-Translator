import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const {
  reanalyzeChapterMock,
  getThumbnailUrlMock,
  showToastMock,
} = vi.hoisted(() => ({
  reanalyzeChapterMock: vi.fn(),
  getThumbnailUrlMock: vi.fn((bookId: string, pageNum: number) => `/thumb/${bookId}/${pageNum}`),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  reanalyzeChapter: reanalyzeChapterMock,
  getThumbnailUrl: getThumbnailUrlMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

import PagesTree from '@/components/insight/PagesTree.vue'

describe('PagesTree', () => {
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
    showToastMock.mockReset()

    ;(globalThis as any).fetch = vi.fn().mockResolvedValue({
      json: async () => ({ success: true, pages: [] }),
    })

    confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
  })

  afterEach(() => {
    confirmSpy.mockRestore()
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
    expect(showToastMock).toHaveBeenCalledWith('章节分析已启动', 'success')
  })

  it('refreshes analyzed page markers without routine console output', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([
      { id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false },
    ])

    mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      store.setAnalyzedPagesCount(1)
      await flushPromises()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })

  it('uses explicit controls for chapter toggles and page selection', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
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

    const chapterToggle = wrapper.find('.tree-chapter-toggle')
    expect(chapterToggle.exists()).toBe(true)
    expect(chapterToggle.element.tagName).toBe('BUTTON')
    expect(chapterToggle.attributes('aria-expanded')).toBe('true')

    const pageItem = wrapper.find('.tree-page-item')
    expect(pageItem.element.tagName).toBe('BUTTON')
    expect(pageItem.attributes('aria-label')).toBe('选择第 1 页')
    expect(pageItem.attributes('aria-pressed')).toBe('false')

    await pageItem.trigger('click')

    expect(store.selectedPageNum).toBe(1)
    expect(pageItem.attributes('aria-pressed')).toBe('true')
  })
})
