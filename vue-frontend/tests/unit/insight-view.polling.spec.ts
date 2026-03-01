import { nextTick } from 'vue'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'

const {
  getAnalysisStatusMock,
  routerReplaceMock,
  routerPushMock,
} = vi.hoisted(() => ({
  getAnalysisStatusMock: vi.fn(),
  routerReplaceMock: vi.fn(),
  routerPushMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({
    replace: routerReplaceMock,
    push: routerPushMock,
  }),
}))

vi.mock('@/api/insight', () => ({
  getAnalysisStatus: getAnalysisStatusMock,
}))

import InsightView from '@/views/InsightView.vue'

describe('InsightView polling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.setAnalyzedPagesCount(0)
    insightStore.dataRefreshKey = 0

    const bookshelfStore = useBookshelfStore()
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined) as any

    getAnalysisStatusMock.mockReset()
    getAnalysisStatusMock.mockResolvedValue({
      success: true,
      analyzed: true,
      fully_analyzed: false,
      analyzed_pages_count: 5,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  it('refreshes panels when polling transitions running -> idle', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.dataRefreshKey = 0

    const bookshelfStore = useBookshelfStore()
    bookshelfStore.fetchBooks = vi.fn().mockResolvedValue(undefined) as any

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppHeader: { template: '<div><slot name="header-links" /></div>' },
          BookSelector: true,
          AnalysisProgress: true,
          OverviewPanel: true,
          TimelinePanel: true,
          QAPanel: true,
          NotesPanel: true,
          PageDetail: true,
          PagesTree: true,
          InsightSettingsModal: true,
          ChapterSelectModal: true,
          ContinuationPanel: true,
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    const refreshKeyBefore = insightStore.dataRefreshKey
    insightStore.setAnalysisStatus('running')
    await nextTick()

    await vi.advanceTimersByTimeAsync(3000)
    await flushPromises()

    expect(getAnalysisStatusMock).toHaveBeenCalledTimes(2)
    expect(getAnalysisStatusMock).toHaveBeenCalledWith('book-1')
    expect(insightStore.analysisStatus).toBe('idle')
    expect(insightStore.dataRefreshKey).not.toBe(refreshKeyBefore)
  })
})
