import { nextTick } from 'vue'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types'

const {
  getBookDetailMock,
  getAnalysisStatusMock,
  routerReplaceMock,
  routerPushMock,
} = vi.hoisted(() => ({
  getBookDetailMock: vi.fn(),
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

vi.mock('@/api/bookshelf', () => ({
  getBookDetail: getBookDetailMock,
}))

import InsightView from '@/views/InsightView.vue'

type BookDetailSuccess = {
  success: true
  book: BookData
}

function createBook(id: string, title: string): BookData {
  return {
    id,
    title,
    total_pages: 1,
    chapters: [],
    createdAt: '2026-06-25T00:00:00Z',
    updatedAt: '2026-06-25T00:00:00Z',
  }
}

function createDeferred<T>() {
  let resolve: (value: T) => void = () => {}
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

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
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined) as any

    getAnalysisStatusMock.mockReset()
    getBookDetailMock.mockReset()
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
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined) as any

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
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

  it('cancels the delayed completed refresh when unmounted', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.dataRefreshKey = 0

    const bookshelfStore = useBookshelfStore()
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined) as any

    getAnalysisStatusMock.mockResolvedValue({
      success: true,
      analyzed: true,
      fully_analyzed: true,
      analyzed_pages_count: 5,
    })

    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
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
    expect(insightStore.analysisStatus).toBe('completed')
    expect(getAnalysisStatusMock).toHaveBeenCalledTimes(1)

    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(1000)
    await flushPromises()

    expect(getAnalysisStatusMock).toHaveBeenCalledTimes(1)
    expect(insightStore.dataRefreshKey).toBe(refreshKeyBefore)
  })

  it('uses safe header navigation semantics', () => {
    const wrapper = shallowMount(InsightView, {
      global: {
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
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

    expect(wrapper.find('a[href="javascript:void(0)"]').exists()).toBe(false)
    expect(wrapper.get('a[href="https://www.mashirosaber.top/use/manga-insight.html"]').attributes('rel'))
      .toBe('noopener noreferrer')
  })

  it('ignores stale book load responses when a newer selection finishes first', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    const bookshelfStore = useBookshelfStore()
    bookshelfStore.loadBooks = vi.fn().mockResolvedValue(undefined) as any

    const firstLoad = createDeferred<BookDetailSuccess>()
    const secondLoad = createDeferred<BookDetailSuccess>()
    getBookDetailMock.mockImplementation((bookId: string) => (
      bookId === 'book-1' ? firstLoad.promise : secondLoad.promise
    ))

    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
          AppHeader: { template: '<div><slot name="header-links" /></div>' },
          SidebarLayout: { template: '<main><slot /></main>' },
          BookSelector: {
            template: `
              <div>
                <button data-testid="select-book-1" @click="$emit('select', 'book-1')">Book 1</button>
                <button data-testid="select-book-2" @click="$emit('select', 'book-2')">Book 2</button>
              </div>
            `,
          },
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
          CharacterStudioEntryPanel: true,
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    const [firstButton, secondButton] = wrapper.findAll('[data-testid^="select-book-"]')
    firstButton!.element.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    secondButton!.element.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await nextTick()

    secondLoad.resolve({ success: true, book: createBook('book-2', 'Second Book') })
    await flushPromises()

    expect(wrapper.text()).toContain('Second Book')

    firstLoad.resolve({ success: true, book: createBook('book-1', 'First Book') })
    await flushPromises()

    expect(insightStore.currentBookId).toBe('book-2')
    expect(wrapper.text()).toContain('Second Book')
    expect(wrapper.text()).not.toContain('First Book')
    expect(routerReplaceMock).toHaveBeenLastCalledWith({ query: { book: 'book-2' } })
  })
})
