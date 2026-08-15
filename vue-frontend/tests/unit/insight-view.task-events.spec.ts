import { nextTick } from 'vue'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import type { Component } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { enableAutoUnmount, flushPromises, mount, shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { V2Job } from '@/api/v2/jobs'
import type { BookData } from '@/types'

enableAutoUnmount(afterEach)

const {
  getBookDetailMock,
  getAnalysisStatusMock,
  getInsightChaptersMock,
  getNotesMock,
  routerReplaceMock,
  routerPushMock,
  routeMock,
  showToastMock,
} = vi.hoisted(() => ({
  getBookDetailMock: vi.fn(),
  getAnalysisStatusMock: vi.fn(),
  getInsightChaptersMock: vi.fn(),
  getNotesMock: vi.fn(),
  routerReplaceMock: vi.fn(),
  routerPushMock: vi.fn(),
  routeMock: {
    current: null as { query: Record<string, unknown> } | null,
  },
  showToastMock: vi.fn(),
}))

vi.mock('vue-router', async () => {
  const { reactive } = await import('vue')
  const route = reactive({ query: {} as Record<string, unknown> })
  routeMock.current = route
  return {
    useRoute: () => route,
    useRouter: () => ({
      replace: routerReplaceMock,
      push: routerPushMock,
    }),
  }
})

vi.mock('@/api/insight', () => ({
  getAnalysisStatus: getAnalysisStatusMock,
  getInsightChapters: getInsightChaptersMock,
  getNotes: getNotesMock,
}))

vi.mock('@/api/bookshelf', () => ({
  getBookDetail: getBookDetailMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

import InsightView from '@/views/InsightView.vue'

const ProductPageHeaderStub = {
  props: {
    variant: {
      type: String,
      default: 'default',
    },
    navLabel: {
      type: String,
      default: '页面导航',
    },
    actionsLabel: {
      type: String,
      default: '页面操作',
    },
  },
  template: `
    <header class="product-page-header" :class="'product-page-header--' + variant">
      <slot name="meta" />
      <nav :aria-label="navLabel"><slot name="nav" /></nav>
      <div role="group" :aria-label="actionsLabel"><slot name="actions" /></div>
    </header>
  `,
}

function insightViewStubs(
  overrides: Record<string, Component | boolean> = {},
): Record<string, Component | boolean> {
  return {
    AppShell: { template: '<section><slot name="header" /><slot /></section>' },
    ProductPageHeader: ProductPageHeaderStub,
    ProductThemeToggle: false,
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
    CharacterStudioEntryPanel: true,
    'router-link': { template: '<a><slot /></a>' },
    ...overrides,
  }
}

type BookDetailSuccess = BookData

function createBook(id: string, title: string): BookData {
  return {
    id,
    title,
    totalPages: 1,
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

function setRouteQuery(query: Record<string, unknown>): void {
  if (!routeMock.current) throw new Error('route mock is not initialized')
  routeMock.current.query = query
}

function insightJob(overrides: Partial<V2Job> = {}): V2Job {
  return {
    jobId: 'analysis-job-1',
    kind: 'insight_analysis',
    retryOfJobId: null,
    retryMode: null,
    status: 'running',
    queueRank: 1,
    bookId: 'book-1',
    progress: {
      executionMode: 'sequential',
      jobStatus: 'running',
      totalItems: 10,
      completedItems: 4,
      failedItems: 0,
      skippedItems: 0,
      cancelledItems: 0,
      pools: [
        {
          kind: 'insight_analyze_page',
          total: 4,
          completed: 2,
          failed: 1,
          skipped: 1,
          cancelled: 0,
          waiting: 0,
          processing: 0,
          lockWaiting: false,
          current: [],
        },
        {
          kind: 'insight_publish_run',
          total: 1,
          completed: 0,
          failed: 0,
          skipped: 0,
          cancelled: 0,
          waiting: 1,
          processing: 0,
          lockWaiting: false,
          current: [],
        },
      ],
    },
    target: {},
    createdAt: null,
    ...overrides,
  }
}

describe('InsightView task event projection', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)

    setRouteQuery({})

    getAnalysisStatusMock.mockReset()
    getBookDetailMock.mockReset()
    getInsightChaptersMock.mockReset()
    getNotesMock.mockReset()
    routerReplaceMock.mockReset()
    routerPushMock.mockReset()
    showToastMock.mockReset()
    getBookDetailMock.mockResolvedValue(createBook('book-1', 'First Book'))
    getAnalysisStatusMock.mockResolvedValue({
      fullyAnalyzed: false,
      analyzedPagesCount: 5,
    })
    getInsightChaptersMock.mockResolvedValue([])
    getNotesMock.mockResolvedValue({ items: [], nextCursor: null })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('clears every book-scoped projection when the selected book changes', () => {
    const insightStore = useInsightStore()
    insightStore.setCurrentBook('book-1')
    insightStore.setCurrentTaskId('analysis-job-1')
    insightStore.setAnalysisStatus('running')
    insightStore.updateProgress(7, 20, '分析中')
    insightStore.setBookTotalPages(20)
    insightStore.setAnalyzedPagesCount(7)
    insightStore.setChapters([{
      id: 'chapter-1',
      title: '第一章',
      startPage: 1,
      endPage: 20,
      analyzed: false,
    }])
    insightStore.selectPage(7)
    insightStore.addQAMessage({
      id: 'message-1',
      role: 'user',
      content: '上一册的问题',
    })
    insightStore.setStreaming(true)
    insightStore.setError('上一册错误')

    insightStore.setCurrentBook('book-2')

    expect(insightStore.currentBookId).toBe('book-2')
    expect(insightStore.currentTaskId).toBeNull()
    expect(insightStore.analysisStatus).toBe('idle')
    expect(insightStore.progress).toEqual({ current: 0, total: 0, status: 'idle' })
    expect(insightStore.totalPageCount).toBe(0)
    expect(insightStore.analyzedPageCount).toBe(0)
    expect(insightStore.chapters).toEqual([])
    expect(insightStore.selectedPageNum).toBeNull()
    expect(insightStore.qaHistory).toEqual([])
    expect(insightStore.isStreaming).toBe(false)
    expect(insightStore.error).toBeNull()
  })

  it('projects active progress and refreshes backend facts on a terminal task event', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })

    await flushPromises()
    insightStore.setCurrentBook('book-1')
    const taskCenterStore = useTaskCenterStore()
    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.queue = [insightJob()]
    await nextTick()

    expect(insightStore.analysisStatus).toBe('running')
    expect(insightStore.currentTaskId).toBe('analysis-job-1')
    expect(insightStore.progress.current).toBe(4)
    expect(insightStore.progress.total).toBe(4)

    taskCenterStore.latestEvent = {
      eventId: 101,
      jobId: 'analysis-job-1',
      type: 'job_finished',
      payload: { status: 'completed' },
      createdAt: null,
    }
    await flushPromises()

    expect(getAnalysisStatusMock).toHaveBeenCalledTimes(1)
    expect(getAnalysisStatusMock).toHaveBeenCalledWith('book-1')
    expect(insightStore.analysisStatus).toBe('completed')
    expect(insightStore.currentTaskId).toBeNull()
    expect(insightStore.progress.current).toBe(4)
    expect(insightStore.progress.total).toBe(4)
    expect(insightStore.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('does not apply an old terminal event after the user switches books', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.setCurrentBook('book-1')
    const statusLoad = createDeferred<{
      analyzedPagesCount: number
      fullyAnalyzed: boolean
    }>()
    getAnalysisStatusMock.mockReturnValueOnce(statusLoad.promise)

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })

    await flushPromises()
    insightStore.setCurrentBook('book-1')
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.queue = [insightJob()]
    await nextTick()
    expect(insightStore.currentTaskId).toBe('analysis-job-1')

    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.latestEvent = {
      eventId: 104,
      jobId: 'analysis-job-1',
      type: 'job_finished',
      payload: { status: 'completed' },
      createdAt: null,
    }
    await nextTick()
    expect(getAnalysisStatusMock).toHaveBeenCalledWith('book-1')

    insightStore.setCurrentBook('book-2')
    statusLoad.resolve({ analyzedPagesCount: 10, fullyAnalyzed: true })
    await flushPromises()

    expect(insightStore.currentBookId).toBe('book-2')
    expect(insightStore.currentTaskId).toBeNull()
    expect(insightStore.analysisStatus).toBe('idle')
    expect(insightStore.analyzedPageCount).toBe(0)
    expect(insightStore.progress).toEqual({ current: 0, total: 0, status: 'idle' })
    expect(insightStore.dataRefreshKey).toBe(refreshKeyBefore)
  })

  it('keeps a recoverable interrupted job available for the distinct continue command', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })

    await flushPromises()
    insightStore.setCurrentBook('book-1')
    insightStore.setAnalysisStatus('failed')
    insightStore.setCurrentTaskId(null)
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.history = [insightJob({ status: 'interrupted' })]
    await nextTick()

    expect(insightStore.analysisStatus).toBe('interrupted')
    expect(insightStore.currentTaskId).toBe('analysis-job-1')
  })

  it('refreshes derived Insight facts when a backend rebuild finishes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })

    await flushPromises()
    insightStore.setCurrentBook('book-1')
    insightStore.dataRefreshKey = 0
    const taskCenterStore = useTaskCenterStore()
    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.queue = [insightJob({
      jobId: 'derived-job-1',
      kind: 'derived_rebuild',
    })]
    await nextTick()

    taskCenterStore.latestEvent = {
      eventId: 103,
      jobId: 'derived-job-1',
      type: 'job_finished',
      payload: {},
      createdAt: null,
    }
    await flushPromises()

    expect(getAnalysisStatusMock).not.toHaveBeenCalled()
    expect(insightStore.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('does not process task events after the view is unmounted', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()

    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs({ ProductHeaderAction: false }),
      },
    })

    await flushPromises()
    insightStore.setCurrentBook('book-1')
    insightStore.setAnalysisStatus('idle')
    insightStore.dataRefreshKey = 0
    const taskCenterStore = useTaskCenterStore()
    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.queue = [insightJob()]
    await nextTick()
    expect(insightStore.currentTaskId).toBe('analysis-job-1')

    wrapper.unmount()
    taskCenterStore.latestEvent = {
      eventId: 102,
      jobId: 'analysis-job-1',
      type: 'job_finished',
      payload: {},
      createdAt: null,
    }
    await flushPromises()

    expect(getAnalysisStatusMock).not.toHaveBeenCalled()
    expect(insightStore.dataRefreshKey).toBe(refreshKeyBefore)
  })

  it('uses safe header navigation semantics', async () => {
    const wrapper = mount(InsightView, {
      global: {
        stubs: insightViewStubs(),
      },
    })
    const settingsStore = useSettingsStore()

    expect(wrapper.get('.product-page-header--fixed').exists()).toBe(true)
    expect(wrapper.get('nav[aria-label="漫画分析导航"]').exists()).toBe(true)
    const actionGroup = wrapper.get('[role="group"][aria-label="漫画分析操作"]')
    expect(actionGroup.find('[aria-label="设置"]').exists()).toBe(true)
    const themeToggle = wrapper.get('.insight-header__theme-toggle')
    expect(actionGroup.find('.insight-header__theme-toggle').exists()).toBe(true)
    expect(themeToggle.attributes('aria-label')).toBe('切换深色模式')
    await themeToggle.trigger('click')
    expect(settingsStore.theme).toBe('dark')
    expect(wrapper.find('a[href="javascript:void(0)"]').exists()).toBe(false)
    expect(wrapper.get('a[href="https://www.mashirosaber.top/use/manga-insight.html"]').attributes('rel'))
      .toBe('noopener noreferrer')
  })

  it('keeps the insight view free of legacy DOM id hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    for (const legacyId of [
      'id="settingsBtn"',
      'id="themeToggle"',
      'id="totalPages"',
      'id="analyzedPages"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
  })

  it('keeps task event projection free of page-local polling', () => {
    const viewSource = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(viewSource).not.toContain('setInterval(')
    expect(viewSource).not.toContain('setTimeout(')
    expect(viewSource).toContain('taskCenterStore.latestEvent')
  })

  it('keeps page header action appearance on the product header primitive', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).toContain('nav-label="漫画分析导航"')
    expect(source).toContain('actions-label="漫画分析操作"')
    expect(source).not.toContain('.insight-settings-action {')
  })

  it('collapses fixed-header navigation labels on mobile through the header action primitive', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source.match(/collapse-label-on-mobile/g) ?? []).toHaveLength(4)
    expect(source).not.toContain('.insight-header__nav-link .product-header-action__label')
  })

  it('names page-local Insight tokens by concrete owner roles', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).not.toContain('--insight-border-color')
    expect(source).toContain('--insight-view-sidebar-divider')
  })

  it('keeps each insight tab content shrinkable inside the shared workspace shell', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')
    const tabContentBlock = source.match(/\.insight-view__tab-content\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(tabContentBlock).toContain('flex: 1')
    expect(tabContentBlock).toContain('min-width: 0')
    expect(tabContentBlock).toContain('min-height: 0')
    expect(tabContentBlock).toContain('overflow-y: auto')
  })

  it('does not keep stale local spinner styles in the page view', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).not.toContain('.loading-spinner')
  })

  it('does not keep stale local empty-state selectors in the page view', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).not.toContain('.insight-page .placeholder-text')
    expect(source).not.toContain('.insight-page .empty-hint')
    expect(source).toContain('class="insight-view__book-cover-placeholder"')
    expect(source).toContain('.insight-view__book-cover-placeholder')
  })

  it('keeps page-local presentation hooks under the InsightView owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    for (const currentHook of [
      'class="insight-view__main"',
      'class="insight-view__book-summary"',
      'class="insight-view__book-cover-frame"',
      'class="insight-view__book-cover"',
      'class="insight-view__book-title"',
      'class="insight-view__book-meta"',
      'class="insight-view__book-meta-item"',
      'class="insight-view__book-meta-icon"',
      'class="insight-view__content"',
      'class="insight-view__select-book-prompt"',
      'class="insight-view__select-book-icon"',
      'class="insight-view__select-book-title"',
      'class="insight-view__select-book-description"',
      'class="insight-view__tabbed-workspace"',
      'class="insight-view__mobile-nav-button"',
      'class="insight-view__tab-content"',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const legacyHook of [
      'class="insight-main"',
      'class="insight-content"',
      'class="insight-tabbed-workspace"',
      'class="sidebar-section book-info-section"',
      'class="book-cover-wrapper"',
      'class="book-cover"',
      'class="book-cover-placeholder"',
      'class="insight-book-title"',
      'class="book-meta"',
      'class="meta-item"',
      'class="meta-icon"',
      'class="select-book-prompt"',
      'class="prompt-icon"',
      'class="mobile-nav-btn"',
      'class="tab-content"',
    ]) {
      expect(source).not.toContain(legacyHook)
    }

    expect(source).not.toContain('.select-book-prompt h2')
    expect(source).not.toContain('.select-book-prompt p')
    expect(source).not.toContain('.sidebar-section')
    expect(source).not.toContain('.insight-page .insight-view__')
  })

  it('uses the current bookshelf DTO without wire aliases', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).not.toContain('normalizeBookData')
    expect(source).not.toContain('function getChapterPageCount')
    expect(source).not.toContain('chapter.page_count')
    expect(source).not.toContain('chapter.image_count')
    expect(source).not.toContain('book.total_pages')
    expect(source).not.toContain('currentBook?.total_pages')
    expect(source).toContain('book.totalPages')
    expect(source).toContain('currentBook?.totalPages')
  })

  it('uses the shared product tabbed workspace shell for selected-book panels', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    setRouteQuery({ book: 'book-1' })

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await flushPromises()

    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(true)
    expect(wrapper.find('.product-tabbed-workspace__panels > .insight-view__tab-content').exists()).toBe(true)
    const tabLabels = wrapper
      .findAll('.product-tabbed-workspace__tab-label')
      .map(tab => tab.text())
    expect(tabLabels).toEqual(['概览', '智能问答', '时间线', '续写', '角色工坊'])
    for (const tab of wrapper.findAll('[role="tab"]')) {
      const panelId = tab.attributes('aria-controls')
      expect(panelId).toBeTruthy()
      expect(wrapper.get(`#${panelId}`).attributes('aria-labelledby')).toBe(tab.attributes('id'))
    }
    expect(wrapper.findAll('[role="tabpanel"]')).toHaveLength(5)
    await wrapper.findAll('[role="tab"]')[1]!.trigger('click')
    expect(wrapper.findAll('[role="tabpanel"]')).toHaveLength(5)
    expect(wrapper.get('#product-workspace-panel-overview').attributes('style')).toContain('display: none')
    expect(wrapper.get('#product-workspace-panel-qa').attributes('style') ?? '').not.toContain('display: none')
  })

  it('does not navigate to an invalid book-only translation route', async () => {
    const wrapper = mount(InsightView, {
      global: {
        stubs: insightViewStubs(),
      },
    })
    const insightStore = useInsightStore()
    insightStore.setCurrentBook('book-1')
    insightStore.setChapters([])
    await nextTick()

    const translateAction = wrapper.findAll('button').find(
      button => button.text().includes('翻译'),
    )
    expect(translateAction).toBeDefined()
    await translateAction!.trigger('click')

    expect(routerPushMock).not.toHaveBeenCalled()
    expect(showToastMock).toHaveBeenCalledWith(
      '当前书籍还没有章节，请先在书架中创建章节',
      'warning',
    )
  })

  it('uses the shared product three-pane workspace shell for selected-book layout', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    setRouteQuery({ book: 'book-1' })

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await flushPromises()

    expect(wrapper.find('.product-three-pane-workspace').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__pane--left .insight-view__book-summary').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__main .product-tabbed-workspace').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__pane--right').exists()).toBe(true)
  })

  it('keeps the selected-book three-pane layout on the drawer breakpoint contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).toContain('mobile-mode="drawer"')
    expect(source).toContain('@media (--breakpoint-lg-down)')
    expect(source).toMatch(/@media \(--breakpoint-lg-down\)[\s\S]*\.insight-view__mobile-nav-button[\s\S]*display: inline-flex/)
    expect(source).not.toContain('@media (--breakpoint-md-up)')
  })

  it('delegates mobile drawer trigger chrome to the icon-button primitive', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const mobileButtonBlock = style.match(/\.insight-view__mobile-nav-button\s*\{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(source).toContain('<UiIconButton')
    expect(source).toContain(':active="showMobileSidebar"')
    expect(source).toContain(':pressed="showMobileSidebar"')
    expect(source).toContain(':active="showMobileWorkspace"')
    expect(source).toContain(':pressed="showMobileWorkspace"')
    expect(source).not.toContain('variant="toolbar"\n              class="insight-view__mobile-nav-button"')
    expect(source).not.toContain('.insight-page .insight-view__mobile-nav-button:hover')
    expect(source).not.toContain('.insight-page .insight-view__mobile-nav-button.active')
    expect(mobileButtonBlock).not.toMatch(/\b(width|height|border|border-radius|background|color|cursor|font-size|transition)\s*:/)
  })

  it('keeps all book-owned panels unmounted until the core book snapshot is complete', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    setRouteQuery({ book: 'book-1' })
    const bookLoad = createDeferred<BookDetailSuccess>()
    getBookDetailMock.mockReturnValueOnce(bookLoad.promise)

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await nextTick()

    expect(wrapper.text()).toContain('正在读取书籍详情与分析状态')
    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(false)
    expect(wrapper.find('.product-three-pane-workspace__pane--right').exists()).toBe(false)

    bookLoad.resolve(createBook('book-1', 'First Book'))
    await flushPromises()

    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(true)
    expect(wrapper.find('.product-three-pane-workspace__pane--right').exists()).toBe(true)
  })

  it('fails the core load instead of presenting fallback chapter facts', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    setRouteQuery({ book: 'book-1' })
    getInsightChaptersMock.mockRejectedValueOnce(new Error('章节状态读取失败'))

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await flushPromises()

    const insightStore = useInsightStore()
    expect(insightStore.error).toBe('章节状态读取失败')
    expect(insightStore.chapters).toEqual([])
    expect(wrapper.text()).toContain('书籍加载失败')
    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(false)
    expect(wrapper.find('.product-three-pane-workspace__pane--right').exists()).toBe(false)
  })

  it('clears the selected-book projection when the route removes its book identity', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    setRouteQuery({ book: 'book-1' })

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await flushPromises()
    expect(useInsightStore().currentBookId).toBe('book-1')

    setRouteQuery({})
    await nextTick()

    expect(useInsightStore().currentBookId).toBeNull()
    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(false)
    expect(wrapper.find('.product-three-pane-workspace__pane--right').exists()).toBe(false)
    expect(wrapper.text()).toContain('选择要分析的书籍')
  })

  it('reprocesses a terminal event after its durable job projection arrives', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs(),
      },
    })
    await flushPromises()

    const insightStore = useInsightStore()
    const taskCenterStore = useTaskCenterStore()
    insightStore.setCurrentBook('book-1')
    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.latestEvent = {
      eventId: 106,
      jobId: 'derived-job-late',
      type: 'job_finished',
      payload: { status: 'completed' },
      createdAt: null,
    }
    await nextTick()
    expect(insightStore.dataRefreshKey).toBe(refreshKeyBefore)

    taskCenterStore.history = [insightJob({
      jobId: 'derived-job-late',
      kind: 'derived_rebuild',
      status: 'completed',
    })]
    await nextTick()

    expect(insightStore.dataRefreshKey).toBe(refreshKeyBefore + 1)
    wrapper.unmount()
  })

  it('ignores stale book load responses when a newer selection finishes first', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()

    const firstLoad = createDeferred<BookDetailSuccess>()
    const secondLoad = createDeferred<BookDetailSuccess>()
    getBookDetailMock.mockImplementation((bookId: string) => (
      bookId === 'book-1' ? firstLoad.promise : secondLoad.promise
    ))

    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: insightViewStubs({
          ProductThreePaneWorkspace: {
            template: '<main><slot name="left" /><slot /><slot name="right" /></main>',
          },
          BookSelector: {
            template: `
              <div>
                <button data-testid="select-book-1" @click="$emit('select', 'book-1')">Book 1</button>
                <button data-testid="select-book-2" @click="$emit('select', 'book-2')">Book 2</button>
              </div>
            `,
          },
        }),
      },
    })

    const [firstButton, secondButton] = wrapper.findAll('[data-testid^="select-book-"]')
    firstButton!.element.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    secondButton!.element.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await nextTick()

    secondLoad.resolve(createBook('book-2', 'Second Book'))
    await flushPromises()

    expect(wrapper.text()).toContain('Second Book')

    firstLoad.resolve(createBook('book-1', 'First Book'))
    await flushPromises()

    expect(insightStore.currentBookId).toBe('book-2')
    expect(wrapper.text()).toContain('Second Book')
    expect(wrapper.text()).not.toContain('First Book')
    expect(routerReplaceMock).toHaveBeenLastCalledWith({ query: { book: 'book-2' } })
  })
})
