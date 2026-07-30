import { nextTick } from 'vue'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount, shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { V2Job } from '@/api/v2/jobs'
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

function stubBookshelfLoadBooks() {
  const bookshelfStore = useBookshelfStore()
  vi.spyOn(bookshelfStore, 'loadBooks').mockResolvedValue(undefined)
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
      pools: [],
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

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.setAnalyzedPagesCount(0)
    insightStore.dataRefreshKey = 0

    stubBookshelfLoadBooks()

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
    vi.clearAllMocks()
  })

  it('projects active progress and refreshes backend facts on a terminal task event', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.dataRefreshKey = 0

    stubBookshelfLoadBooks()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
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
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    await flushPromises()
    const taskCenterStore = useTaskCenterStore()
    const refreshKeyBefore = insightStore.dataRefreshKey
    taskCenterStore.queue = [insightJob()]
    await nextTick()

    expect(insightStore.analysisStatus).toBe('running')
    expect(insightStore.currentTaskId).toBe('analysis-job-1')
    expect(insightStore.progress.current).toBe(4)

    taskCenterStore.latestEvent = {
      eventId: 101,
      jobId: 'analysis-job-1',
      type: 'job_finished',
      payload: {},
      createdAt: null,
    }
    await flushPromises()

    expect(getAnalysisStatusMock).toHaveBeenCalledTimes(1)
    expect(getAnalysisStatusMock).toHaveBeenCalledWith('book-1')
    expect(insightStore.analysisStatus).toBe('completed')
    expect(insightStore.currentTaskId).toBeNull()
    expect(insightStore.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('does not present a recoverable interrupted job as actively running', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('failed')
    insightStore.setCurrentTaskId(null)
    stubBookshelfLoadBooks()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
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
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    await flushPromises()
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.queue = [insightJob({ status: 'interrupted' })]
    await nextTick()

    expect(insightStore.analysisStatus).toBe('failed')
    expect(insightStore.currentTaskId).toBeNull()
  })

  it('refreshes derived Insight facts when a backend rebuild finishes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.dataRefreshKey = 0
    stubBookshelfLoadBooks()

    shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
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
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    await flushPromises()
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
    insightStore.currentBookId = 'book-1'
    insightStore.setAnalysisStatus('idle')
    insightStore.dataRefreshKey = 0

    stubBookshelfLoadBooks()

    const wrapper = shallowMount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
          ProductHeaderAction: false,
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
          'router-link': { template: '<a><slot /></a>' },
        },
      },
    })

    await flushPromises()
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
        stubs: {
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
          'router-link': { template: '<a><slot /></a>' },
        },
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

  it('keeps task event projection typed and free of page-local polling', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/insight-view.task-events.spec.ts'), 'utf8')
    const viewSource = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
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

  it('uses the shared bookshelf normalizer instead of reading wire aliases in the page view', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/InsightView.vue'), 'utf8')

    expect(source).toContain("import { normalizeBookData } from '@/utils/bookshelfModels'")
    expect(source).not.toContain('function getChapterPageCount')
    expect(source).not.toContain('chapter.page_count')
    expect(source).not.toContain('chapter.image_count')
    expect(source).not.toContain('book.total_pages')
    expect(source).not.toContain('currentBook?.total_pages')
    expect(source).toContain('normalizedBook.totalPages')
    expect(source).toContain('currentBook?.totalPages')
  })

  it('uses the shared product tabbed workspace shell for selected-book panels', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
          AppShell: { template: '<section><slot name="header" /><slot /></section>' },
          ProductPageHeader: ProductPageHeaderStub,
          SidebarLayout: { template: '<main><slot /></main>' },
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
        },
      },
    })

    expect(wrapper.find('.product-tabbed-workspace').exists()).toBe(true)
    expect(wrapper.find('.product-tabbed-workspace__panels > .insight-view__tab-content').exists()).toBe(true)
    const tabLabels = wrapper
      .findAll('.product-tabbed-workspace__tab-label')
      .map(tab => tab.text())
    expect(tabLabels).toEqual(['概览', '智能问答', '时间线', '续写', '角色工坊'])
  })

  it('uses the shared product three-pane workspace shell for selected-book layout', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'

    const wrapper = mount(InsightView, {
      global: {
        plugins: [pinia],
        stubs: {
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
        },
      },
    })

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

  it('ignores stale book load responses when a newer selection finishes first', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    stubBookshelfLoadBooks()

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
          ProductPageHeader: ProductPageHeaderStub,
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
