import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { nextTick } from 'vue'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const {
  reanalyzeChapterMock,
  getInsightPagesPageMock,
  showToastMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  reanalyzeChapterMock: vi.fn(),
  getInsightPagesPageMock: vi.fn(),
  showToastMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  reanalyzeChapter: reanalyzeChapterMock,
  getInsightPagesPage: getInsightPagesPageMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import PagesTree from '@/components/insight/PagesTree.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

function pageSummary(
  pageNumber: number,
  options: { analyzed?: boolean; bookId?: string; chapterId?: string } = {},
) {
  const bookId = options.bookId ?? 'book-1'
  return {
    activeAnalysisId: options.analyzed ? `analysis-${pageNumber}` : null,
    analysisState: options.analyzed ? 'ready' : 'not_analyzed',
    chapterId: options.chapterId ?? 'ch-1',
    displayPageNumber: pageNumber,
    pageId: `${bookId}-page-${pageNumber}`,
    sourceAssetId: `${bookId}-asset-${pageNumber}`,
    thumbnailUrl: `/thumb/${bookId}/${pageNumber}`,
  }
}

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
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])

    reanalyzeChapterMock.mockReset()
    reanalyzeChapterMock.mockResolvedValue({
      jobId: 'task-chapter-1',
    })
    getInsightPagesPageMock.mockReset().mockResolvedValue({
      items: [pageSummary(1), pageSummary(2)],
      nextCursor: null,
    })
    showToastMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)

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
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const reanalyzeButton = wrapper.find('button[aria-label="重新分析第1章"]')
    expect(reanalyzeButton.exists()).toBe(true)

    await reanalyzeButton.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '重新分析章节',
      message: '确定要重新分析此章节吗？',
      confirmText: '重新分析',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(reanalyzeChapterMock).toHaveBeenCalledWith('book-1', 'ch-1')
    expect(store.currentTaskId).toBe('task-chapter-1')
    expect(store.analysisStatus).toBe('queued')
    expect(showToastMock).toHaveBeenCalledWith('章节分析已启动', 'success')
  })

  it('refreshes analyzed page markers without routine console output', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])

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
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const chapterToggle = wrapper.find('.pages-tree-panel__chapter-toggle')
    expect(chapterToggle.exists()).toBe(true)
    expect(chapterToggle.element.tagName).toBe('BUTTON')
    expect(chapterToggle.attributes('aria-expanded')).toBe('true')

    const thumbnailGrid = wrapper.getComponent(VirtualThumbnailGrid)
    expect(thumbnailGrid.props('ariaLabel')).toBe('第1章页面导航')
    expect(thumbnailGrid.props('items')).toEqual([
      {
        id: 1,
        src: '/thumb/book-1/1',
        alt: '第1页',
        label: '第 1 页',
        selected: false,
        marked: false,
      },
      {
        id: 2,
        src: '/thumb/book-1/2',
        alt: '第2页',
        label: '第 2 页',
        selected: false,
        marked: false,
      },
    ])

    thumbnailGrid.vm.$emit('select', 1)
    await nextTick()

    expect(store.selectedPageNum).toBe(1)
    expect(wrapper.getComponent(VirtualThumbnailGrid).props('items')[0]).toMatchObject({
      id: 1,
      selected: true,
    })
  })

  it('uses product cards and action rows for chapter navigation controls', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    expect(wrapper.findComponent(ProductRecordCard).exists()).toBe(true)
    expect(
      wrapper
        .findAllComponents(ProductActionRow)
        .some(row => row.props('ariaLabel') === '第1章章节操作')
    ).toBe(true)
    const reanalyzeAction = wrapper.getComponent(UiIconButton)
    expect(reanalyzeAction.props('label')).toBe('重新分析第1章')
    expect(reanalyzeAction.props('title')).toBe('重新分析此章节')
    expect(wrapper.find('.tree-chapter-header').exists()).toBe(false)
    expect(wrapper.find('.btn-reanalyze-chapter').exists()).toBe(false)

    const emptyTreePinia = createPinia()
    setActivePinia(emptyTreePinia)
    const emptyTreeStore = useInsightStore()
    emptyTreeStore.currentBookId = 'book-1'
    emptyTreeStore.setBookTotalPages(101)
    emptyTreeStore.setChapters([])
    getInsightPagesPageMock.mockResolvedValueOnce({
      items: Array.from({ length: 100 }, (_, index) =>
        pageSummary(index + 1, { chapterId: '' })
      ),
      nextCursor: 100,
    })

    const emptyTreeWrapper = mount(PagesTree, {
      global: {
        plugins: [emptyTreePinia],
      },
    })
    await flushPromises()

    expect(emptyTreeWrapper.getComponent(VirtualThumbnailGrid).props('items')).toHaveLength(100)
    expect(emptyTreeWrapper.find('.pages-tree-panel__load-more').exists()).toBe(true)
  })

  it('uses product chips for page counts and chapter analysis state', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([{ id: 'ch-1', title: '第1章', startPage: 1, endPage: 2, analyzed: false }])
    getInsightPagesPageMock.mockResolvedValueOnce({
      items: [pageSummary(1, { analyzed: true }), pageSummary(2, { analyzed: true })],
      nextCursor: null,
    })

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const chipLists = wrapper.findAllComponents(ProductChipList)
    expect(chipLists.some(chipList => chipList.props('ariaLabel') === '内容导航统计')).toBe(true)
    expect(chipLists.some(chipList => chipList.props('ariaLabel') === '第1章章节状态')).toBe(true)

    const chapterStatus = chipLists.find(
      chipList => chipList.props('ariaLabel') === '第1章章节状态'
    )
    expect(chapterStatus?.props('items')).toEqual([
      {
        id: 'ch-1-pages',
        label: '2页',
        tone: 'neutral',
      },
      {
        id: 'ch-1-analysis',
        label: '已分析',
        tone: 'success',
      },
    ])
    expect(wrapper.find('.page-count-badge').exists()).toBe(false)
    expect(wrapper.find('.tree-chapter-status').exists()).toBe(false)
  })

  it('uses the product section header for content navigation statistics', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/PagesTree.vue'),
      'utf8'
    )
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({
      title: '内容导航',
      size: 'sm',
    })
    expect(header.findComponent(ProductChipList).props('ariaLabel')).toBe('内容导航统计')
    expect(source).toContain(
      "import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'"
    )
    expect(source).not.toContain('class="section-header"')
    expect(source).not.toContain('class="section-title"')
    expect(source).not.toContain('.section-header')
    expect(source).not.toContain('.section-title')
  })

  it('renders the no-pages state through product status feedback', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(0)
    store.setChapters([])

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/PagesTree.vue'),
      'utf8'
    )
    const emptyState = wrapper.getComponent(ProductStatusBanner)

    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      role: 'note',
      iconName: 'image',
      title: '暂无页面',
    })
    expect(wrapper.text()).toContain('导入或选择书籍后将在这里显示页面缩略图。')
    expect(source).toContain(
      "import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'"
    )
    expect(source).not.toContain('empty-hint')
  })

  it('keeps chapter metadata on product chip contracts instead of local status classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/PagesTree.vue'),
      'utf8'
    )

    expect(source).toContain('ProductChipList')
    expect(source).not.toContain('page-count-badge')
    expect(source).not.toContain('tree-chapter-status')
    expect(source).not.toContain('tree-chapter-meta')
  })

  it('uses pages-tree-panel owner hooks for content navigation styling', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/PagesTree.vue'),
      'utf8'
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const oldHooks = [
      'sidebar-section',
      'pages-tree-section',
      'pages-tree',
      'pages-tree-header',
      'tree-header-chips',
      'tree-empty-status',
      'tree-all-pages',
      'tree-load-more',
      'tree-chapter',
      'tree-chapter-main',
      'tree-chapter-toggle',
      'tree-expand-icon',
      'tree-chapter-title',
      'tree-chapter-chips',
      'tree-pages-grid',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('class="pages-tree-panel"')
    expect(source).toContain('pages-tree-panel__chapter-toggle')
    expect(source).toContain('pages-tree-panel__pages-grid')
    expect(styleBlock).not.toMatch(/\.tree-/)
  })

  it('ignores stale page-summary responses after switching books', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(2)
    store.setChapters([])

    const book1Pages = deferred<{ items: ReturnType<typeof pageSummary>[]; nextCursor: null }>()
    const book2Pages = deferred<{ items: ReturnType<typeof pageSummary>[]; nextCursor: null }>()
    getInsightPagesPageMock
      .mockReset()
      .mockReturnValueOnce(book1Pages.promise)
      .mockReturnValueOnce(book2Pages.promise)

    const wrapper = mount(PagesTree, {
      global: {
        plugins: [pinia],
      },
    })
    await nextTick()
    expect(getInsightPagesPageMock).toHaveBeenCalledWith('book-1', {
      cursor: 0,
      limit: 100,
    })

    store.currentBookId = 'book-2'
    await nextTick()
    expect(getInsightPagesPageMock).toHaveBeenCalledWith('book-2', {
      cursor: 0,
      limit: 100,
    })

    book2Pages.resolve({
      items: [
        pageSummary(1, { bookId: 'book-2', chapterId: '' }),
        pageSummary(2, { analyzed: true, bookId: 'book-2', chapterId: '' }),
      ],
      nextCursor: null,
    })
    await flushPromises()

    const pageItemsAfterBook2 = wrapper.getComponent(VirtualThumbnailGrid).props('items')
    expect(pageItemsAfterBook2[0]).toMatchObject({ id: 1, marked: false })
    expect(pageItemsAfterBook2[1]).toMatchObject({ id: 2, marked: true })

    book1Pages.resolve({
      items: [
        pageSummary(1, { analyzed: true, bookId: 'book-1', chapterId: '' }),
        pageSummary(2, { bookId: 'book-1', chapterId: '' }),
      ],
      nextCursor: null,
    })
    await flushPromises()

    const pageItemsAfterStaleBook1 = wrapper.getComponent(VirtualThumbnailGrid).props('items')
    expect(pageItemsAfterStaleBook1[0]).toMatchObject({ id: 1, marked: false })
    expect(pageItemsAfterStaleBook1[1]).toMatchObject({ id: 2, marked: true })
  })

  it('keeps a 1000-page expanded chapter to a bounded thumbnail DOM window', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(1000)
    store.setChapters([
      { id: 'ch-large', title: '大章节', startPage: 1, endPage: 1000, analyzed: false },
    ])
    getInsightPagesPageMock.mockResolvedValueOnce({
      items: Array.from({ length: 100 }, (_, index) =>
        pageSummary(index + 1, { chapterId: 'ch-large' })
      ),
      nextCursor: 100,
    })

    const wrapper = mount(PagesTree, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const grid = wrapper.getComponent(VirtualThumbnailGrid)
    expect(grid.props('items')).toHaveLength(100)
    expect(grid.findAll('[data-product-thumbnail-id]').length).toBeLessThanOrEqual(8)
    expect(wrapper.find('.pages-tree-panel__load-more').exists()).toBe(true)
  })

  it('distinguishes a partially analyzed chapter from an untouched chapter', async () => {
    getInsightPagesPageMock.mockResolvedValueOnce({
      items: [pageSummary(1, { analyzed: true }), pageSummary(2)],
      nextCursor: null,
    })

    const wrapper = mount(PagesTree)
    await flushPromises()

    expect(wrapper.text()).toContain('部分分析')
    expect(wrapper.text()).not.toContain('待分析')
  })
})
