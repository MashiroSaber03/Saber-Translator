import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { useInsightStore } from '@/stores/insightStore'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

const {
  downloadCurrentOverviewMock,
  exportAnalysisMock,
  getGeneratedTemplatesMock,
  getOverviewMock,
  getRecentAnalyzedPagesMock,
  regenerateOverviewMock,
  showToastMock,
  triggerBlobDownloadMock,
} = vi.hoisted(() => ({
  downloadCurrentOverviewMock: vi.fn(),
  exportAnalysisMock: vi.fn(),
  getGeneratedTemplatesMock: vi.fn(),
  getOverviewMock: vi.fn(),
  getRecentAnalyzedPagesMock: vi.fn(),
  regenerateOverviewMock: vi.fn(),
  showToastMock: vi.fn(),
  triggerBlobDownloadMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  downloadCurrentOverview: downloadCurrentOverviewMock,
  exportAnalysis: exportAnalysisMock,
  regenerateOverview: regenerateOverviewMock,
  getGeneratedTemplates: getGeneratedTemplatesMock,
  getRecentAnalyzedPages: getRecentAnalyzedPagesMock,
  getOverview: getOverviewMock,
}))

vi.mock('@/utils/browserDownload', () => ({
  triggerBlobDownload: triggerBlobDownloadMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

vi.mock('marked', () => ({
  marked: {
    parse: (value: string) => value,
  },
}))

import OverviewPanel from '@/components/insight/OverviewPanel.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

describe('OverviewPanel', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.dataRefreshKey = 0

    getGeneratedTemplatesMock.mockReset().mockResolvedValue(['story_summary'])
    getRecentAnalyzedPagesMock.mockReset().mockResolvedValue([
      { page_num: 2, summary: '第 2 页' },
    ])
    getOverviewMock.mockReset().mockResolvedValue('缓存中的故事概要')
    regenerateOverviewMock.mockReset().mockResolvedValue({
      kind: 'queued',
      jobId: 'overview-job-1',
    })
    downloadCurrentOverviewMock.mockReset().mockResolvedValue(new Blob(['overview']))
    exportAnalysisMock.mockReset().mockResolvedValue({ jobId: 'export-job-1' })
    showToastMock.mockReset()
    triggerBlobDownloadMock.mockReset()
  })

  it('queues story_summary regeneration without fabricating a completion refresh', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    const refreshKeyBefore = store.dataRefreshKey

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const templateSelect = wrapper.getComponent(UiSelect)
    templateSelect.vm.$emit('update:modelValue', 'story_summary')
    templateSelect.vm.$emit('change', 'story_summary')
    await flushPromises()

    await wrapper.findAllComponents(UiIconButton)[1]!.trigger('click')
    await flushPromises()

    expect(regenerateOverviewMock).toHaveBeenCalledWith('book-1', 'story_summary', true)
    expect(store.dataRefreshKey).toBe(refreshKeyBefore)
  })

  it('shows durable queued feedback while a new overview is generated', async () => {
    regenerateOverviewMock.mockResolvedValueOnce({
      kind: 'queued',
      jobId: 'overview-job-1',
    })

    const wrapper = mount(OverviewPanel)
    await flushPromises()

    await wrapper.findAllComponents(UiIconButton)[0]!.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('概览生成中')
    expect(wrapper.text()).toContain('概览生成已进入任务中心，完成后将自动加载。')
    expect(wrapper.text()).not.toContain('尚未生成概览')
  })

  it('reloads generated overview data without routine console output when refresh key changes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      store.triggerDataRefresh()
      await flushPromises()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })

  it('sanitizes cached overview markdown before rendering', async () => {
    getGeneratedTemplatesMock.mockResolvedValue(['no_spoiler'])
    getOverviewMock.mockResolvedValue(
      [
        '<script>alert("xss")</script>',
        '<a href="javascript:alert(1)">bad link</a>',
        '<a href="https://safe.example">safe link</a>',
      ].join(''),
    )

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.html()).not.toContain('javascript:')
    const safeLink = wrapper.find('a[href="https://safe.example"]')
    expect(safeLink.exists()).toBe(true)
    expect(safeLink.attributes('rel')).toBe('noopener noreferrer')
  })

  it('uses button semantics for recent analyzed page shortcuts', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(5)
    store.setAnalyzedPagesCount(2)

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const recentItem = wrapper.get('[aria-label="查看第 2 页分析详情"]')
    expect(recentItem.element.tagName).toBe('BUTTON')
    expect(recentItem.attributes('type')).toBe('button')
    expect(recentItem.attributes('aria-label')).toBe('查看第 2 页分析详情')

    await recentItem.trigger('click')

    expect(store.selectedPageNum).toBe(2)
  })

  it('uses product action primitives for overview commands and recent pages', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setBookTotalPages(5)
    store.setAnalyzedPagesCount(2)

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/OverviewPanel.vue'),
      'utf8'
    )

    expect(wrapper.findAllComponents(UiIconButton)).toHaveLength(2)
    expect(wrapper.findAllComponents(ProductActionRow).some(row => row.props('ariaLabel') === '概览导出操作')).toBe(true)
    expect(wrapper.findComponent(ProductRecordCard).props('as')).toBe('button')
    expect(source).not.toContain('class="button-icon"')
    expect(source).not.toContain('overview-action-button')
    expect(source).not.toContain('recent-page-item')
    expect(source).toContain('class="overview-panel"')
    expect(source).toContain('overview-panel__card')
    expect(source).not.toMatch(/\.(?:overview-grid|overview-card|card-header|card-title|card-content|markdown-content|stat-item|stat-value|stat-label|export-actions|recent-page-card|recent-page-content|page-number|page-summary)\b/)
  })

  it('names the template selector when it is rendered outside a visible field label', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const templateSelect = wrapper.getComponent(UiSelect)
    expect(templateSelect.get('[role="combobox"]').attributes('aria-label')).toBe('选择概览模板')
  })

  it('keeps the overview cards on a responsive grid contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/OverviewPanel.vue'),
      'utf8'
    )
    const gridStyle = source.match(/\.overview-panel \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''
    const summaryStyle = source.match(/\.overview-panel__card--summary \{(?<body>[\s\S]*?)\n\}/)?.groups?.body ?? ''

    expect(gridStyle).toContain('repeat(auto-fit')
    expect(gridStyle).toContain('minmax(min(100%, 280px), 1fr)')
    expect(gridStyle).not.toContain('repeat(2, 1fr)')
    expect(summaryStyle).toContain('grid-column: 1 / -1')
    expect(summaryStyle).not.toContain('grid-column: span 2')
  })

  it('uses product status feedback for overview loading and empty states', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/OverviewPanel.vue'),
      'utf8'
    )
    expect(source).not.toContain('placeholder-text')
    expect(source).not.toContain('loading-text')
    expect(source).toContain('ProductStatusBanner')

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalyzedPagesCount(0)

    getGeneratedTemplatesMock.mockResolvedValueOnce([])
    getRecentAnalyzedPagesMock.mockResolvedValueOnce([])
    getOverviewMock.mockResolvedValueOnce(null)

    const emptyWrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const emptyBanners = emptyWrapper.findAllComponents(ProductStatusBanner)
    expect(emptyBanners.map(banner => banner.props('title'))).toEqual(['', ''])
    expect(emptyBanners.map(banner => banner.text())).toEqual([
      '选择模板类型，点击生成按钮',
      '暂无分析记录',
    ])

    const loadingPinia = createPinia()
    setActivePinia(loadingPinia)
    const loadingStore = useInsightStore()
    loadingStore.currentBookId = 'book-1'
    const pendingOverview = deferred<string>()
    getGeneratedTemplatesMock.mockResolvedValueOnce(['no_spoiler'])
    getOverviewMock.mockReturnValueOnce(pendingOverview.promise)

    const loadingWrapper = mount(OverviewPanel, {
      global: {
        plugins: [loadingPinia],
      },
    })
    await flushPromises()

    const loadingBanner = loadingWrapper.findComponent(ProductStatusBanner)
    expect(loadingBanner.props()).toMatchObject({
      ariaLive: 'polite',
      iconName: 'refresh',
      title: '正在加载概览',
      tone: 'neutral',
    })
  })

  it('ignores stale cached overview responses after switching books', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalyzedPagesCount(0)

    const book1Overview = deferred<string>()
    const book2Overview = deferred<string>()
    getGeneratedTemplatesMock.mockReset().mockResolvedValue(['no_spoiler'])
    getOverviewMock.mockReset()
      .mockReturnValueOnce(book1Overview.promise)
      .mockReturnValueOnce(book2Overview.promise)

    const wrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()
    expect(getOverviewMock).toHaveBeenCalledWith('book-1', 'no_spoiler')

    store.currentBookId = 'book-2'
    await nextTick()
    await flushPromises()
    expect(getOverviewMock).toHaveBeenCalledWith('book-2', 'no_spoiler')

    book2Overview.resolve('book-2 overview')
    await flushPromises()
    expect(wrapper.text()).toContain('book-2 overview')

    book1Overview.resolve('book-1 stale overview')
    await flushPromises()

    expect(wrapper.text()).toContain('book-2 overview')
    expect(wrapper.text()).not.toContain('book-1 stale overview')
  })

  it('loads the selected overview directly even when template metadata fails', async () => {
    getGeneratedTemplatesMock.mockRejectedValueOnce(new Error('metadata unavailable'))
    getOverviewMock.mockResolvedValueOnce('direct overview content')

    const wrapper = mount(OverviewPanel)
    await flushPromises()

    expect(wrapper.text()).toContain('direct overview content')
    expect(wrapper.text()).not.toContain('概览操作失败')
  })

  it('shows a retryable error instead of treating a recent-page failure as an empty list', async () => {
    getGeneratedTemplatesMock.mockResolvedValueOnce([])
    getOverviewMock.mockResolvedValueOnce(null)
    getRecentAnalyzedPagesMock.mockRejectedValueOnce(new Error('recent pages unavailable'))

    const wrapper = mount(OverviewPanel)
    await flushPromises()

    const error = wrapper.findAllComponents(ProductStatusBanner)
      .find(banner => banner.props('title') === '最近分析加载失败')
    expect(error).toBeTruthy()
    expect(error?.text()).toContain('recent pages unavailable')
    expect(wrapper.text()).not.toContain('暂无分析记录')
  })

  it('stacks the recent-page retry action below errors in the narrow card', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/OverviewPanel.vue'), 'utf8')
    const errorRule = source.match(/\.overview-panel__recent-error\s*{([\s\S]*?)}/)?.[1]

    expect(errorRule).toContain('--product-status-banner-flex-direction: column')
    expect(errorRule).toContain('--product-status-banner-actions-width: 100%')
  })

  it('clears a stale loading state when the next book has no cached template', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    const book1Overview = deferred<string>()
    getGeneratedTemplatesMock.mockReset()
      .mockResolvedValueOnce(['no_spoiler'])
      .mockResolvedValueOnce([])
    getOverviewMock.mockReset().mockReturnValueOnce(book1Overview.promise)

    const wrapper = mount(OverviewPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('正在加载概览')

    store.currentBookId = 'book-2'
    await nextTick()
    await flushPromises()

    expect(wrapper.text()).not.toContain('正在加载概览')
    expect(wrapper.text()).toContain('选择模板类型，点击生成按钮')

    book1Overview.resolve('book-1 stale overview')
    await flushPromises()
    expect(wrapper.text()).not.toContain('book-1 stale overview')
  })

  it('prevents duplicate overview generation commands', async () => {
    const queued = deferred<{ kind: 'queued'; jobId: string }>()
    regenerateOverviewMock.mockReset().mockReturnValue(queued.promise)
    const wrapper = mount(OverviewPanel)
    await flushPromises()

    const generateButton = wrapper.findAllComponents(UiIconButton)[0]!
    await generateButton.trigger('click')
    await generateButton.trigger('click')

    expect(regenerateOverviewMock).toHaveBeenCalledTimes(1)
    queued.resolve({ kind: 'queued', jobId: 'overview-job-1' })
    await flushPromises()
  })

  it('keeps load failures out of exportable overview content', async () => {
    getGeneratedTemplatesMock.mockResolvedValue(['no_spoiler'])
    getOverviewMock.mockRejectedValue(new Error('temporary provider failure'))
    const wrapper = mount(OverviewPanel)
    await flushPromises()

    expect(wrapper.text()).toContain('概览操作失败')
    expect(wrapper.text()).toContain('temporary provider failure')
    const exportCurrentButton = wrapper.findAllComponents({ name: 'UiButton' })[0]!
    expect(exportCurrentButton.attributes('disabled')).toBeDefined()
  })

  it('captures the requested book and template for one current-overview export', async () => {
    getGeneratedTemplatesMock.mockResolvedValue(['no_spoiler'])
    getOverviewMock.mockResolvedValue('overview body')
    const download = deferred<Blob>()
    downloadCurrentOverviewMock.mockReset().mockReturnValue(download.promise)
    const wrapper = mount(OverviewPanel)
    await flushPromises()

    const exportCurrentButton = wrapper.findAllComponents({ name: 'UiButton' })[0]!
    await exportCurrentButton.trigger('click')
    const store = useInsightStore()
    store.currentBookId = 'book-2'
    await nextTick()
    await exportCurrentButton.trigger('click')

    expect(downloadCurrentOverviewMock).toHaveBeenCalledTimes(1)
    expect(downloadCurrentOverviewMock).toHaveBeenCalledWith('book-1', 'no_spoiler')

    const blob = new Blob(['overview'])
    download.resolve(blob)
    await flushPromises()
    expect(triggerBlobDownloadMock).toHaveBeenCalledWith(blob, 'book-1_no_spoiler.md')
  })
})
