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

const { regenerateOverviewMock, getGeneratedTemplatesMock, getAnalysisStatusMock, getOverviewMock } = vi.hoisted(() => ({
  regenerateOverviewMock: vi.fn(),
  getGeneratedTemplatesMock: vi.fn(),
  getAnalysisStatusMock: vi.fn(),
  getOverviewMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  regenerateOverview: regenerateOverviewMock,
  getGeneratedTemplates: getGeneratedTemplatesMock,
  getAnalysisStatus: getAnalysisStatusMock,
  getOverview: getOverviewMock,
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

    getGeneratedTemplatesMock.mockReset().mockResolvedValue({
      success: true,
      generated: ['story_summary'],
    })
    getAnalysisStatusMock.mockReset().mockResolvedValue({
      success: true,
      analyzed_pages_count: 5,
    })
    getOverviewMock.mockReset().mockResolvedValue({
      success: true,
      content: '缓存中的故事概要',
    })
    regenerateOverviewMock.mockReset().mockResolvedValue({
      success: true,
      content: '重新生成的故事概要',
    })
  })

  it('broadcasts a refresh when story_summary is regenerated', async () => {
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
    expect(store.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('shows durable queued feedback while a new overview is generated', async () => {
    regenerateOverviewMock.mockResolvedValueOnce({
      success: true,
      task_id: 'overview-job-1',
      message: '概览重建已进入任务中心',
    })

    const wrapper = mount(OverviewPanel)
    await flushPromises()

    await wrapper.findAllComponents(UiIconButton)[0]!.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('概览生成中')
    expect(wrapper.text()).toContain('概览重建已进入任务中心')
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
    getGeneratedTemplatesMock.mockResolvedValue({
      success: true,
      generated: ['no_spoiler'],
    })
    getOverviewMock.mockResolvedValue({
      success: true,
      content: [
        '<script>alert("xss")</script>',
        '<a href="javascript:alert(1)">bad link</a>',
        '<a href="https://safe.example">safe link</a>',
      ].join(''),
    })

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
    expect(templateSelect.attributes('aria-label')).toBe('选择概览模板')
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

    getGeneratedTemplatesMock.mockResolvedValueOnce({
      success: true,
      generated: [],
    })

    const emptyWrapper = mount(OverviewPanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const emptyBanners = emptyWrapper.findAllComponents(ProductStatusBanner)
    expect(emptyBanners.map(banner => banner.props('title'))).toEqual([
      '尚未生成概览',
      '暂无分析记录',
    ])

    const loadingPinia = createPinia()
    setActivePinia(loadingPinia)
    const loadingStore = useInsightStore()
    loadingStore.currentBookId = 'book-1'
    const pendingOverview = deferred<{ success: true; content: string }>()
    getGeneratedTemplatesMock.mockResolvedValueOnce({
      success: true,
      generated: ['no_spoiler'],
    })
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

    const book1Overview = deferred<{ success: true; content: string }>()
    const book2Overview = deferred<{ success: true; content: string }>()
    getGeneratedTemplatesMock.mockReset().mockResolvedValue({
      success: true,
      generated: ['no_spoiler'],
    })
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

    book2Overview.resolve({
      success: true,
      content: 'book-2 overview',
    })
    await flushPromises()
    expect(wrapper.text()).toContain('book-2 overview')

    book1Overview.resolve({
      success: true,
      content: 'book-1 stale overview',
    })
    await flushPromises()

    expect(wrapper.text()).toContain('book-2 overview')
    expect(wrapper.text()).not.toContain('book-1 stale overview')
  })
})
