import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { useInsightStore } from '@/stores/insightStore'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

const {
  getPageDataMock,
  reanalyzePageMock,
  trackJobMock,
  downloadPageAnalysisMock,
  triggerBlobDownloadMock,
} = vi.hoisted(() => ({
  getPageDataMock: vi.fn(),
  reanalyzePageMock: vi.fn(),
  trackJobMock: vi.fn(),
  downloadPageAnalysisMock: vi.fn(),
  triggerBlobDownloadMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getPageData: getPageDataMock,
  reanalyzePage: reanalyzePageMock,
  downloadPageAnalysis: downloadPageAnalysisMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({ trackJob: trackJobMock }),
}))

vi.mock('@/utils/browserDownload', () => ({
  triggerBlobDownload: triggerBlobDownloadMock,
}))

import PageDetail from '@/components/insight/PageDetail.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

describe('PageDetail', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    store.setAnalysisStatus('idle')
    store.dataRefreshKey = 0

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValue({
      analysis: {
        page_num: 3,
        analysisState: 'ready',
        page_summary: '旧摘要',
        panels: [],
      },
      sourceUrl: '/page.png',
    })

    reanalyzePageMock.mockReset()
    reanalyzePageMock.mockResolvedValue({
      jobId: 'task-123',
    })
    trackJobMock.mockReset()
    downloadPageAnalysisMock.mockReset()
    downloadPageAnalysisMock.mockResolvedValue(new Blob(['analysis']))
    triggerBlobDownloadMock.mockReset()

  })

  it('maps preview and status owner colors through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).not.toContain('--page-detail-error-')
    expect(styleBlock).toContain('--color-status-success')
    expect(styleBlock).toContain('--color-overlay-scrim')
    expect(styleBlock).toContain('color-mix(in srgb')
  })

  it('uses page-detail-panel owner hooks for structural styling', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const oldHooks = [
      'loading-state',
      'page-detail-loading-indicator',
      'page-detail-content',
      'page-detail-header',
      'page-nav-buttons',
      'page-indicator',
      'page-detail-error-feedback',
      'page-detail-image',
      'page-detail-image__fallback',
      'image-overlay',
      'zoom-hint',
      'analysis-status-tag',
      'page-summary',
      'page-summary-feedback',
      'scene-mood-info',
      'info-item',
      'info-label',
      'info-value',
      'dialogues-section',
      'dialogue-feedback',
      'dialogue-item',
      'dialogue-speaker',
      'dialogue-text',
      'dialogue-original',
      'original-label',
      'page-detail-actions',
      'page-detail-action-spinner',
      'image-preview-modal',
      'image-preview-content',
      'preview-close',
      'preview-nav',
      'preview-nav-btn',
      'preview-page-info',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('page-detail-panel__header')
    expect(source).toContain('page-detail-panel__image-trigger')
    expect(source).toContain('page-detail-panel__preview-nav')
    expect(styleBlock).not.toMatch(/\.page-detail-panel\s+\./)
  })

  it('starts async reanalyze and refreshes on dataRefreshKey', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    store.setAnalysisStatus('idle')
    store.dataRefreshKey = 0

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    expect(getPageDataMock).toHaveBeenCalledTimes(1)

    const reanalyzeButton = wrapper.findAll('button').find(button => button.text().includes('重新分析'))
    expect(reanalyzeButton).toBeTruthy()

    await reanalyzeButton!.trigger('click')
    await flushPromises()

    expect(reanalyzePageMock).toHaveBeenCalledWith('book-1', 3)
    expect(trackJobMock).toHaveBeenCalledWith('task-123')
    expect(store.currentTaskId).toBeNull()
    expect(store.analysisStatus).toBe('idle')

    // 不应在启动后立即当作同步完成并刷新详情
    expect(getPageDataMock).toHaveBeenCalledTimes(1)

    store.triggerDataRefresh()
    await nextTick()
    await flushPromises()

    // 分析完成信号到达后自动刷新当前页详情
    expect(getPageDataMock).toHaveBeenCalledTimes(2)
  })

  it('submits page reanalysis once and ignores its result after changing context', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.selectPage(3)
    store.setBookTotalPages(20)
    const submission = deferred<{ jobId: string }>()
    reanalyzePageMock.mockReturnValueOnce(submission.promise)

    const wrapper = mount(PageDetail, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const reanalyzeButton = wrapper.findAll('button')
      .find(button => button.text().includes('重新分析'))!
    await reanalyzeButton.trigger('click')
    await reanalyzeButton.trigger('click')
    expect(reanalyzePageMock).toHaveBeenCalledTimes(1)

    store.setCurrentBook('book-2')
    submission.resolve({ jobId: 'book-1-page-3-job' })
    await flushPromises()

    expect(store.currentTaskId).toBeNull()
    expect(store.analysisStatus).toBe('idle')
  })

  it('downloads with the accepted page identity even if navigation changes mid-request', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.selectPage(3)
    store.setBookTotalPages(20)
    const download = deferred<Blob>()
    downloadPageAnalysisMock.mockReturnValueOnce(download.promise)

    const wrapper = mount(PageDetail, {
      global: { plugins: [pinia] },
    })
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text().includes('导出此页'))!.trigger('click')

    store.selectPage(4)
    const blob = new Blob(['page-3'])
    download.resolve(blob)
    await flushPromises()

    expect(downloadPageAnalysisMock).toHaveBeenCalledWith('book-1', 3)
    expect(triggerBlobDownloadMock).toHaveBeenCalledWith(blob, 'book-1_page_3.md')
  })

  it('uses the shared spinner primitive for the reanalyze action loading state', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('btn-spinner')

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    store.setAnalysisStatus('idle')
    const submission = deferred<{ jobId: string }>()
    reanalyzePageMock.mockReturnValueOnce(submission.promise)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const reanalyzeButton = wrapper.findAll('button').find(button => button.text().includes('重新分析'))
    expect(reanalyzeButton).toBeTruthy()

    await reanalyzeButton!.trigger('click')
    await nextTick()

    const runningButton = wrapper.findAll('button').find(button => button.text().includes('启动中...'))
    expect(runningButton).toBeTruthy()
    const spinner = runningButton!.getComponent(UiSpinner)
    expect(spinner.props('decorative')).toBe(true)
    submission.resolve({ jobId: 'task-123' })
    await flushPromises()
  })

  it('uses shared button variants for page navigation controls', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('btn-page-nav')
    expect(source).not.toContain(':class="{ disabled: !hasPrevPage }"')
    expect(source).not.toContain(':class="{ disabled: !hasNextPage }"')

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const [prevButton, nextButton] = wrapper.findAllComponents(UiButton)
      .filter(button => button.element.closest('.page-detail-panel__nav-buttons'))
    expect(prevButton?.props()).toMatchObject({ variant: 'secondary', size: 'xs' })
    expect(nextButton?.props()).toMatchObject({ variant: 'secondary', size: 'xs' })
  })

  it('lets the page detail header wrap its navigation controls in narrow panels', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    const headerStyle = source.match(/\.page-detail-panel__header \{(?<body>[\s\S]*?)\n\}/)
      ?.groups?.body ?? ''

    expect(headerStyle).toContain('display: flex')
    expect(headerStyle).toContain('flex-wrap: wrap')
    expect(headerStyle).toContain('min-width: 0')
  })

  it('does not override shared button primitive variables at the page root', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    const rootStyle = source.match(/\.page-detail-panel \{(?<body>[\s\S]*?)\n\}/)
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(source).toContain('class="page-detail-panel"')
    expect(source).not.toContain('workspace-section page-detail-section')
    expect(source).not.toContain('.workspace-section.page-detail-section')
  })

  it('reveals the page preview zoom affordance on keyboard focus as well as hover', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')

    expect(source).toContain('.page-detail-panel__image-trigger:focus-visible .page-detail-panel__image-overlay')
    expect(source).toContain('.page-detail-panel__image-trigger:focus-visible .page-detail-panel__zoom-hint')
  })

  it('uses the product section header for the page detail panel heading', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({
      title: '页面详情',
      iconName: 'file-text',
      size: 'sm',
    })
    expect(header.get('.product-section-header__icon-text').text()).toBe('📄')
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="section-title"')
    expect(source).not.toContain('.section-title')
  })

  it('uses the product empty state when no page is selected', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('placeholder-text')
    expect(source).not.toContain('empty-icon')

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectedPageNum = null
    store.setBookTotalPages(20)

    getPageDataMock.mockReset()
    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await nextTick()

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props()).toMatchObject({
      title: '点击左侧导航树中的页面查看详情',
      role: 'note',
    })
    expect(emptyState.get('.product-empty-state__icon-text').text()).toBe('📄')
    expect(getPageDataMock).not.toHaveBeenCalled()
  })

  it('uses product status feedback for page detail load errors', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('class="error-message"')
    expect(source).not.toContain('.error-message')

    getPageDataMock.mockReset()
    getPageDataMock.mockRejectedValueOnce(new Error('加载页面失败'))

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'alert-triangle',
      role: 'alert',
      tone: 'danger',
    })
    expect(banner.text()).toContain('加载页面失败')
  })

  it('uses product status feedback for local page summary and dialogue empty states', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('class="page-summary empty"')
    expect(source).not.toContain('class="dialogues-section empty"')
    expect(source).not.toMatch(/\.page-summary\.empty\b/)
    expect(source).not.toMatch(/\.dialogues-section\.empty\b/)

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValueOnce({
      analysis: {
        analysisState: 'not_analyzed',
        page_num: 3,
        page_summary: '',
        panels: [],
      },
      sourceUrl: '',
    })

    const unanalyzedPinia = createPinia()
    setActivePinia(unanalyzedPinia)
    const unanalyzedStore = useInsightStore()
    unanalyzedStore.currentBookId = 'book-1'
    unanalyzedStore.selectPage(3)
    unanalyzedStore.setBookTotalPages(20)

    const unanalyzedWrapper = mount(PageDetail, {
      global: {
        plugins: [unanalyzedPinia],
      },
    })
    await flushPromises()

    expect(unanalyzedWrapper.getComponent(ProductStatusBanner).props()).toMatchObject({
      iconName: 'file-text',
      role: 'note',
      tone: 'neutral',
    })
    expect(unanalyzedWrapper.getComponent(ProductStatusBanner).props('title')).toBe('')
    expect(unanalyzedWrapper.getComponent(ProductStatusBanner).text()).toBe('此页尚未分析，点击下方按钮开始分析')

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValueOnce({
      analysis: {
        analysisState: 'ready',
        page_num: 3,
        page_summary: '已有摘要',
        panels: [],
      },
      sourceUrl: '',
    })

    const analyzedPinia = createPinia()
    setActivePinia(analyzedPinia)
    const analyzedStore = useInsightStore()
    analyzedStore.currentBookId = 'book-1'
    analyzedStore.selectPage(3)
    analyzedStore.setBookTotalPages(20)

    const analyzedWrapper = mount(PageDetail, {
      global: {
        plugins: [analyzedPinia],
      },
    })
    await flushPromises()

    const banners = analyzedWrapper.findAllComponents(ProductStatusBanner)
    expect(banners.at(-1)?.props()).toMatchObject({
      iconName: 'sparkles',
      role: 'note',
      title: '此页没有识别出关键事件',
      tone: 'neutral',
    })
  })

  it('renders loading feedback through the shared spinner primitive', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('class="loading-spinner"')
    expect(source).not.toContain('.loading-spinner')
    expect(source).toContain('page-detail-panel__loading-indicator')
    expect(source).not.toContain('page-detail-loading-spinner')

    const pendingPage = deferred<{
      analysis: { page_num: number; page_summary: string; panels: never[] }
      sourceUrl: string
    }>()
    getPageDataMock.mockReset()
    getPageDataMock.mockReturnValueOnce(pendingPage.promise)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await nextTick()

    const spinner = wrapper.getComponent(UiSpinner)
    expect(spinner.props('label')).toBe('加载页面详情')
    expect(spinner.props('decorative')).toBe(false)
    expect(wrapper.text()).toContain('加载中...')
  })

  it('opens image preview from an accessible preview trigger', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const host = document.createElement('div')
    document.body.appendChild(host)
    const wrapper = mount(PageDetail, {
      attachTo: host,
      global: {
        plugins: [pinia],
      },
    })

    try {
      await flushPromises()

      const previewTrigger = wrapper.find('.page-detail-panel__image-trigger')
      expect(previewTrigger.element.tagName).toBe('BUTTON')
      expect(previewTrigger.attributes('aria-label')).toBe('预览第 3 页图片')

      previewTrigger.element.focus()
      await previewTrigger.trigger('click')
      await nextTick()

      const previewModal = wrapper.get('.page-detail-panel__image-preview-modal')
      expect(previewModal.exists()).toBe(true)
      expect(previewModal.attributes('role')).toBe('dialog')
      expect(previewModal.attributes('aria-modal')).toBe('true')
      expect(previewModal.attributes('aria-label')).toBe('第 3 页图片预览')
      expect(document.activeElement).toBe(previewModal.element)
      expect(document.body.style.overflow).toBe('hidden')

      const closeButton = wrapper.get('button[aria-label="关闭图片预览"]')
      expect(closeButton.getComponent(UiIcon).props('name')).toBe('x')
      expect(closeButton.text()).not.toContain('×')

      expect(wrapper.get('.page-detail-panel__preview-nav button[title="上一页 (←)"]').attributes('aria-label')).toBe('预览上一页')
      expect(wrapper.get('.page-detail-panel__preview-nav button[title="下一页 (→)"]').attributes('aria-label')).toBe('预览下一页')
      const iconActions = wrapper.findAllComponents(UiIconButton)
      expect(iconActions.some(action => action.props('label') === '关闭图片预览')).toBe(true)
      expect(iconActions.some(action => action.props('label') === '预览上一页')).toBe(true)
      expect(iconActions.some(action => action.props('label') === '预览下一页')).toBe(true)

      await closeButton.trigger('click')
      await nextTick()
      expect(document.activeElement).toBe(previewTrigger.element)
      expect(document.body.style.overflow).toBe('')
    } finally {
      wrapper.unmount()
      host.remove()
    }
  })

  it('renders an accessible fallback instead of mutating image DOM when the page image fails', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')

    expect(source).not.toContain('HTMLImageElement).style.display')

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    await wrapper.get('.page-detail-panel__image-trigger img').trigger('error')
    await nextTick()

    const previewTrigger = wrapper.get('.page-detail-panel__image-trigger')
    expect(previewTrigger.attributes('aria-label')).toBe('第 3 页图片加载失败')
    expect(previewTrigger.attributes('disabled')).toBeDefined()
    expect(wrapper.find('.page-detail-panel__image-trigger img').exists()).toBe(false)
    expect(wrapper.get('.page-detail-panel__image-fallback').text()).toContain('图片加载失败')

    await previewTrigger.trigger('click')
    expect(wrapper.find('.page-detail-panel__image-preview-modal').exists()).toBe(false)
  })

  it('renders a no-image fallback without requesting the current document as an empty image URL', async () => {
    getPageDataMock.mockResolvedValueOnce({
      analysis: {
        analysisState: 'not_analyzed',
        page_num: 3,
        page_summary: '',
      },
      sourceUrl: '',
    })

    const wrapper = mount(PageDetail)
    await flushPromises()

    const previewTrigger = wrapper.get('.page-detail-panel__image-trigger')
    expect(previewTrigger.attributes('aria-label')).toBe('第 3 页暂无图片')
    expect(previewTrigger.attributes('disabled')).toBeDefined()
    expect(wrapper.find('.page-detail-panel__image-trigger img').exists()).toBe(false)
    expect(wrapper.get('.page-detail-panel__image-fallback').text()).toContain('暂无页面图片')
  })

  it('ignores stale page detail responses after selecting another page', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const page3 = deferred<{
      analysis: { page_num: number; page_summary: string; panels: never[] }
      sourceUrl: string
    }>()
    const page4 = deferred<{
      analysis: { page_num: number; page_summary: string; panels: never[] }
      sourceUrl: string
    }>()
    getPageDataMock.mockReset()
    getPageDataMock
      .mockReturnValueOnce(page3.promise)
      .mockReturnValueOnce(page4.promise)

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await nextTick()
    expect(getPageDataMock).toHaveBeenCalledWith('book-1', 3)

    store.selectPage(4)
    await nextTick()
    expect(getPageDataMock).toHaveBeenCalledWith('book-1', 4)

    page4.resolve({
      analysis: {
        page_num: 4,
        page_summary: '第 4 页摘要',
        panels: [],
      },
      sourceUrl: '',
    })
    await flushPromises()

    expect(wrapper.text()).toContain('第 4 页摘要')

    page3.resolve({
      analysis: {
        page_num: 3,
        page_summary: '第 3 页迟到摘要',
        panels: [],
      },
      sourceUrl: '',
    })
    await flushPromises()

    expect(wrapper.text()).toContain('第 4 页摘要')
    expect(wrapper.text()).not.toContain('第 3 页迟到摘要')
  })

  it.each([
    ['not_analyzed', '○ 未分析', '此页尚未分析，点击下方按钮开始分析'],
    ['running', '… 分析中', '正在分析此页，完成后会自动更新'],
    ['failed', '! 分析失败', '此页分析失败，可点击下方按钮重试'],
    ['stale', '△ 结果已过期', '此页分析结果已过期，可重新分析以更新结果'],
    ['ready', '✓ 已分析', '此页分析完成，但没有生成页面摘要'],
  ] as const)('renders the %s backend page state explicitly', async (state, label, message) => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    getPageDataMock.mockReset().mockResolvedValueOnce({
      analysis: { analysisState: state, page_num: 3 },
      sourceUrl: '/page.png',
    })

    const wrapper = mount(PageDetail, { global: { plugins: [pinia] } })
    await flushPromises()

    const status = wrapper.get('.page-detail-panel__analysis-status')
    expect(status.attributes('data-state')).toBe(state)
    expect(status.text()).toBe(label)
    expect(wrapper.getComponent(ProductStatusBanner).text()).toContain(message)
    wrapper.unmount()
  })
})
