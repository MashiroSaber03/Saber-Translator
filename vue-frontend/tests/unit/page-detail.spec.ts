import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { useInsightStore } from '@/stores/insightStore'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

const { getPageDataMock, reanalyzePageMock, getPageImageUrlMock } = vi.hoisted(() => ({
  getPageDataMock: vi.fn(),
  reanalyzePageMock: vi.fn(),
  getPageImageUrlMock: vi.fn(() => '/page.png'),
}))

vi.mock('@/api/insight', () => ({
  getPageData: getPageDataMock,
  reanalyzePage: reanalyzePageMock,
  getPageImageUrl: getPageImageUrlMock,
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
      success: true,
      analysis: {
        page_num: 3,
        page_summary: '旧摘要',
        panels: [],
      },
    })

    reanalyzePageMock.mockReset()
    reanalyzePageMock.mockResolvedValue({
      success: true,
      task_id: 'task-123',
    })

    getPageImageUrlMock.mockReset()
    getPageImageUrlMock.mockReturnValue('/page.png')
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

    const setStatusSpy = vi.spyOn(store, 'setAnalysisStatus')
    const setTaskSpy = vi.spyOn(store, 'setCurrentTaskId')

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
    expect(setTaskSpy).toHaveBeenCalledWith('task-123')
    expect(setStatusSpy).toHaveBeenCalledWith('running')

    // 不应在启动后立即当作同步完成并刷新详情
    expect(getPageDataMock).toHaveBeenCalledTimes(1)

    store.triggerDataRefresh()
    await nextTick()
    await flushPromises()

    // 分析完成信号到达后自动刷新当前页详情
    expect(getPageDataMock).toHaveBeenCalledTimes(2)
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

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    const reanalyzeButton = wrapper.findAll('button').find(button => button.text().includes('重新分析'))
    expect(reanalyzeButton).toBeTruthy()

    await reanalyzeButton!.trigger('click')
    await flushPromises()

    const runningButton = wrapper.findAll('button').find(button => button.text().includes('分析中...'))
    expect(runningButton).toBeTruthy()
    const spinner = runningButton!.getComponent(UiSpinner)
    expect(spinner.props('decorative')).toBe(true)
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

  it('does not assert shared button primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/page-detail.spec.ts'), 'utf8')
    const buttonClassPrefix = 'ui-' + 'button--'

    expect(source).not.toContain(buttonClassPrefix)
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
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="section-title"')
    expect(source).not.toContain('.section-title')
  })

  it('uses product status feedback when no page is selected', async () => {
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

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props()).toMatchObject({
      iconName: 'file-text',
      title: '选择页面查看详情',
      tone: 'neutral',
    })
    expect(getPageDataMock).not.toHaveBeenCalled()
  })

  it('uses product status feedback for page detail load errors', async () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/PageDetail.vue'), 'utf8')
    expect(source).not.toContain('class="error-message"')
    expect(source).not.toContain('.error-message')

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValueOnce({
      success: false,
      error: '加载页面失败',
    })

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
      success: true,
      analysis: {
        analyzed: false,
        page_num: 3,
        page_summary: '',
        panels: [],
      },
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
      title: '此页尚未分析',
      tone: 'neutral',
    })

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValueOnce({
      success: true,
      analysis: {
        analyzed: true,
        page_num: 3,
        page_summary: '已有摘要',
        panels: [],
      },
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
      iconName: 'message',
      role: 'note',
      title: '此页没有检测到对话内容',
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
      success: true
      analysis: { page_num: number; page_summary: string; panels: never[] }
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

      await previewTrigger.trigger('click')
      await nextTick()

      const previewModal = wrapper.get('.page-detail-panel__image-preview-modal')
      expect(previewModal.exists()).toBe(true)
      expect(document.activeElement).toBe(previewModal.element)

      const closeButton = wrapper.get('button[aria-label="关闭图片预览"]')
      expect(closeButton.getComponent(UiIcon).props('name')).toBe('x')
      expect(closeButton.text()).not.toContain('×')

      expect(wrapper.get('.page-detail-panel__preview-nav button[title="上一页 (←)"]').attributes('aria-label')).toBe('预览上一页')
      expect(wrapper.get('.page-detail-panel__preview-nav button[title="下一页 (→)"]').attributes('aria-label')).toBe('预览下一页')
      const iconActions = wrapper.findAllComponents(UiIconButton)
      expect(iconActions.some(action => action.props('label') === '关闭图片预览')).toBe(true)
      expect(iconActions.some(action => action.props('label') === '预览上一页')).toBe(true)
      expect(iconActions.some(action => action.props('label') === '预览下一页')).toBe(true)
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

  it('ignores stale page detail responses after selecting another page', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)

    const page3 = deferred<{
      success: true
      analysis: { page_num: number; page_summary: string; panels: never[] }
    }>()
    const page4 = deferred<{
      success: true
      analysis: { page_num: number; page_summary: string; panels: never[] }
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
      success: true,
      analysis: {
        page_num: 4,
        page_summary: '第 4 页摘要',
        panels: [],
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('第 4 页摘要')

    page3.resolve({
      success: true,
      analysis: {
        page_num: 3,
        page_summary: '第 3 页迟到摘要',
        panels: [],
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('第 4 页摘要')
    expect(wrapper.text()).not.toContain('第 3 页迟到摘要')
  })
})
