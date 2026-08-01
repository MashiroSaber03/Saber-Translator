import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

const { startAnalysisMock, cancelAnalysisMock, confirmProductActionMock } = vi.hoisted(() => ({
  startAnalysisMock: vi.fn(),
  cancelAnalysisMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  startAnalysis: startAnalysisMock,
  pauseAnalysis: vi.fn(),
  resumeAnalysis: vi.fn(),
  cancelAnalysis: cancelAnalysisMock,
  exportAnalysis: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import AnalysisProgress from '@/components/insight/AnalysisProgress.vue'

describe('AnalysisProgress', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setCurrentTaskId(null)
    store.setAnalysisStatus('idle')
    store.setIncrementalAnalysis(true)
    store.setBookTotalPages(20)
    store.setChapters([])
    store.setAnalyzedPagesCount(0)
    store.updateProgress(0, 0)

    startAnalysisMock.mockReset()
    cancelAnalysisMock.mockReset()
    confirmProductActionMock.mockReset()
    startAnalysisMock.mockRejectedValue({
      status: 409,
      message: '书籍 book-1 已有运行中的任务',
    })
    cancelAnalysisMock.mockResolvedValue(undefined)
    confirmProductActionMock.mockResolvedValue(true)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows readable error on 409 and does not enter running state', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setIncrementalAnalysis(true)
    store.setBookTotalPages(20)
    store.setChapters([])

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.get('button[aria-label="开始分析"]').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenCalled()
    expect(wrapper.text()).toContain('书籍 book-1 已有运行中的任务')
    expect(wrapper.emitted('start-polling')).toBeUndefined()
    expect(store.analysisStatus).toBe('idle')

    const errorBanner = wrapper.getComponent(ProductStatusBanner)
    expect(errorBanner.props('tone')).toBe('danger')

    const errorDismiss = wrapper.get('[aria-label="清除分析错误"]')
    expect(errorDismiss.element.tagName).toBe('BUTTON')
    expect(errorDismiss.attributes('aria-label')).toBe('清除分析错误')

    await errorDismiss.trigger('click')

    expect(wrapper.text()).not.toContain('书籍 book-1 已有运行中的任务')
  })

  it('shows full rerun description and sends full mode when incremental is off', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setIncrementalAnalysis(false)
    store.setBookTotalPages(20)
    store.setChapters([])

    startAnalysisMock.mockResolvedValue({
      success: true,
      task_id: 'task-full',
    })

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    const analysisModeSelect = wrapper.getComponent(UiSelect)
    expect(analysisModeSelect.props('options')).toEqual([
      expect.objectContaining({ value: 'full' }),
      expect.objectContaining({ value: 'chapter' }),
      expect.objectContaining({ value: 'page' }),
    ])
    expect(wrapper.text()).toContain('全量重跑整本书（会清理旧结果）')

    await wrapper.get('button[aria-label="开始分析"]').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenCalledWith('book-1', expect.objectContaining({ mode: 'full' }))
  })

  it('names analysis mode selectors when they are not wrapped in visible fields', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)
    store.setChapters([{ id: 'chapter-1', title: '第一章', startPage: 1, endPage: 8 }])

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    const modeSelect = wrapper.getComponent(UiSelect)
    expect(modeSelect.attributes('aria-label')).toBe('选择分析范围')

    modeSelect.vm.$emit('change', 'chapter')
    await flushPromises()

    const chapterSelect = wrapper.findAllComponents(UiSelect)
      .find(select => select.attributes('aria-label') === '选择分析章节')

    expect(chapterSelect).toBeTruthy()
  })

  it('allows retry when status is failed', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('failed')
    store.setIncrementalAnalysis(false)
    store.setBookTotalPages(20)
    store.setChapters([])

    startAnalysisMock.mockResolvedValue({
      success: true,
      task_id: 'task-retry',
    })

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.find('button[aria-label="重新分析"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('重新分析')

    await wrapper.get('button[aria-label="重新分析"]').trigger('click')
    await flushPromises()
    expect(startAnalysisMock).toHaveBeenCalled()
  })

  it('uses product confirmation before cancelling analysis', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setCurrentTaskId('task-1')
    store.setAnalysisStatus('running')
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.get('button[aria-label="取消分析"]').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '取消分析',
      message: '确定要取消分析吗？',
      confirmText: '取消分析',
      cancelText: '继续分析',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(cancelAnalysisMock).toHaveBeenCalledWith('task-1')
    expect(store.analysisStatus).toBe('idle')
    expect(wrapper.find('.progress-bar-slim').exists()).toBe(false)
  })

  it('keeps analysis owner colors on semantic tokens instead of raw values', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/AnalysisProgress.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--shadow-action-brand')
    expect(source).toContain('class="analysis-progress-panel"')
    expect(source).toContain('analysis-progress-panel__status-dot')
    expect(source).not.toContain('sidebar-section analysis-control-compact')
    expect(styleBlock).not.toContain('.sidebar-section.analysis-control-compact')
    expect(styleBlock).not.toMatch(/\.(?:status-dot|status-left|status-progress|progress-message)\b/)
  })

  it('uses product action rows and shared button variants for analysis controls', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/AnalysisProgress.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(source).toContain("import ProductActionRow from '@/components/product/ProductActionRow.vue'")
    expect(source).toContain('aria-label="分析启动操作"')
    expect(source).toContain('aria-label="运行中的分析操作"')
    expect(source).toContain('aria-label="分析附加操作"')
    expect(styleBlock).not.toMatch(/btn-analysis-start|btn-control|btn-pause|btn-resume|btn-cancel|button-icon-sm/)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findComponent(ProductActionRow).exists()).toBe(true)
    const exportAction = wrapper.getComponent(UiIconButton)
    expect(exportAction.props('label')).toBe('导出分析报告')
    expect(exportAction.props('title')).toBe('导出分析报告')
  })

  it('keeps analysis progress structural hooks under the panel owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/AnalysisProgress.vue'), 'utf8')
    const oldHooks = [
      'analysis-progress__progress',
      'analysis-progress__error',
      'analysis-progress__page-number',
      'analysis-progress__incremental-checkbox',
      'analysis-action-row',
      'analysis-mode-select',
      'analysis-start-action',
      'analysis-action-button',
      'page-input-wrapper',
      'page-hint',
      'mode-description',
      'estimated-time',
      'analysis-options-row',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('analysis-progress-panel__progress')
    expect(source).toContain('analysis-progress-panel__action-row')
    expect(source).toContain('analysis-progress-panel__mode-description')
  })
})
