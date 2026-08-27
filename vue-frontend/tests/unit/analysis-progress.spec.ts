import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

const {
  startAnalysisMock,
  trackJobMock,
  pauseJobMock,
  resumeJobMock,
  cancelJobMock,
  continueJobMock,
  exportAnalysisMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  startAnalysisMock: vi.fn(),
  trackJobMock: vi.fn(),
  pauseJobMock: vi.fn(),
  resumeJobMock: vi.fn(),
  cancelJobMock: vi.fn(),
  continueJobMock: vi.fn(),
  exportAnalysisMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  startAnalysis: startAnalysisMock,
  exportAnalysis: exportAnalysisMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => ({
    trackJob: trackJobMock,
    pause: pauseJobMock,
    resume: resumeJobMock,
    continueJob: continueJobMock,
    cancel: cancelJobMock,
  }),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import AnalysisProgress from '@/components/insight/AnalysisProgress.vue'

function createDeferred<T>() {
  let resolve: (value: T) => void = () => {}
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('AnalysisProgress', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setCurrentTaskId(null)
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)
    store.setChapters([])
    store.setAnalyzedPagesCount(0)
    store.updateProgress(0, 0)

    startAnalysisMock.mockReset()
    trackJobMock.mockReset()
    pauseJobMock.mockReset()
    resumeJobMock.mockReset()
    cancelJobMock.mockReset()
    continueJobMock.mockReset()
    exportAnalysisMock.mockReset()
    confirmProductActionMock.mockReset()
    startAnalysisMock.mockRejectedValue({
      status: 409,
      message: '书籍 book-1 已有运行中的任务',
    })
    pauseJobMock.mockResolvedValue(undefined)
    resumeJobMock.mockResolvedValue(undefined)
    cancelJobMock.mockResolvedValue(undefined)
    continueJobMock.mockResolvedValue(undefined)
    exportAnalysisMock.mockResolvedValue(undefined)
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

    const errorDismiss = wrapper.get('[aria-label="清除分析提示"]')
    expect(errorDismiss.element.tagName).toBe('BUTTON')
    expect(errorDismiss.attributes('aria-label')).toBe('清除分析提示')

    await errorDismiss.trigger('click')

    expect(wrapper.text()).not.toContain('书籍 book-1 已有运行中的任务')
  })

  it('maps the analysis scope and independent incremental control to backend modes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)
    store.setChapters([])

    startAnalysisMock.mockResolvedValue({
      success: true,
      jobId: 'task-full',
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
    expect(analysisModeSelect.props('modelValue')).toBe('full')

    const incrementalCheckbox = wrapper.getComponent(UiCheckbox)
    expect(incrementalCheckbox.props('label')).toBe('增量模式')
    expect(incrementalCheckbox.props('modelValue')).toBe(true)

    await wrapper.get('button[aria-label="开始分析"]').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenLastCalledWith(
      'book-1',
      expect.objectContaining({ mode: 'incremental' }),
    )
  })

  it('describes an idle book with existing results as partially analyzed', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)
    store.setAnalyzedPagesCount(3)

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.find('.analysis-progress-panel__status-label').text()).toBe('部分分析')
  })

  it('sends full mode when the incremental control is disabled', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('idle')
    store.setBookTotalPages(20)
    store.setChapters([])

    startAnalysisMock.mockResolvedValue({
      success: true,
      jobId: 'task-full',
    })

    const wrapper = mount(AnalysisProgress, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.get('input[aria-label="增量模式"]').setValue(false)
    await flushPromises()
    expect(wrapper.getComponent(UiCheckbox).props('modelValue')).toBe(false)
    expect(wrapper.text()).toContain('旧结果持续可读')

    await wrapper.get('button[aria-label="开始分析"]').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenLastCalledWith(
      'book-1',
      expect.objectContaining({ mode: 'full' }),
    )
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
    expect(modeSelect.get('button').attributes('aria-label')).toBe('选择分析范围')

    modeSelect.vm.$emit('change', 'chapter')
    await flushPromises()

    const chapterSelect = wrapper.findAllComponents(UiSelect)
      .find(select => select.get('button').attributes('aria-label') === '选择分析章节')

    expect(chapterSelect).toBeTruthy()
  })

  it('allows retry when status is failed', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setAnalysisStatus('failed')
    store.setBookTotalPages(20)
    store.setChapters([])

    startAnalysisMock.mockResolvedValue({
      success: true,
      jobId: 'task-retry',
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

  it('tracks a new analysis without writing an optimistic task state', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.setAnalysisStatus('completed')
    store.updateProgress(20, 20)
    startAnalysisMock.mockResolvedValue({
      jobId: 'task-retry',
      runId: 'run-retry',
    })

    const wrapper = mount(AnalysisProgress, {
      global: { plugins: [pinia] },
    })

    await wrapper.get('button[aria-label="重新分析"]').trigger('click')
    await flushPromises()

    expect(trackJobMock).toHaveBeenCalledWith('task-retry')
    expect(store.currentTaskId).toBeNull()
    expect(store.analysisStatus).toBe('completed')
    expect(store.progress.current).toBe(20)
    expect(store.progress.total).toBe(20)
  })

  it('uses product confirmation before analysis cancellation', async () => {
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
    expect(cancelJobMock).toHaveBeenCalledWith('task-1')
    expect(store.analysisStatus).toBe('running')
    expect(store.currentTaskId).toBe('task-1')
    expect(wrapper.text()).toContain('分析中')
  })

  it('uses the distinct continue command for an interrupted job', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.setCurrentTaskId('task-interrupted')
    store.setAnalysisStatus('interrupted')

    const wrapper = mount(AnalysisProgress, {
      global: { plugins: [pinia] },
    })

    await wrapper.get('button[aria-label="继续中断任务"]').trigger('click')
    await flushPromises()

    expect(continueJobMock).toHaveBeenCalledWith('task-interrupted')
    expect(store.analysisStatus).toBe('interrupted')
  })

  it('does not submit the same control command twice while it is pending', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.setCurrentTaskId('task-1')
    store.setAnalysisStatus('running')
    const pauseRequest = createDeferred<void>()
    pauseJobMock.mockReturnValueOnce(pauseRequest.promise)

    const wrapper = mount(AnalysisProgress, {
      global: { plugins: [pinia] },
    })
    const pauseButton = wrapper.get('button[aria-label="暂停分析"]')
    await pauseButton.trigger('click')
    await pauseButton.trigger('click')

    expect(pauseJobMock).toHaveBeenCalledTimes(1)
    expect(pauseButton.attributes('disabled')).toBeDefined()

    pauseRequest.resolve()
    await flushPromises()
    expect(store.analysisStatus).toBe('running')
  })

  it('does not write an accepted old-book job into the newly selected book', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.setBookTotalPages(20)
    const submission = createDeferred<{ jobId: string; success: boolean }>()
    startAnalysisMock.mockReturnValueOnce(submission.promise)

    const wrapper = mount(AnalysisProgress, {
      global: { plugins: [pinia] },
    })
    await wrapper.get('button[aria-label="开始分析"]').trigger('click')
    store.setCurrentBook('book-2')
    submission.resolve({ success: true, jobId: 'book-1-job' })
    await flushPromises()

    expect(store.currentBookId).toBe('book-2')
    expect(store.currentTaskId).toBeNull()
    expect(store.analysisStatus).toBe('idle')
  })

  it('shows successful export acceptance as information rather than an error', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.setCurrentBook('book-1')
    store.setAnalyzedPagesCount(3)

    const wrapper = mount(AnalysisProgress, {
      global: { plugins: [pinia] },
    })
    await wrapper.get('button[aria-label="导出分析报告"]').trigger('click')
    await flushPromises()

    expect(exportAnalysisMock).toHaveBeenCalledWith('book-1')
    expect(wrapper.text()).toContain('导出任务已进入任务中心')
    expect(wrapper.getComponent(ProductStatusBanner).props('tone')).toBe('info')
  })

  it('only shows the incremental option for the full-book scope', async () => {
    const wrapper = mount(AnalysisProgress)
    expect(wrapper.findComponent(UiCheckbox).exists()).toBe(true)

    wrapper.getComponent(UiSelect).vm.$emit('change', 'chapter')
    await flushPromises()

    expect(wrapper.findComponent(UiCheckbox).exists()).toBe(false)
  })

  it('uses product action rows and shared button variants for analysis controls', () => {
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
})
