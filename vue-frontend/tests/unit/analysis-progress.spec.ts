import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const { startAnalysisMock } = vi.hoisted(() => ({
  startAnalysisMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  startAnalysis: startAnalysisMock,
  pauseAnalysis: vi.fn(),
  resumeAnalysis: vi.fn(),
  cancelAnalysis: vi.fn(),
  exportAnalysis: vi.fn(),
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
    startAnalysisMock.mockRejectedValue({
      status: 409,
      message: '书籍 book-1 已有运行中的任务',
    })
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
        stubs: {
          CustomSelect: {
            template: '<div class="custom-select-stub"></div>',
            props: ['modelValue', 'options'],
            emits: ['update:modelValue', 'change'],
          },
        },
      },
    })

    await wrapper.find('.btn-analysis-start').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenCalled()
    expect(wrapper.text()).toContain('书籍 book-1 已有运行中的任务')
    expect(wrapper.emitted('start-polling')).toBeUndefined()
    expect(store.analysisStatus).toBe('idle')

    const errorDismiss = wrapper.find('.error-message')
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
        stubs: {
          CustomSelect: {
            template: '<div class="custom-select-stub"></div>',
            props: ['modelValue', 'options'],
            emits: ['update:modelValue', 'change'],
          },
        },
      },
    })

    expect(wrapper.text()).toContain('全量重跑整本书（会清理旧结果）')

    await wrapper.find('.btn-analysis-start').trigger('click')
    await flushPromises()

    expect(startAnalysisMock).toHaveBeenCalledWith('book-1', expect.objectContaining({ mode: 'full' }))
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
        stubs: {
          CustomSelect: {
            template: '<div class="custom-select-stub"></div>',
            props: ['modelValue', 'options'],
            emits: ['update:modelValue', 'change'],
          },
        },
      },
    })

    expect(wrapper.find('.btn-analysis-start').exists()).toBe(true)
    expect(wrapper.text()).toContain('重新分析')

    await wrapper.find('.btn-analysis-start').trigger('click')
    await flushPromises()
    expect(startAnalysisMock).toHaveBeenCalled()
  })
})
