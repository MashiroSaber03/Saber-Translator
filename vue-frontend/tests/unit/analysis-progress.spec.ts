import { beforeEach, describe, expect, it, vi } from 'vitest'
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
  })
})
