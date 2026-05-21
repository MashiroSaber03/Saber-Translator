import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

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
        stubs: {
          CustomSelect: {
            template: '<button class="custom-select-stub" @click="$emit(\'update:modelValue\', \'story_summary\'); $emit(\'change\')">story_summary</button>',
            props: ['modelValue', 'options'],
            emits: ['update:modelValue', 'change'],
          },
        },
      },
    })
    await flushPromises()

    await wrapper.find('.custom-select-stub').trigger('click')
    await flushPromises()

    await wrapper.findAll('.btn-icon')[1]!.trigger('click')
    await flushPromises()

    expect(regenerateOverviewMock).toHaveBeenCalledWith('book-1', 'story_summary', true)
    expect(store.dataRefreshKey).not.toBe(refreshKeyBefore)
  })
})
