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

    await wrapper.findAll('.button-icon')[1]!.trigger('click')
    await flushPromises()

    expect(regenerateOverviewMock).toHaveBeenCalledWith('book-1', 'story_summary', true)
    expect(store.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('reloads generated overview data without routine console output when refresh key changes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    mount(OverviewPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          CustomSelect: {
            template: '<button class="custom-select-stub">story_summary</button>',
            props: ['modelValue', 'options'],
          },
        },
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
        stubs: {
          CustomSelect: {
            template: '<button class="custom-select-stub">no_spoiler</button>',
            props: ['modelValue', 'options'],
          },
        },
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
        stubs: {
          CustomSelect: {
            template: '<button class="custom-select-stub">story_summary</button>',
            props: ['modelValue', 'options'],
          },
        },
      },
    })
    await flushPromises()

    const recentItem = wrapper.find('.recent-page-item')
    expect(recentItem.element.tagName).toBe('BUTTON')
    expect(recentItem.attributes('type')).toBe('button')
    expect(recentItem.attributes('aria-label')).toBe('查看第 2 页分析详情')

    await recentItem.trigger('click')

    expect(store.selectedPageNum).toBe(2)
  })
})
