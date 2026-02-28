import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const { getTimelineMock, regenerateTimelineMock, getThumbnailUrlMock } = vi.hoisted(() => ({
  getTimelineMock: vi.fn(),
  regenerateTimelineMock: vi.fn(),
  getThumbnailUrlMock: vi.fn(() => '/thumb.jpg'),
}))

vi.mock('@/api/insight', () => ({
  getTimeline: getTimelineMock,
  regenerateTimeline: regenerateTimelineMock,
  getThumbnailUrl: getThumbnailUrlMock,
}))

import TimelinePanel from '@/components/insight/TimelinePanel.vue'

describe('TimelinePanel', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.dataRefreshKey = 0

    getThumbnailUrlMock.mockClear()
    getTimelineMock.mockReset()
    regenerateTimelineMock.mockReset()
  })

  it('normalizes story_arcs in load and regenerate flows', async () => {
    getTimelineMock.mockResolvedValue({
      success: true,
      mode: 'enhanced',
      story_arcs: [
        {
          id: 'arc-1',
          name: '开端',
          description: '开端描述',
          page_range: { start: 1, end: 3 },
        },
      ],
      characters: [],
      plot_threads: [],
      summary: { one_sentence: '初始概要' },
      stats: { total_events: 1, total_pages: 3, total_arcs: 1, total_characters: 0, total_threads: 0 },
    })
    regenerateTimelineMock.mockResolvedValue({
      success: true,
      mode: 'enhanced',
      story_arcs: [
        {
          id: 'arc-2',
          name: '高潮',
          description: '高潮描述',
          page_range: { start: 4, end: 6 },
        },
      ],
      characters: [],
      plot_threads: [],
      summary: { one_sentence: '重生概要' },
      stats: { total_events: 2, total_pages: 6, total_arcs: 1, total_characters: 0, total_threads: 0 },
    })

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(TimelinePanel, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('开端')
    expect(wrapper.text()).toContain('初始概要')

    await wrapper.find('.timeline-header button').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('高潮')
    expect(wrapper.text()).toContain('重生概要')
  })
})
