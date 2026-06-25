import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, nextTick } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { useTimelinePanel } from '@/components/insight/timeline/useTimelinePanel'

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

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

describe('TimelinePanel', () => {
  enableAutoUnmount(afterEach)

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
    const initialTimeline = {
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
    }
    const regeneratedTimeline = {
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
    }
    getTimelineMock.mockResolvedValueOnce(initialTimeline).mockResolvedValue(regeneratedTimeline)
    regenerateTimelineMock.mockResolvedValue(regeneratedTimeline)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    const refreshKeyBefore = store.dataRefreshKey

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
    expect(store.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('ignores stale timeline responses after switching books', async () => {
    const firstTimeline = deferred<Record<string, unknown>>()
    const secondTimeline = deferred<Record<string, unknown>>()
    getTimelineMock
      .mockReturnValueOnce(firstTimeline.promise)
      .mockReturnValueOnce(secondTimeline.promise)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(TimelinePanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(getTimelineMock).toHaveBeenCalledWith('book-1')

    store.currentBookId = 'book-2'
    await nextTick()
    expect(getTimelineMock).toHaveBeenCalledWith('book-2')

    secondTimeline.resolve({
      success: true,
      mode: 'enhanced',
      story_arcs: [{ id: 'book-2-arc', name: '当前书时间线', page_range: { start: 2, end: 4 } }],
      stats: { total_events: 1, total_pages: 4 },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('当前书时间线')

    firstTimeline.resolve({
      success: true,
      mode: 'enhanced',
      story_arcs: [{ id: 'book-1-arc', name: '旧书时间线', page_range: { start: 1, end: 3 } }],
      stats: { total_events: 1, total_pages: 3 },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('当前书时间线')
    expect(wrapper.text()).not.toContain('旧书时间线')
  })

  it('ignores timeline responses after the owner unmounts', async () => {
    const pendingTimeline = deferred<Record<string, unknown>>()
    getTimelineMock.mockReturnValueOnce(pendingTimeline.promise)

    const Harness = defineComponent({
      setup() {
        const timeline = useTimelinePanel()
        return { timelineData: timeline.timelineData }
      },
      template: '<div />',
    })

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(Harness, {
      global: {
        plugins: [pinia],
      },
    })
    expect(getTimelineMock).toHaveBeenCalledWith('book-1')

    wrapper.unmount()
    pendingTimeline.resolve({
      success: true,
      mode: 'enhanced',
      story_arcs: [{ id: 'late-arc', name: '卸载后的时间线', page_range: { start: 1, end: 2 } }],
      stats: { total_events: 1, total_pages: 2 },
    })
    await flushPromises()

    expect(wrapper.vm.timelineData).toBeNull()
  })
})
