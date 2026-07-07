import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, nextTick } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import { useTimelinePanel } from '@/components/insight/timeline/useTimelinePanel'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

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

    await wrapper.find('.timeline-header__regenerate-action').trigger('click')
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

  it('renders loading feedback through the shared spinner primitive', async () => {
    const pendingTimeline = deferred<Record<string, unknown>>()
    getTimelineMock.mockReturnValueOnce(pendingTimeline.promise)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(TimelinePanel, {
      global: {
        plugins: [pinia],
      },
    })
    await nextTick()

    const spinner = wrapper.getComponent(UiSpinner)
    expect(spinner.props('label')).toBe('加载时间线')
    expect(spinner.props('decorative')).toBe(false)
    expect(wrapper.text()).toContain('加载时间线...')

    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'), 'utf8')
    expect(source).toContain('timeline-panel__loading-indicator')
    expect(source).not.toContain('timeline-loading-spinner')
  })

  it('renders load errors through the product status banner', async () => {
    getTimelineMock.mockRejectedValueOnce(new Error('时间线服务不可用'))

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

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('danger')
    expect(banner.props('ariaLive')).toBe('assertive')
    expect(wrapper.text()).toContain('时间线服务不可用')
  })

  it('renders the no-data state through the product empty-state pattern', async () => {
    getTimelineMock.mockResolvedValueOnce({
      success: true,
      mode: 'enhanced',
      story_arcs: [],
      characters: [],
      plot_threads: [],
      stats: { total_events: 0, total_pages: 0 },
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

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props('iconName')).toBe('bar-chart')
    expect(emptyState.props('title')).toBe('时间线尚未生成')
    expect(emptyState.props('description')).toBe('完成漫画分析后会自动生成时间线，或点击下方按钮手动生成')
    expect(wrapper.find('.timeline-empty-state').exists()).toBe(false)
    expect(wrapper.find('.empty-icon').exists()).toBe(false)

    const generateButton = wrapper.findAll('button').find(button => button.text().includes('生成时间线'))
    expect(generateButton).toBeTruthy()
    await generateButton!.trigger('click')
    expect(regenerateTimelineMock).toHaveBeenCalledWith('book-1')
  })

  it('maps timeline owner shadows through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--shadow-medium')
    expect(styleBlock).toContain('--shadow-soft')
  })

  it('does not redefine the shared button primitive skin in the timeline panel owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'), 'utf8')

    expect(source).not.toContain('--ui-button-')
  })

  it('uses timeline-panel owner hooks for parent layout styling', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const oldHooks = [
      'timeline-tab',
      'timeline-container',
      'timeline-status-banner',
      'loading-state',
      'timeline-loading-indicator',
      'timeline-empty',
      'timeline-section',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('class="timeline-panel"')
    expect(source).toContain('timeline-panel__loading-indicator')
    expect(source).toContain('timeline-panel__section-title')
    expect(styleBlock).not.toMatch(/\.timeline-section\s+h4/)
  })
})
