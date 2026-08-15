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
import type { TimelineData } from '@/types/insight'

const { getTimelineMock, regenerateTimelineMock } = vi.hoisted(() => ({
  getTimelineMock: vi.fn(),
  regenerateTimelineMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getTimeline: getTimelineMock,
  regenerateTimeline: regenerateTimelineMock,
}))

import TimelinePanel from '@/components/insight/TimelinePanel.vue'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(nextResolve => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

function timelineData(overrides: Partial<TimelineData> = {}): TimelineData {
  return {
    timeline_version_id: 'timeline-default',
    mode: 'simple',
    groups: [
      {
        id: 'event-default',
        page_range: { start: 1, end: 1 },
        events: ['默认事件'],
        summary: '默认事件',
        thumbnail_page: 1,
      },
    ],
    events: [
      {
        eventId: 'event-default',
        summary: '默认事件',
        page_ids: ['page-default'],
        page_numbers: [1],
      },
    ],
    stats: { total_events: 1, total_pages: 1, total_characters: 0 },
    story_summary: '',
    main_characters: [],
    page_thumbnails: { 1: '/api/v2/assets/thumb-default' },
    next_event_cursor: null,
    next_character_cursor: null,
    ...overrides,
  }
}

describe('TimelinePanel', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.dataRefreshKey = 0

    getTimelineMock.mockReset()
    regenerateTimelineMock.mockReset()
  })

  it('loads the current timeline and refreshes queued regeneration after backend completion', async () => {
    const initialTimeline = timelineData({
      timeline_version_id: 'timeline-initial',
      mode: 'enhanced',
      plot_arcs: [
        {
          id: 'arc-1',
          name: '开端',
          description: '开端描述',
          page_range: { start: 1, end: 3 },
        },
      ],
      main_characters: [],
      plot_threads: [],
      story_summary: '初始概要',
      stats: {
        total_events: 1,
        total_pages: 3,
        total_arcs: 1,
        total_characters: 0,
        total_threads: 0,
      },
    })
    const regeneratedTimeline = timelineData({
      timeline_version_id: 'timeline-regenerated',
      mode: 'enhanced',
      plot_arcs: [
        {
          id: 'arc-2',
          name: '高潮',
          description: '高潮描述',
          page_range: { start: 4, end: 6 },
        },
      ],
      main_characters: [],
      plot_threads: [],
      story_summary: '重生概要',
      stats: {
        total_events: 2,
        total_pages: 6,
        total_arcs: 1,
        total_characters: 0,
        total_threads: 0,
      },
    })
    getTimelineMock.mockResolvedValueOnce(initialTimeline).mockResolvedValue(regeneratedTimeline)
    regenerateTimelineMock.mockResolvedValue('timeline-job-1')

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

    expect(wrapper.text()).toContain('时间线生成已进入任务中心，完成后将自动加载。')
    expect(wrapper.text()).toContain('开端')
    expect(wrapper.text()).not.toContain('高潮')
    expect(store.dataRefreshKey).toBe(refreshKeyBefore)

    store.triggerDataRefresh()
    await flushPromises()

    expect(wrapper.text()).toContain('高潮')
    expect(wrapper.text()).toContain('重生概要')
    expect(store.dataRefreshKey).not.toBe(refreshKeyBefore)
  })

  it('ignores stale timeline responses after switching books', async () => {
    const firstTimeline = deferred<TimelineData | null>()
    const secondTimeline = deferred<TimelineData | null>()
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

    secondTimeline.resolve(
      timelineData({
        timeline_version_id: 'book-2-timeline',
        mode: 'enhanced',
        plot_arcs: [
          {
            id: 'book-2-arc',
            name: '当前书时间线',
            description: '当前书描述',
            page_range: { start: 2, end: 4 },
          },
        ],
        stats: { total_events: 1, total_pages: 4 },
      })
    )
    await flushPromises()
    expect(wrapper.text()).toContain('当前书时间线')

    firstTimeline.resolve(
      timelineData({
        timeline_version_id: 'book-1-timeline',
        mode: 'enhanced',
        plot_arcs: [
          {
            id: 'book-1-arc',
            name: '旧书时间线',
            description: '旧书描述',
            page_range: { start: 1, end: 3 },
          },
        ],
        stats: { total_events: 1, total_pages: 3 },
      })
    )
    await flushPromises()

    expect(wrapper.text()).toContain('当前书时间线')
    expect(wrapper.text()).not.toContain('旧书时间线')
  })

  it('ignores timeline responses after the owner unmounts', async () => {
    const pendingTimeline = deferred<TimelineData | null>()
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
    pendingTimeline.resolve(
      timelineData({
        timeline_version_id: 'late-timeline',
        mode: 'enhanced',
        plot_arcs: [
          {
            id: 'late-arc',
            name: '卸载后的时间线',
            description: '迟到描述',
            page_range: { start: 1, end: 2 },
          },
        ],
        stats: { total_events: 1, total_pages: 2 },
      })
    )
    await flushPromises()

    expect(wrapper.vm.timelineData).toBeNull()
  })

  it('prevents duplicate regeneration commands and releases the old-book lock', async () => {
    getTimelineMock.mockResolvedValue(null)
    const regeneration = deferred<string>()
    regenerateTimelineMock.mockReturnValue(regeneration.promise)

    const Harness = defineComponent({
      setup() {
        return useTimelinePanel()
      },
      template: '<div />',
    })
    const wrapper = mount(Harness)
    await flushPromises()

    void wrapper.vm.regenerateTimeline()
    void wrapper.vm.regenerateTimeline()
    await nextTick()
    expect(regenerateTimelineMock).toHaveBeenCalledTimes(1)
    expect(wrapper.vm.isRegenerating).toBe(true)

    const store = useInsightStore()
    store.currentBookId = 'book-2'
    await nextTick()
    expect(wrapper.vm.isRegenerating).toBe(false)

    regeneration.resolve('timeline-job-1')
    await flushPromises()
    expect(wrapper.vm.pendingMessage).toBe('')
  })

  it('releases an old load-more request when switching books', async () => {
    const loadMore = deferred<TimelineData | null>()
    getTimelineMock
      .mockResolvedValueOnce(
        timelineData({
          timeline_version_id: 'timeline-page-1',
          groups: [
            {
              id: 'group-1',
              page_range: { start: 1, end: 2 },
              events: ['event'],
              thumbnail_page: 1,
            },
          ],
          next_event_cursor: 1,
          stats: { total_events: 2, total_pages: 2 },
        })
      )
      .mockReturnValueOnce(loadMore.promise)
      .mockResolvedValueOnce(null)

    const Harness = defineComponent({
      setup() {
        return useTimelinePanel()
      },
      template: '<div />',
    })
    const wrapper = mount(Harness)
    await flushPromises()

    void wrapper.vm.loadMoreTimeline()
    await nextTick()
    expect(wrapper.vm.isLoadingMore).toBe(true)

    const store = useInsightStore()
    store.currentBookId = 'book-2'
    await nextTick()
    expect(wrapper.vm.isLoadingMore).toBe(false)

    loadMore.resolve(
      timelineData({
        timeline_version_id: 'timeline-page-1',
        groups: [
          {
            id: 'stale-group',
            page_range: { start: 3, end: 4 },
            events: ['stale'],
            thumbnail_page: 3,
          },
        ],
        stats: { total_events: 2, total_pages: 4 },
      })
    )
    await flushPromises()
    expect(wrapper.vm.timelineData).toBeNull()
  })

  it('renders loading feedback through the shared spinner primitive', async () => {
    const pendingTimeline = deferred<TimelineData | null>()
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

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'),
      'utf8'
    )
    expect(source).toContain('timeline-panel__loading-indicator')
    expect(source).not.toContain('timeline-loading-spinner')
  })

  it('renders load errors through the product status banner', async () => {
    getTimelineMock.mockRejectedValueOnce(new Error('时间线服务不可用')).mockResolvedValueOnce(null)

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
    expect(wrapper.findComponent(ProductEmptyState).exists()).toBe(false)

    await wrapper.get('button').trigger('click')
    await flushPromises()

    expect(getTimelineMock).toHaveBeenCalledTimes(2)
    expect(wrapper.findComponent(ProductEmptyState).exists()).toBe(true)
  })

  it('renders the no-data state through the product empty-state pattern', async () => {
    getTimelineMock.mockResolvedValueOnce(null)

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
    expect(emptyState.get('.product-empty-state__icon-text').text()).toBe('📈')
    expect(emptyState.props('title')).toBe('时间线尚未生成')
    expect(emptyState.props('description')).toBe(
      '完成漫画分析后会自动生成时间线，或点击下方按钮手动生成'
    )
    expect(wrapper.find('.timeline-empty-state').exists()).toBe(false)
    expect(wrapper.find('.empty-icon').exists()).toBe(false)

    const generateButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('生成时间线'))
    expect(generateButton).toBeTruthy()
    await generateButton!.trigger('click')
    expect(regenerateTimelineMock).toHaveBeenCalledWith('book-1')
  })

  it('shows durable queued feedback until a generated timeline can be loaded', async () => {
    getTimelineMock.mockResolvedValueOnce(null)
    regenerateTimelineMock.mockResolvedValueOnce('timeline-job-1')

    const wrapper = mount(TimelinePanel)
    await flushPromises()

    const generateButton = wrapper
      .findAll('button')
      .find(button => button.text().includes('生成时间线'))
    await generateButton!.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('时间线生成中')
    expect(wrapper.text()).toContain('时间线生成已进入任务中心，完成后将自动加载。')
    expect(wrapper.text()).not.toContain('时间线尚未生成')
  })

  it('maps timeline owner shadows through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'),
      'utf8'
    )
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--shadow-medium')
    expect(styleBlock).toContain('--shadow-soft')
  })

  it('does not redefine the shared button primitive skin in the timeline panel owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'),
      'utf8'
    )

    expect(source).not.toContain('--ui-button-')
  })

  it('uses timeline-panel owner hooks for parent layout styling', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/TimelinePanel.vue'),
      'utf8'
    )
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
