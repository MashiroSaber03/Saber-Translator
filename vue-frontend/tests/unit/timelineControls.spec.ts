import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import PlotThreadsList from '@/components/insight/timeline/PlotThreadsList.vue'
import TimelineArcCard from '@/components/insight/timeline/TimelineArcCard.vue'
import TimelineCharacterGrid from '@/components/insight/timeline/TimelineCharacterGrid.vue'
import TimelineGroupCard from '@/components/insight/timeline/TimelineGroupCard.vue'
import TimelineHeader from '@/components/insight/timeline/TimelineHeader.vue'
import TimelineStats from '@/components/insight/timeline/TimelineStats.vue'
import TimelineSummaryCard from '@/components/insight/timeline/TimelineSummaryCard.vue'
import TimelineTrack from '@/components/insight/timeline/TimelineTrack.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

const arc = {
  id: 'arc-1',
  name: '开端',
  description: '主角踏上旅程。',
  page_range: { start: 2, end: 5 },
}

const group = {
  id: 'group-1',
  page_range: { start: 8, end: 10 },
  thumbnail_page: 8,
  summary: '冲突升级。',
  events: ['遭遇敌人'],
}

describe('Timeline child controls', () => {
  it('separates arc thumbnail navigation from arc expansion', async () => {
    const wrapper = mount(TimelineArcCard, {
      props: {
        arc,
        arcId: 'arc-1',
        expanded: false,
        thumbnailUrl: '/thumb-2.jpg',
      },
    })

    const thumbnailButton = wrapper.find('.timeline-event-card-shell__thumbnail-action')
    expect(thumbnailButton.element.tagName).toBe('BUTTON')
    expect(thumbnailButton.attributes('aria-label')).toBe('查看第 2 页')

    await thumbnailButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(2)

    const toggleButton = wrapper.find('.timeline-event-card-shell__toggle')
    expect(toggleButton.element.tagName).toBe('BUTTON')
    expect(toggleButton.attributes('aria-expanded')).toBe('false')

    await toggleButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('arc-1')
  })

  it('renders an arc thumbnail fallback from component state after image errors', async () => {
    const wrapper = mount(TimelineArcCard, {
      props: {
        arc,
        arcId: 'arc-1',
        expanded: false,
        thumbnailUrl: '/thumb-2.jpg',
      },
    })

    await wrapper.get('img.timeline-event-card-shell__thumbnail').trigger('error')

    expect(wrapper.find('img.timeline-event-card-shell__thumbnail').exists()).toBe(false)
    expect(wrapper.get('.timeline-event-card-shell__thumbnail-fallback').text()).toBe('第2页')
  })

  it('retries thumbnail rendering when the thumbnail source changes after a failure', async () => {
    const wrapper = mount(TimelineArcCard, {
      props: {
        arc,
        arcId: 'arc-1',
        expanded: false,
        thumbnailUrl: '/thumb-2.jpg',
      },
    })

    await wrapper.get('img.timeline-event-card-shell__thumbnail').trigger('error')
    expect(wrapper.find('.timeline-event-card-shell__thumbnail-fallback').exists()).toBe(true)

    await wrapper.setProps({ thumbnailUrl: '/thumb-2-retry.jpg' })

    const image = wrapper.get('img.timeline-event-card-shell__thumbnail')
    expect(image.attributes('src')).toBe('/thumb-2-retry.jpg')
    expect(wrapper.find('.timeline-event-card-shell__thumbnail-fallback').exists()).toBe(false)
  })

  it('separates group thumbnail navigation from group expansion', async () => {
    const wrapper = mount(TimelineGroupCard, {
      props: {
        expanded: true,
        group,
        thumbnailUrl: '/thumb-8.jpg',
      },
    })

    const thumbnailButton = wrapper.find('.timeline-event-card-shell__thumbnail-action')
    expect(thumbnailButton.element.tagName).toBe('BUTTON')
    expect(thumbnailButton.attributes('aria-label')).toBe('查看第 8 页')

    await thumbnailButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(8)

    const toggleButton = wrapper.find('.timeline-event-card-shell__toggle')
    expect(toggleButton.element.tagName).toBe('BUTTON')
    expect(toggleButton.attributes('aria-expanded')).toBe('true')

    await toggleButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('group-1')
  })

  it('renders a group thumbnail fallback from component state after image errors', async () => {
    const wrapper = mount(TimelineGroupCard, {
      props: {
        expanded: true,
        group,
        thumbnailUrl: '/thumb-8.jpg',
      },
    })

    await wrapper.get('img.timeline-event-card-shell__thumbnail').trigger('error')

    expect(wrapper.find('img.timeline-event-card-shell__thumbnail').exists()).toBe(false)
    expect(wrapper.get('.timeline-event-card-shell__thumbnail-fallback').text()).toBe('第8页')
  })

  it('renders arc and group cards through the shared timeline event shell', () => {
    const arcWrapper = mount(TimelineArcCard, {
      props: {
        arc,
        arcId: 'arc-1',
        expanded: false,
        thumbnailUrl: '/thumb-2.jpg',
      },
    })
    const groupWrapper = mount(TimelineGroupCard, {
      props: {
        expanded: true,
        group,
        thumbnailUrl: '/thumb-8.jpg',
      },
    })

    expect(arcWrapper.find('.timeline-event-card-shell').exists()).toBe(true)
    expect(groupWrapper.find('.timeline-event-card-shell').exists()).toBe(true)
  })

  it('uses native buttons for character page jumps', async () => {
    const wrapper = mount(TimelineCharacterGrid, {
      props: {
        characters: [
          {
            name: '夏',
            description: '主角',
            first_appearance: 3,
          },
        ],
      },
    })

    const recordCard = wrapper.getComponent(ProductRecordCard)
    expect(recordCard.props('as')).toBe('button')

    const characterButton = wrapper.get('button[aria-label="查看角色夏首次出现的第 3 页"]')
    expect(characterButton.element.tagName).toBe('BUTTON')
    expect(characterButton.attributes('aria-label')).toBe('查看角色夏首次出现的第 3 页')

    await characterButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(3)
  })

  it('keeps the character grid responsive inside narrow timeline panels', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/timeline/TimelineCharacterGrid.vue'),
      'utf8',
    )

    expect(source).toContain('minmax(min(100%, 280px), 1fr)')
    expect(source).not.toContain('minmax(280px, 1fr)')
  })

  it('uses native buttons for timeline node toggles', async () => {
    const wrapper = mount(TimelineTrack, {
      props: {
        expandedIds: [],
        groups: [group],
        isEnhancedData: false,
        plotArcs: [],
        thumbnailUrlFor: (pageNum: number) => `/thumb-${pageNum}.jpg`,
      },
    })

    const nodeButton = wrapper.find('.timeline-track__node-dot')
    expect(nodeButton.element.tagName).toBe('BUTTON')
    expect(nodeButton.attributes('aria-expanded')).toBe('false')

    await nodeButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('group-1')
  })

  it('renders stats through the product chip list contract', () => {
    const wrapper = mount(TimelineStats, {
      props: {
        stats: {
          total_events: 12,
          total_pages: 30,
          total_arcs: 3,
          total_characters: 4,
          total_threads: 2,
        },
        totalEvents: 12,
        totalPages: 30,
      },
    })

    const chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('ariaLabel')).toBe('时间线统计')
    expect(chipList.props('items')).toEqual([
      { id: 'arcs', iconName: 'book-marked', label: '3 个剧情弧', tone: 'neutral' },
      { id: 'events', iconName: 'bar-chart', label: '12 个事件', tone: 'neutral' },
      { id: 'characters', iconName: 'users', label: '4 个角色', tone: 'neutral' },
      { id: 'threads', iconName: 'link', label: '2 条线索', tone: 'neutral' },
      { id: 'pages', iconName: 'file-text', label: '30 页', tone: 'neutral' },
    ])
  })

  it('uses the shared spinner for timeline regeneration feedback', () => {
    const wrapper = mount(TimelineHeader, {
      props: {
        isLoading: false,
        isRegenerating: true,
      },
    })

    expect(wrapper.getComponent(UiSpinner).props('size')).toBe(14)
    expect(wrapper.get('button').attributes('aria-busy')).toBe('true')
    expect(wrapper.text()).toContain('生成中...')
  })

  it('renders story summary themes through the product chip contract', () => {
    const wrapper = mount(TimelineSummaryCard, {
      props: {
        storySummary: '主角在冲突中建立新的同盟。',
        plotThreads: [
          { id: 'thread-1', name: '同盟', description: '盟友伏笔' },
          { id: 'thread-2', name: '背叛', description: '反转伏笔' },
          { id: 'thread-3', name: '钥匙', description: '道具伏笔' },
          { id: 'thread-4', name: '预言', description: '主题伏笔' },
          { id: 'thread-5', name: '牺牲', description: '结局伏笔' },
          { id: 'thread-6', name: '隐藏', description: '不应显示' },
        ],
      },
    })

    const chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('ariaLabel')).toBe('故事主题')
    expect(chipList.props('label')).toBe('主题：')
    expect(chipList.props('items')).toEqual([
      { id: 'thread-1', label: '同盟', tone: 'inverse' },
      { id: 'thread-2', label: '背叛', tone: 'inverse' },
      { id: 'thread-3', label: '钥匙', tone: 'inverse' },
      { id: 'thread-4', label: '预言', tone: 'inverse' },
      { id: 'thread-5', label: '牺牲', tone: 'inverse' },
    ])
  })

  it('renders plot threads through product record cards and status chips', () => {
    const wrapper = mount(PlotThreadsList, {
      props: {
        threads: [
          {
            id: 'thread-1',
            name: '钥匙',
            type: 'prop',
            status: '进行中',
            description: '钥匙还没有解释用途。',
            introduced_at: 4,
          },
          {
            id: 'thread-2',
            name: '同盟',
            type: 'relationship',
            status: '已解决',
            description: '同盟关系已经回收。',
            introduced_at: 2,
            resolved_at: 8,
          },
        ],
      },
    })

    const cards = wrapper.findAllComponents(ProductRecordCard)
    expect(cards).toHaveLength(2)
    expect(cards[0].props('accent')).toBe(true)
    expect(cards[0].props('ariaLabel')).toBe('线索：钥匙')

    const chipLists = wrapper.findAllComponents(ProductChipList)
    expect(chipLists[0].props('items')).toEqual([
      { id: 'thread-1-status', label: '进行中', tone: 'warning' },
      { id: 'thread-1-introduced', label: '第 4 页引入', tone: 'neutral' },
    ])
    expect(chipLists[1].props('items')[0]).toEqual({ id: 'thread-2-status', label: '已解决', tone: 'success' })
  })

  it('uses explicit owner hooks across timeline child components', () => {
    const timelineSources = [
      'src/components/insight/timeline/TimelineHeader.vue',
      'src/components/insight/timeline/TimelineSummaryCard.vue',
      'src/components/insight/timeline/TimelineCharacterGrid.vue',
      'src/components/insight/timeline/TimelineTrack.vue',
      'src/components/insight/timeline/TimelineEventCardShell.vue',
      'src/components/insight/timeline/TimelineArcCard.vue',
      'src/components/insight/timeline/TimelineGroupCard.vue',
      'src/components/insight/timeline/PlotThreadsList.vue',
    ].map(file => readFileSync(resolve(process.cwd(), file), 'utf8')).join('\n')

    for (const currentHook of [
      'timeline-header__title',
      'timeline-header__regenerate-action',
      'timeline-summary-card__title',
      'timeline-summary-card__summary',
      'timeline-character-grid__section',
      'timeline-character-grid__card',
      'timeline-track__group',
      'timeline-track__node-dot',
      'timeline-event-card-shell__thumbnail-action',
      'timeline-event-card-shell__toggle',
      'timeline-arc-card__mood-label',
      'timeline-group-card__event-item',
      'plot-threads-list__card',
      'plot-threads-list__thread-name',
    ]) {
      expect(timelineSources).toContain(currentHook)
    }

    for (const legacyHook of [
      'class="characters-section"',
      'class="characters-grid"',
      'class="character-card"',
      'class="character-name"',
      'class="character-desc"',
      'class="one-sentence"',
      'class="timeline-group"',
      'class="timeline-node"',
      'class="timeline-node-dot"',
      'class="timeline-card-header"',
      'class="timeline-thumbnail-action"',
      'class="timeline-card-toggle"',
      'class="timeline-card-title"',
      'class="timeline-page-range"',
      'class="timeline-event-count"',
      'class="expand-icon"',
      'class="timeline-mood"',
      'class="label"',
      'class="timeline-events-list"',
      'class="timeline-event-item"',
      'class="plot-thread-card"',
      'class="thread-name"',
      'class="thread-desc"',
    ]) {
      expect(timelineSources).not.toContain(legacyHook)
    }

    expect(timelineSources).not.toContain('.timeline-header h3')
    expect(timelineSources).not.toContain('.timeline-summary-card h4')
    expect(timelineSources).not.toContain('.characters-section h4')
  })
})
