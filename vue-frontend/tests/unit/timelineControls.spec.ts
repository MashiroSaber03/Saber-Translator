import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import TimelineArcCard from '@/components/insight/timeline/TimelineArcCard.vue'
import TimelineCharacterGrid from '@/components/insight/timeline/TimelineCharacterGrid.vue'
import TimelineGroupCard from '@/components/insight/timeline/TimelineGroupCard.vue'
import TimelineTrack from '@/components/insight/timeline/TimelineTrack.vue'

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

    const thumbnailButton = wrapper.find('.timeline-thumbnail-action')
    expect(thumbnailButton.element.tagName).toBe('BUTTON')
    expect(thumbnailButton.attributes('aria-label')).toBe('查看第 2 页')

    await thumbnailButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(2)

    const toggleButton = wrapper.find('.timeline-card-toggle')
    expect(toggleButton.element.tagName).toBe('BUTTON')
    expect(toggleButton.attributes('aria-expanded')).toBe('false')

    await toggleButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('arc-1')
  })

  it('separates group thumbnail navigation from group expansion', async () => {
    const wrapper = mount(TimelineGroupCard, {
      props: {
        expanded: true,
        group,
        thumbnailUrl: '/thumb-8.jpg',
      },
    })

    const thumbnailButton = wrapper.find('.timeline-thumbnail-action')
    expect(thumbnailButton.element.tagName).toBe('BUTTON')
    expect(thumbnailButton.attributes('aria-label')).toBe('查看第 8 页')

    await thumbnailButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(8)

    const toggleButton = wrapper.find('.timeline-card-toggle')
    expect(toggleButton.element.tagName).toBe('BUTTON')
    expect(toggleButton.attributes('aria-expanded')).toBe('true')

    await toggleButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('group-1')
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

    const characterButton = wrapper.find('.character-card')
    expect(characterButton.element.tagName).toBe('BUTTON')
    expect(characterButton.attributes('aria-label')).toBe('查看角色夏首次出现的第 3 页')

    await characterButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(3)
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

    const nodeButton = wrapper.find('.timeline-node-dot')
    expect(nodeButton.element.tagName).toBe('BUTTON')
    expect(nodeButton.attributes('aria-expanded')).toBe('false')

    await nodeButton.trigger('click')
    expect(wrapper.emitted('toggle')?.[0]?.[0]).toBe('group-1')
  })
})
