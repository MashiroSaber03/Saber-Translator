import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import NoteCard from '@/components/insight/notes/NoteCard.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { NoteData } from '@/stores/insightStore'

const qaNote: NoteData = {
  id: 'note-1',
  type: 'qa',
  title: '角色动机',
  content: '问答摘要',
  question: '主角为什么离开？',
  answer: '为了保护同伴。',
  citations: [{ page: 5 }],
  pageNum: 3,
  tags: ['剧情'],
  createdAt: '2026-05-21T10:00:00Z',
  updatedAt: '2026-05-21T10:00:00Z',
}

describe('NoteCard', () => {
  it('uses explicit controls for editing and citation navigation', async () => {
    const wrapper = mount(NoteCard, {
      props: {
        note: qaNote,
      },
    })

    const recordCard = wrapper.getComponent(ProductRecordCard)
    expect(recordCard.props('accent')).toBe(true)
    expect(recordCard.props('ariaLabel')).toBe('笔记：角色动机')
    expect(recordCard.attributes('role')).toBe('listitem')

    const openButton = wrapper.get('button[aria-label="编辑笔记：角色动机"]')
    expect(openButton.element.tagName).toBe('BUTTON')
    expect(wrapper.find('button[aria-label="编辑笔记：角色动机"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="删除笔记：角色动机"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="编辑"]').exists()).toBe(false)
    expect(wrapper.find('button[aria-label="删除"]').exists()).toBe(false)

    await openButton.trigger('click')
    expect(wrapper.emitted('edit')?.[0]?.[0]).toEqual(qaNote)

    const chipLists = wrapper.findAllComponents(ProductChipList)
    expect(chipLists.map(list => list.props('ariaLabel'))).toEqual([
      '笔记标签',
      '引用页码',
      '关联页码',
    ])
    expect(chipLists[0]!.props('items')).toEqual([
      { id: '剧情', label: '剧情', tone: 'neutral' },
    ])
    expect(chipLists[1]!.props('items')).toEqual([
      {
        id: 5,
        label: '第5页',
        ariaLabel: '查看第 5 页',
        interactive: true,
        tone: 'primary',
      },
    ])

    chipLists[1]!.vm.$emit('select', 5)
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(5)

    chipLists[2]!.vm.$emit('select', 3)
    expect(wrapper.emitted('showPage')?.[1]?.[0]).toBe(3)
  })

  it('keeps NoteCard-owned hooks scoped to the note-card owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/notes/NoteCard.vue'),
      'utf8',
    )

    expect(source).toContain('class="note-card__open-button"')
    expect(source).toContain('class="note-card__title"')
    expect(source).toContain('class="note-card__content"')
    expect(source).not.toMatch(/\.(?:note-open-button|note-title|note-content|note-date|note-tags|qa-preview-text|note-type-icon)\b/)
  })
})
