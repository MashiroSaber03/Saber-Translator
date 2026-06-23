import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import NoteCard from '@/components/insight/notes/NoteCard.vue'
import type { NoteData } from '@/stores/insightStore'

const qaNote: NoteData = {
  id: 'note-1',
  type: 'qa',
  title: '角色动机',
  content: '问答摘要',
  question: '主角为什么离开？',
  answer: '为了保护同伴。',
  citations: [{ page: 5 }],
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

    const root = wrapper.find('.note-item')
    expect(root.attributes('role')).toBeUndefined()
    expect(root.attributes('tabindex')).toBeUndefined()

    const openButton = wrapper.find('.note-open-button')
    expect(openButton.exists()).toBe(true)
    expect(openButton.element.tagName).toBe('BUTTON')
    expect(openButton.attributes('aria-label')).toBe('编辑笔记：角色动机')

    await openButton.trigger('click')
    expect(wrapper.emitted('edit')?.[0]?.[0]).toEqual(qaNote)

    const citationButton = wrapper.find('.citation-badge')
    expect(citationButton.element.tagName).toBe('BUTTON')
    expect(citationButton.attributes('aria-label')).toBe('查看第 5 页')

    await citationButton.trigger('click')
    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(5)
  })
})
