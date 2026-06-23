import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'

import NoteEditorModal from '@/components/insight/notes/NoteEditorModal.vue'
import type { NoteData } from '@/stores/insightStore'

const baseModalStub = defineComponent({
  template: '<div><slot name="title" /><slot /><slot name="footer" /></div>',
})

const qaNote: NoteData = {
  id: 'note-qa',
  type: 'qa',
  title: '问答笔记',
  content: '问答内容',
  question: '发生了什么？',
  answer: '角色完成了计划。',
  citations: [{ page: 7 }],
  createdAt: '2026-05-21T10:00:00Z',
  updatedAt: '2026-05-21T10:00:00Z',
}

describe('NoteEditorModal', () => {
  it('uses button semantics for QA citation navigation', async () => {
    const wrapper = mount(NoteEditorModal, {
      props: {
        editingNote: qaNote,
        noteContent: '',
        notePageNum: null,
        noteTags: '',
        noteTitle: '问答笔记',
        noteType: 'qa',
        noteTypeOptions: [
          { label: '问答笔记', value: 'qa' },
          { label: '文本笔记', value: 'text' },
        ],
        visible: true,
      },
      global: {
        stubs: {
          BaseModal: baseModalStub,
        },
      },
    })

    const citationButton = wrapper.find('.qa-citation-badge')
    expect(citationButton.element.tagName).toBe('BUTTON')
    expect(citationButton.attributes('aria-label')).toBe('查看第 7 页')

    await citationButton.trigger('click')

    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(7)
  })
})
