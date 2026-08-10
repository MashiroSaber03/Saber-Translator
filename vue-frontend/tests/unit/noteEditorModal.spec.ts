import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import NoteEditorModal from '@/components/insight/notes/NoteEditorModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
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
        visible: true,
      },
      global: {
        stubs: {
          BaseModal: baseModalStub,
        },
      },
    })

    const citations = wrapper.getComponent(ProductChipList)
    expect(citations.props('ariaLabel')).toBe('引用页码')
    expect(citations.props('items')).toEqual([
      {
        id: 7,
        label: '第7页',
        ariaLabel: '查看第 7 页',
        interactive: true,
        tone: 'primary',
      },
    ])

    citations.vm.$emit('select', 7)

    expect(wrapper.emitted('showPage')?.[0]?.[0]).toBe(7)

    expect(wrapper.getComponent(ProductDetailPanel).props('ariaLabel')).toBe('问答笔记预览')
    const detailSections = wrapper.findAllComponents(ProductDetailSection)
    expect(detailSections.map(section => section.props('label'))).toEqual(['问题', '回答', '引用页码'])
    expect(detailSections.map(section => section.props('framed'))).toEqual([true, true, false])
  })

  it('uses product field primitives for editable text notes', () => {
    const wrapper = mount(NoteEditorModal, {
      props: {
        editingNote: null,
        noteContent: '记录内容',
        notePageNum: 8,
        noteTags: '剧情',
        noteTitle: '伏笔',
        visible: true,
      },
      global: {
        stubs: {
          BaseModal: baseModalStub,
        },
      },
    })

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('variant'))).toEqual([
      'settings',
      'settings',
      'settings',
      'settings',
    ])
    expect(fields.map(field => field.props('label'))).toEqual([
      '标题',
      '内容',
      '关联页码',
      '标签',
    ])
    expect(fields.map(field => field.props('required'))).toEqual([
      false,
      true,
      false,
      false,
    ])

    expect(wrapper.text()).not.toContain('笔记类型')
    expect(wrapper.text()).not.toContain('问答笔记')
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')
  })

  it('renders note editor actions through the product dialog action row', () => {
    const wrapper = mount(NoteEditorModal, {
      props: {
        editingNote: null,
        noteContent: '记录内容',
        notePageNum: 8,
        noteTags: '剧情',
        noteTitle: '伏笔',
        visible: true,
      },
      global: {
        stubs: {
          BaseModal: baseModalStub,
        },
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)

    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('笔记编辑操作')
  })

  it('keeps modal body styling scoped to the note editor owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/notes/NoteEditorModal.vue'),
      'utf8',
    )

    expect(source).toContain('class="note-editor-modal__body"')
    expect(source).not.toContain('notes-modal-body')
  })
})
