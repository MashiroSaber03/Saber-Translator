import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import QASaveNoteModal from '@/components/insight/qa/QASaveNoteModal.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const baseModalStub = defineComponent({
  template: '<div><slot name="title" /><slot /><slot name="footer" /></div>',
})

describe('QASaveNoteModal', () => {
  it('uses product field primitives for note metadata and citation preview chips', () => {
    const wrapper = mount(QASaveNoteModal, {
      props: {
        noteComment: '',
        noteTitle: '角色动机',
        pendingQAData: {
          messageId: 'msg-1',
          question: '主角为什么离开？',
          answer: '**为了保护同伴。**',
          citations: [{ page: 5 }],
        },
        renderMarkdown: (content: string) => content,
        visible: true,
      },
      global: {
        stubs: {
          BaseModal: baseModalStub,
        },
      },
    })

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('variant'))).toEqual(['settings', 'settings'])
    expect(fields.map(field => field.props('label'))).toEqual(['笔记标题', '补充说明'])
    expect(fields.map(field => field.props('hint'))).toEqual(['可选', '可选'])
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')

    const citations = wrapper.getComponent(ProductChipList)
    expect(citations.props('ariaLabel')).toBe('引用页码')
    expect(citations.props('items')).toEqual([
      { id: 5, label: '第5页', tone: 'primary' },
    ])

    expect(wrapper.getComponent(ProductDetailPanel).props('ariaLabel')).toBe('问答预览')
    const detailSections = wrapper.findAllComponents(ProductDetailSection)
    expect(detailSections.map(section => section.props('label'))).toEqual(['问题', '回答', '引用页码'])
    expect(detailSections.map(section => section.props('framed'))).toEqual([true, true, false])
  })

  it('renders save-note actions through the product dialog action row', () => {
    const wrapper = mount(QASaveNoteModal, {
      props: {
        noteComment: '',
        noteTitle: '角色动机',
        pendingQAData: null,
        renderMarkdown: (content: string) => content,
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
    expect(actionRow.props('ariaLabel')).toBe('问答笔记保存操作')
  })

  it('keeps save-note modal hooks under the qa-note-modal owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/qa/QASaveNoteModal.vue'), 'utf8')

    expect(source).not.toContain('qa-note-modal-body')
    expect(source).not.toMatch(/(?<![\w-])note-form(?![\w-])/)
    expect(source).toContain('qa-note-modal__body')
    expect(source).toContain('qa-note-modal__form')
  })
})
