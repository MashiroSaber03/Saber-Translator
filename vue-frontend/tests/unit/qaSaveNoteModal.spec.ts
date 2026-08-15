import { mount } from '@vue/test-utils'
import { defineComponent, reactive } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import QASaveNoteModal from '@/components/insight/qa/QASaveNoteModal.vue'
import { useQANoteModal } from '@/components/insight/useQANoteModal'
import type { useInsightStore } from '@/stores/insightStore'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const { showToastMock } = vi.hoisted(() => ({ showToastMock: vi.fn() }))

vi.mock('@/utils/toast', () => ({ showToast: showToastMock }))

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
    expect(citations.props('items')).toEqual([{ id: 5, label: '第5页', tone: 'primary' }])

    expect(wrapper.getComponent(ProductDetailPanel).props('ariaLabel')).toBe('问答预览')
    const detailSections = wrapper.findAllComponents(ProductDetailSection)
    expect(detailSections.map(section => section.props('label'))).toEqual([
      '问题',
      '回答',
      '引用页码',
    ])
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
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QASaveNoteModal.vue'),
      'utf8'
    )

    expect(source).not.toContain('qa-note-modal-body')
    expect(source).not.toMatch(/(?<![\w-])note-form(?![\w-])/)
    expect(source).toContain('qa-note-modal__body')
    expect(source).toContain('qa-note-modal__form')
  })

  it('keeps the complete question as the default title and ignores duplicate saves', async () => {
    let finishSave!: () => void
    const savePromise = new Promise<void>(resolve => {
      finishSave = resolve
    })
    const addNote = vi.fn().mockReturnValue(savePromise)
    const question =
      '这是一个明显超过三十个字符的完整问题，保存为笔记时不应该被任意截断或丢失任何内容。'
    const store = reactive({
      currentBookId: 'book-1',
      qaHistory: [
        { id: 'user-1', role: 'user', content: question },
        { id: 'assistant-1', role: 'assistant', content: '完整回答', mode: 'precise' },
      ],
      addNote,
    }) as unknown as ReturnType<typeof useInsightStore>
    const modal = useQANoteModal(store)
    modal.openNoteModal(store.qaHistory[1]!)
    modal.noteTitle.value = '   '
    modal.noteComment.value = '   '

    const firstSave = modal.saveNote()
    await modal.saveNote()

    expect(addNote).toHaveBeenCalledTimes(1)
    expect(addNote).toHaveBeenCalledWith(
      expect.objectContaining({
        title: question,
        question,
        content: '完整回答',
      })
    )
    expect(addNote.mock.calls[0]?.[0]).not.toHaveProperty('comment')
    finishSave()
    await firstSave
    expect(store.qaHistory[1]?.saved).toBe(true)
  })

  it('shows the actual backend note error when saving fails', async () => {
    showToastMock.mockReset()
    const store = reactive({
      currentBookId: 'book-1',
      notesError: '笔记 revision 已变化',
      qaHistory: [
        { id: 'user-1', role: 'user', content: '发生了什么？' },
        { id: 'assistant-1', role: 'assistant', content: '回答' },
      ],
      addNote: vi.fn().mockRejectedValue(new Error('request failed')),
    }) as unknown as ReturnType<typeof useInsightStore>
    const modal = useQANoteModal(store)
    modal.openNoteModal(store.qaHistory[1]!)

    await modal.saveNote()

    expect(showToastMock).toHaveBeenCalledWith('笔记 revision 已变化', 'error')
  })
})
