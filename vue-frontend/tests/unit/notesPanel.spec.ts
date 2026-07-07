import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

const { confirmProductActionMock } = vi.hoisted(() => ({
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import NotesPanel from '@/components/insight/NotesPanel.vue'
import NotesList from '@/components/insight/notes/NotesList.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

describe('NotesPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('uses product confirmation before deleting a note', async () => {
    const store = useInsightStore()
    const deleteNoteSpy = vi.spyOn(store, 'deleteNote').mockResolvedValue(undefined)
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)

    const wrapper = mount(NotesPanel, {
      global: {
        stubs: {
          NotesToolbar: defineComponent({
            emits: ['update:filter'],
            setup() {
              return () => h('div', { class: 'notes-toolbar-stub' })
            },
          }),
          NotesList: defineComponent({
            emits: ['delete', 'edit', 'showPage'],
            setup(_props, { emit }) {
              return () => h('button', {
                type: 'button',
                class: 'delete-note',
                onClick: () => emit('delete', 'note-1'),
              }, '删除笔记')
            },
          }),
          NoteEditorModal: defineComponent({
            emits: ['close', 'save', 'showPage'],
            setup() {
              return () => h('div', { class: 'note-editor-modal-stub' })
            },
          }),
        },
      },
    })

    await wrapper.get('.delete-note').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除笔记',
      message: '确定要删除这条笔记吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(deleteNoteSpy).toHaveBeenCalledWith('note-1')
  })

  it('renders notes through the product scroll stack contract', () => {
    const wrapper = mount(NotesList, {
      props: {
        notes: [],
      },
    })

    const stack = wrapper.getComponent(ProductScrollStack)
    expect(stack.props('role')).toBe('list')
    expect(stack.props('ariaLabel')).toBe('笔记列表')
    expect(stack.props('empty')).toBe(true)
    expect(wrapper.text()).toContain('暂无笔记')
  })

  it('renders the empty notes state through product status feedback', () => {
    const wrapper = mount(NotesList, {
      props: {
        notes: [],
      },
    })
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/notes/NotesList.vue'), 'utf8')
    const emptyState = wrapper.getComponent(ProductStatusBanner)

    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      role: 'note',
      iconName: 'file-text',
      title: '暂无笔记',
    })
    expect(wrapper.text()).toContain('添加笔记后会显示在这里。')
    expect(source).toContain("import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'")
    expect(source).toContain('class="notes-list__empty-status"')
    expect(source).not.toContain('notes-list-empty-status')
    expect(source).not.toContain('placeholder-text')
  })

  it('does not redefine the shared button primitive skin in the notes panel owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/NotesPanel.vue'), 'utf8')

    expect(source).not.toContain('--ui-button-')
    expect(source).not.toContain('class="btn-block"')
    expect(source).not.toContain('+ 添加笔记')
    expect(source).toContain('class="notes-panel"')
    expect(source).toContain('class="notes-panel__add-button"')
    expect(source).not.toContain('workspace-section notes-section')
    expect(source).not.toContain('.workspace-section.notes-section')
    expect(source).toContain('<UiIcon name="plus"')
  })

  it('renders the add-note command with the shared icon language', () => {
    const wrapper = mount(NotesPanel, {
      global: {
        stubs: {
          NotesToolbar: defineComponent({
            emits: ['update:filter'],
            setup() {
              return () => h('div', { class: 'notes-toolbar-stub' })
            },
          }),
          NotesList: defineComponent({
            emits: ['delete', 'edit', 'showPage'],
            setup() {
              return () => h('div', { class: 'notes-list-stub' })
            },
          }),
          NoteEditorModal: defineComponent({
            emits: ['close', 'save', 'showPage'],
            setup() {
              return () => h('div', { class: 'note-editor-modal-stub' })
            },
          }),
        },
      },
    })

    const addButton = wrapper.get('.notes-panel__add-button')

    expect(addButton.getComponent(UiIcon).props('name')).toBe('plus')
    expect(addButton.text()).toContain('添加笔记')
    expect(addButton.text()).not.toContain('+')
  })
})
