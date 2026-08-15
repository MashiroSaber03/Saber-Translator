import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const apiMocks = vi.hoisted(() => ({
  promoteQuickWorkspace: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  promoteQuickWorkspace: apiMocks.promoteQuickWorkspace,
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    name: 'BaseModalStub',
    props: {
      modelValue: { type: Boolean, default: false },
    },
    emits: ['close'],
    setup(props, { slots }) {
      return () => props.modelValue
        ? h('div', [slots.default?.(), slots.footer?.()])
        : null
    },
  }),
}))

import QuickWorkspacePromoteModal from '@/components/translate/QuickWorkspacePromoteModal.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'

describe('QuickWorkspacePromoteModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    apiMocks.promoteQuickWorkspace.mockReset()
    apiMocks.promoteQuickWorkspace.mockResolvedValue({
      bookId: 'book-1',
      chapterId: 'chapter-1',
    })
  })

  it('loads the bookshelf when initially mounted open and displays load failures', async () => {
    const store = useBookshelfStore()
    store.loadBooks = vi.fn(async () => {
      store.error = '网络不可用'
    })

    const wrapper = mount(QuickWorkspacePromoteModal, {
      props: { modelValue: true },
    })
    await flushPromises()

    expect(store.loadBooks).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('加载书架失败：网络不可用')
  })

  it('ignores IME confirmation Enter and submits a normal Enter once', async () => {
    const store = useBookshelfStore()
    store.loadBooks = vi.fn(async () => {})
    const wrapper = mount(QuickWorkspacePromoteModal, {
      props: { modelValue: true },
    })
    await wrapper.get('#quick-promote-book-title').setValue('新书')
    const chapterInput = wrapper.get('#quick-promote-chapter-title')
    await chapterInput.setValue('第一章')

    await chapterInput.trigger('keydown', { key: 'Enter', isComposing: true })
    expect(apiMocks.promoteQuickWorkspace).not.toHaveBeenCalled()

    await chapterInput.trigger('keydown', { key: 'Enter' })
    await flushPromises()

    expect(apiMocks.promoteQuickWorkspace).toHaveBeenCalledTimes(1)
    expect(apiMocks.promoteQuickWorkspace).toHaveBeenCalledWith({
      mode: 'new_book',
      title: '新书',
      chapterTitle: '第一章',
    })
  })

  it('does not close the modal while a promotion request is in flight', async () => {
    let resolvePromotion!: (value: { bookId: string; chapterId: string }) => void
    apiMocks.promoteQuickWorkspace.mockReturnValueOnce(new Promise(resolve => {
      resolvePromotion = resolve
    }))
    const store = useBookshelfStore()
    store.loadBooks = vi.fn(async () => {})
    const wrapper = mount(QuickWorkspacePromoteModal, {
      props: { modelValue: true },
    })
    await wrapper.get('#quick-promote-book-title').setValue('新书')
    await wrapper.get('#quick-promote-chapter-title').setValue('第一章')
    await wrapper.findAll('button').find(button => button.text() === '保存到书架')!.trigger('click')

    wrapper.getComponent({ name: 'BaseModalStub' }).vm.$emit('close')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()

    resolvePromotion({ bookId: 'book-1', chapterId: 'chapter-1' })
    await flushPromises()
    expect(wrapper.emitted('promoted')).toEqual([[
      { bookId: 'book-1', chapterId: 'chapter-1' },
    ]])
    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
  })
})
