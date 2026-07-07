import { mount } from '@vue/test-utils'
import { defineComponent, h, nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['title'],
    emits: ['close'],
    setup(props, { emit, slots }) {
      return () => h('section', { role: 'dialog', 'aria-label': props.title }, [
        h('button', { type: 'button', class: 'modal-close', onClick: () => emit('close') }, '关闭'),
        slots.default?.(),
        slots.footer?.(),
      ])
    },
  }),
}))

import ProductConfirmProvider from '@/components/product/ProductConfirmProvider.vue'
import { confirmProductAction } from '@/composables/useProductConfirm'

describe('ProductConfirmProvider', () => {
  it('resolves a product confirmation through the shared modal provider', async () => {
    const wrapper = mount(ProductConfirmProvider)
    const result = confirmProductAction({
      title: '删除图片',
      message: '确定要删除当前图片吗？',
      confirmText: '删除',
      tone: 'danger',
    })

    await nextTick()

    expect(wrapper.get('[role="dialog"]').attributes('aria-label')).toBe('删除图片')
    expect(wrapper.text()).toContain('确定要删除当前图片吗？')

    const confirmButton = wrapper.findAll('button').find(button => button.text() === '删除')
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')

    await expect(result).resolves.toBe(true)
  })

  it('resolves false when the shared modal is cancelled', async () => {
    const wrapper = mount(ProductConfirmProvider)
    const result = confirmProductAction({
      title: '清空图片',
      message: '确定要清空所有图片吗？',
    })

    await nextTick()

    await wrapper.get('.modal-close').trigger('click')

    await expect(result).resolves.toBe(false)
  })
})
