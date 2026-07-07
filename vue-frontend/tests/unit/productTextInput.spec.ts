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

import ProductTextInputProvider from '@/components/product/ProductTextInputProvider.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { requestProductTextInput } from '@/composables/useProductTextInput'

describe('ProductTextInputProvider', () => {
  it('resolves entered text through the shared input provider', async () => {
    const wrapper = mount(ProductTextInputProvider)
    const result = requestProductTextInput({
      title: '保存提示词',
      message: '请输入提示词名称：',
      placeholder: '名称',
      confirmText: '保存',
    })

    await nextTick()

    expect(wrapper.get('[role="dialog"]').attributes('aria-label')).toBe('保存提示词')
    expect(wrapper.text()).toContain('请输入提示词名称：')
    expect(wrapper.get('input').attributes('aria-label')).toBe('名称')

    await wrapper.get('input').setValue('战斗分析')
    const confirmButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')

    await expect(result).resolves.toBe('战斗分析')
  })

  it('resolves null when the shared input is cancelled', async () => {
    const wrapper = mount(ProductTextInputProvider)
    const result = requestProductTextInput({
      title: '命名',
      message: '请输入名称',
    })

    await nextTick()

    await wrapper.get('.modal-close').trigger('click')

    await expect(result).resolves.toBeNull()
  })

  it('uses the product dialog action row for shared input footer actions', async () => {
    const wrapper = mount(ProductTextInputProvider)
    requestProductTextInput({
      title: '命名',
      message: '请输入名称',
    })

    await nextTick()

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('文本输入操作')
  })
})
