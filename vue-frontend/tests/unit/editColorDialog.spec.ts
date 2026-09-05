import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import BubbleEditor from '@/components/edit/BubbleEditor.vue'
import EditColorDialog from '@/components/edit/EditColorDialog.vue'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { BubbleColorField } from '@/types/bubble'

vi.mock('@/api/v2/settings', () => ({ listV2Fonts: vi.fn().mockResolvedValue([]) }))
const wrappers: Array<ReturnType<typeof mount>> = []
afterEach(() => wrappers.splice(0).forEach(wrapper => wrapper.unmount()))

function editor() {
  const wrapper = mount(BubbleEditor, {
    props: { bubble: createBubbleState({ inpaintMethod: 'solid', strokeEnabled: true }), bubbleIndex: 0 },
    global: { plugins: [createPinia()], stubs: { teleport: true, UiCombobox: true, JapaneseKeyboard: true } },
  })
  wrappers.push(wrapper)
  return wrapper
}

describe('editor color dialog', () => {
  it.each([
    ['文字颜色', 'textColor'], ['背景填充颜色', 'fillColor'], ['描边颜色', 'strokeColor'],
  ] as Array<[string, BubbleColorField]>)('%s uses image picking without a native screen picker', async (label, field) => {
    const wrapper = editor()
    await wrapper.get(`button[aria-label="${label}"]`).trigger('click')
    const dialog = wrapper.getComponent(EditColorDialog)
    expect(wrapper.find('input[type="color"]').exists()).toBe(false)
    await dialog.findAll('button').find(button => button.text() === '从图片取色')!.trigger('click')
    expect(wrapper.emitted('pickColor')).toEqual([[field]])
    expect(wrapper.emitted('update')).toBeUndefined()
    expect(wrapper.findComponent(EditColorDialog).exists()).toBe(false)
  })

  it('validates manual HEX entry and commits only when applied', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const dialog = wrapper.getComponent(EditColorDialog)
    const apply = () => dialog.findAll('button').find(button => button.text() === '应用颜色')!
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#oops')
    expect(apply().attributes('disabled')).toBeDefined()
    await dialog.get('input[aria-label="HEX 色值"]').setValue('12AB56')
    expect(wrapper.emitted('update')).toBeUndefined()
    await apply().trigger('click')
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#12ab56' }]])
  })

  it('supports RGB adjustments and recovers invalid HEX input through a color swatch', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const dialog = wrapper.getComponent(EditColorDialog)
    await dialog.get('input[aria-label="红（R）"]').setValue('17')
    expect(dialog.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#110000')
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#oops')
    await dialog.get('button[aria-label="白色"]').trigger('click')
    await dialog.findAll('button').find(button => button.text() === '应用颜色')!.trigger('click')
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#ffffff' }]])
  })

  it('discards draft colors when cancelled or the selected bubble changes', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    let dialog = wrapper.getComponent(EditColorDialog)
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#123456')
    await dialog.findAll('button').find(button => button.text() === '取消')!.trigger('click')
    expect(wrapper.emitted('update')).toBeUndefined()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    dialog = wrapper.getComponent(EditColorDialog)
    expect(dialog.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#000000')
    await wrapper.setProps({ bubble: createBubbleState({ backendBubbleId: 'new' }) })
    expect(wrapper.findComponent(EditColorDialog).exists()).toBe(false)
    expect(wrapper.emitted('update')).toBeUndefined()
  })
})
