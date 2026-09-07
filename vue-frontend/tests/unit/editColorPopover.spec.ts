import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import BubbleEditor from '@/components/edit/BubbleEditor.vue'
import EditColorPopover from '@/components/edit/EditColorPopover.vue'
import { createBubbleState } from '@/utils/bubbleFactory'
import type { BubbleColorField } from '@/types/bubble'

vi.mock('@/api/v2/settings', () => ({ listV2Fonts: vi.fn().mockResolvedValue([]) }))
const wrappers: Array<ReturnType<typeof mount>> = []
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe = vi.fn()
    disconnect = vi.fn()
  })
})
afterEach(() => {
  wrappers.splice(0).forEach(wrapper => wrapper.unmount())
  vi.unstubAllGlobals()
})

function editor() {
  const wrapper = mount(BubbleEditor, {
    attachTo: document.body,
    props: { bubble: createBubbleState({ inpaintMethod: 'solid', strokeEnabled: true }), bubbleIndex: 0 },
    global: { plugins: [createPinia()], stubs: { teleport: true, UiCombobox: true, JapaneseKeyboard: true } },
  })
  wrappers.push(wrapper)
  return wrapper
}

describe('editor color popover', () => {
  it.each([
    ['文字颜色', 'textColor'], ['背景填充颜色', 'fillColor'], ['描边颜色', 'strokeColor'],
  ] as Array<[string, BubbleColorField]>)('%s uses image picking without a native screen picker', async (label, field) => {
    const wrapper = editor()
    await wrapper.get(`button[aria-label="${label}"]`).trigger('click')
    const dialog = wrapper.getComponent(EditColorPopover)
    expect(wrapper.find('input[type="color"]').exists()).toBe(false)
    await dialog.findAll('button').find(button => button.text() === '从图片取色')!.trigger('click')
    expect(wrapper.emitted('pickColor')).toEqual([[field]])
    expect(wrapper.emitted('update')).toBeUndefined()
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
  })

  it('applies valid HEX immediately and keeps the popover open', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const dialog = wrapper.getComponent(EditColorPopover)
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#oops')
    expect(wrapper.emitted('update')).toBeUndefined()
    await dialog.get('input[aria-label="HEX 色值"]').setValue('12AB56')
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#12ab56' }]])
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(true)
    expect(dialog.text()).not.toContain('应用颜色')
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#234567')
    expect(wrapper.emitted('update')?.at(-1)).toEqual([{ textColor: '#234567' }])
  })

  it('supports RGB adjustments and recovers invalid HEX input through a color swatch', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const dialog = wrapper.getComponent(EditColorPopover)
    await dialog.get('input[aria-label="红（R）"]').setValue('17')
    expect(dialog.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#110000')
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#oops')
    await dialog.get('button[aria-label="白色"]').trigger('click')
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#110000' }], [{ textColor: '#ffffff' }]])
  })

  it('keeps the applied color when closed and closes when selection changes', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const dialog = wrapper.getComponent(EditColorPopover)
    await dialog.get('input[aria-label="HEX 色值"]').setValue('#123456')
    await dialog.findAll('button').find(button => button.text() === '关闭')!.trigger('click')
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#123456' }]])
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    expect(wrapper.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#123456')
    await wrapper.setProps({ bubble: createBubbleState({ backendBubbleId: 'new' }) })
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
    expect(wrapper.emitted('update')).toHaveLength(1)
  })

  it.each(['backend', 'client'])('keeps the color picker open when saving replaces the same %s bubble', async identity => {
    const wrapper = editor()
    const bubble = createBubbleState(identity === 'backend'
      ? { backendBubbleId: 'saved-bubble' }
      : { clientMutationId: 'new-bubble' })
    await wrapper.setProps({ bubble })
    await wrapper.get('button[aria-label="描边颜色"]').trigger('click')
    await wrapper.getComponent(EditColorPopover).get('input[aria-label="HEX 色值"]').setValue('#123456')

    await wrapper.setProps({ bubble: { ...bubble, strokeColor: '#123456', strokeWidth: 0.5 } })

    const dialog = wrapper.getComponent(EditColorPopover)
    expect(dialog.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#123456')
    expect(wrapper.emitted('update')).toEqual([[{ strokeColor: '#123456' }]])
  })

  it('anchors to the clicked button, toggles closed and switches fields without reverting applied colors', async () => {
    const wrapper = editor()
    const text = wrapper.get('button[aria-label="文字颜色"]')
    await text.trigger('click')
    expect(wrapper.getComponent(EditColorPopover).props('anchor')).toBe(text.element)
    expect(text.attributes('aria-expanded')).toBe('true')
    expect(wrapper.get('[role="dialog"]').attributes('aria-modal')).not.toBe('true')
    await wrapper.get('input[aria-label="HEX 色值"]').setValue('#123456')

    const fill = wrapper.get('button[aria-label="背景填充颜色"]')
    await fill.trigger('pointerdown')
    await fill.trigger('click')
    expect(wrapper.findAllComponents(EditColorPopover)).toHaveLength(1)
    expect(wrapper.getComponent(EditColorPopover).props('anchor')).toBe(fill.element)
    expect(wrapper.get('input[aria-label="HEX 色值"]').element).toHaveProperty('value', '#FFFFFF')
    expect(text.attributes('aria-expanded')).toBe('false')
    await fill.trigger('pointerdown')
    await fill.trigger('click')
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
    expect(wrapper.emitted('update')).toEqual([[{ textColor: '#123456' }]])
  })

  it('dismisses on an outside pointer press without blocking its target', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    const event = new MouseEvent('pointerdown', { bubbles: true, cancelable: true })
    document.body.dispatchEvent(event)
    await wrapper.vm.$nextTick()
    expect(event.defaultPrevented).toBe(false)
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
    expect(wrapper.emitted('update')).toBeUndefined()
  })

  it('consumes Escape and returns focus to its button without changing the color', async () => {
    const wrapper = editor()
    const trigger = wrapper.get('button[aria-label="文字颜色"]')
    await trigger.trigger('click')
    await wrapper.get('[role="dialog"]').trigger('keydown', { key: 'Escape' })
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
    expect(document.activeElement).toBe(trigger.element)
    expect(wrapper.emitted('update')).toBeUndefined()
  })

  it('closes when focus leaves the popover, including to browser chrome', async () => {
    const wrapper = editor()
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    await wrapper.get('[role="dialog"]').trigger('focusout', { relatedTarget: wrapper.get('input[aria-label="红（R）"]').element })
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(true)
    await wrapper.get('[role="dialog"]').trigger('focusout', { relatedTarget: null })
    expect(wrapper.findComponent(EditColorPopover).exists()).toBe(false)
    expect(wrapper.emitted('update')).toBeUndefined()
  })
})
