import { mount } from '@vue/test-utils'
import { nextTick, ref } from 'vue'
import { afterEach, describe, expect, it } from 'vitest'
import UiColorPicker from '@/components/ui/UiColorPicker.vue'

const wrappers: Array<ReturnType<typeof mount>> = []
afterEach(() => wrappers.splice(0).forEach(wrapper => wrapper.unmount()))

function picker(color = '#000000') {
  const wrapper = mount({
    components: { UiColorPicker },
    setup: () => ({ color: ref(color) }),
    template: '<UiColorPicker v-model="color" />',
  })
  wrappers.push(wrapper)
  const spectrum = wrapper.get('[role="slider"][aria-label="色盘"]')
  let captured: number | null = null
  Object.assign(spectrum.element, {
    getBoundingClientRect: () => new DOMRect(10, 20, 200, 120),
    setPointerCapture: (id: number) => { captured = id },
    hasPointerCapture: (id: number) => captured === id,
    releasePointerCapture: () => { captured = null },
  })
  const hex = () => (wrapper.get('input[aria-label="HEX 色值"]').element as HTMLInputElement).value
  const pointer = async (type: string, options: PointerEventInit) => {
    spectrum.element.dispatchEvent(new PointerEvent(type, { bubbles: true, ...options }))
    await nextTick()
  }
  return { wrapper, spectrum, hex, pointer }
}

describe('continuous color picker', () => {
  it('keeps a chosen hue while black and supports point selection plus captured dragging outside the palette', async () => {
    const { wrapper, pointer, hex } = picker()
    await wrapper.get('input[aria-label="色相"]').setValue('240')
    expect(hex()).toBe('#000000')
    await pointer('pointerdown', { button: 0, isPrimary: true, pointerId: 1, clientX: 210, clientY: 20 })
    expect(hex()).toBe('#0000ff')
    await pointer('pointermove', { pointerId: 1, clientX: 110, clientY: 80 })
    expect(hex()).toBe('#404080')
    await pointer('pointermove', { pointerId: 1, clientX: 250, clientY: 180 })
    expect(hex()).toBe('#000000')
    await pointer('pointermove', { pointerId: 1, clientX: 210, clientY: 20 })
    await pointer('pointerup', { pointerId: 1, clientX: 210, clientY: 20 })
    expect(hex()).toBe('#0000ff')
    await pointer('pointermove', { pointerId: 1, clientX: 10, clientY: 20 })
    expect(hex()).toBe('#0000ff')
  })

  it('ignores hovering, right clicks and secondary touch pointers', async () => {
    const { pointer, hex } = picker('#123456')
    await pointer('pointermove', { pointerId: 1, clientX: 210, clientY: 20 })
    await pointer('pointerdown', { button: 2, isPrimary: true, pointerId: 1, clientX: 210, clientY: 20 })
    await pointer('pointerdown', { button: 0, isPrimary: false, pointerId: 2, clientX: 210, clientY: 20 })
    expect(hex()).toBe('#123456')
  })

  it('syncs HEX, RGB and swatches with the palette while keeping hue at white', async () => {
    const { wrapper, pointer, hex } = picker()
    const hue = wrapper.get('input[aria-label="色相"]')
    await wrapper.get('input[aria-label="HEX 色值"]').setValue('#00ff00')
    expect(hue.element).toHaveProperty('value', '120')
    await wrapper.get('input[aria-label="蓝（B）"]').setValue('255')
    expect(hue.element).toHaveProperty('value', '180')
    await wrapper.get('button[aria-label="白色"]').trigger('click')
    expect(hue.element).toHaveProperty('value', '180')
    await hue.setValue('300')
    expect(hex()).toBe('#ffffff')
    await pointer('pointerdown', { button: 0, isPrimary: true, pointerId: 1, clientX: 210, clientY: 20 })
    expect(hex()).toBe('#ff00ff')
    await wrapper.get('input[aria-label="HEX 色值"]').setValue('#oops')
    await pointer('pointermove', { pointerId: 1, clientX: 10, clientY: 20 })
    await pointer('pointerup', { pointerId: 1, clientX: 10, clientY: 20 })
    expect(hex()).toBe('#ffffff')
    expect(wrapper.get('input[aria-label="HEX 色值"]').attributes('aria-invalid')).toBeUndefined()
  })

  it('supports keyboard changes on both axes and preserves precise cursor coordinates after RGB rounding', async () => {
    const { spectrum, pointer, hex } = picker('#0000ff')
    await spectrum.trigger('keydown', { key: 'ArrowLeft', shiftKey: true })
    expect(hex()).toBe('#1919ff')
    await spectrum.trigger('keydown', { key: 'ArrowDown', shiftKey: true })
    expect(hex()).toBe('#1717e6')
    await spectrum.trigger('keydown', { key: 'Home' })
    expect(hex()).toBe('#e6e6e6')
    await spectrum.trigger('keydown', { key: 'End' })
    expect(hex()).toBe('#0000e6')
    await pointer('pointerdown', { button: 0, isPrimary: true, pointerId: 1, clientX: 121, clientY: 63 })
    expect(spectrum.attributes('aria-valuetext')).toBe('饱和度 56%，明度 64%')
  })

  it('keeps the sampled color when validation changes the palette position before pointer release', async () => {
    const { wrapper, spectrum, pointer, hex } = picker('#ff0000')
    await wrapper.get('input[aria-label="HEX 色值"]').setValue('#oops')
    await pointer('pointerdown', { button: 0, isPrimary: true, pointerId: 1, clientX: 160, clientY: 50 })
    expect(hex()).toBe('#bf3030')
    Object.assign(spectrum.element, { getBoundingClientRect: () => new DOMRect(10, 60, 200, 120) })
    const updates = wrapper.getComponent(UiColorPicker).emitted('update:modelValue')!.length
    await pointer('pointerup', { pointerId: 1, clientX: 160, clientY: 50 })
    expect(hex()).toBe('#bf3030')
    expect(wrapper.getComponent(UiColorPicker).emitted('update:modelValue')).toHaveLength(updates)
  })

  it('uses the number fields to clamp channels and rounds fractional input to an RGB byte', async () => {
    const { wrapper, hex } = picker('#123456')
    await wrapper.get('input[aria-label="红（R）"]').setValue('999')
    expect(hex()).toBe('#ff3456')
    await wrapper.get('input[aria-label="绿（G）"]').setValue('-10')
    expect(hex()).toBe('#ff0056')
    await wrapper.get('input[aria-label="蓝（B）"]').setValue('20.6')
    expect(hex()).toBe('#ff0015')
  })
})
