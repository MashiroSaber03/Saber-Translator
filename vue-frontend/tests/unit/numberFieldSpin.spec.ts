import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import UiNumberField from '@/components/ui/UiNumberField.vue'

describe('decimal input with a separate arrow step', () => {
  it('accepts decimals and increments without snapping to integers', async () => {
    const wrapper = mount(UiNumberField, { props: { modelValue: 2.5, min: 0, step: 0.1, spinStep: 1 } })
    await wrapper.get('input').setValue('1.2')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([1.2])
    expect(wrapper.get('input').element.validity.stepMismatch).toBe(false)
    await wrapper.setProps({ modelValue: 2.5 })
    await wrapper.get('[aria-label="增加数值"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([3.5])
    await wrapper.get('input').trigger('keydown', { key: 'ArrowDown' })
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([1.5])
    await wrapper.setProps({ modelValue: 0.5 })
    await wrapper.get('[aria-label="减少数值"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([0])
    wrapper.unmount()
  })
  it('retains the existing step for other number fields', async () => {
    const wrapper = mount(UiNumberField, { props: { modelValue: 2.5, step: 0.1, controls: true } })
    await wrapper.get('[aria-label="增加数值"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual([2.6])
    wrapper.unmount()
  })

  it.each([
    ['ArrowUp', '增加数值', 101],
    ['ArrowDown', '减少数值', 99],
  ])('starts mixed percentages from 100 for %s and the matching button', async (key, label, expected) => {
    const wrapper = mount(UiNumberField, { props: {
      modelValue: null, nullable: true, emptyValue: 100, min: 10, max: 400, spinStep: 1,
    } })
    expect(wrapper.get('input').element.value).toBe('')
    expect(wrapper.emitted('change')).toBeUndefined()
    await wrapper.get('input').trigger('keydown', { key })
    expect(wrapper.emitted('change')?.at(-1)).toEqual([expected])
    await wrapper.get(`[aria-label="${label}"]`).trigger('click')
    expect(wrapper.emitted('change')?.at(-1)).toEqual([expected])
    wrapper.unmount()
  })

  it('uses zero for mixed spacing while retaining known values and bounds', async () => {
    const wrapper = mount(UiNumberField, { props: {
      modelValue: null, nullable: true, emptyValue: 0, min: -50, max: 200, spinStep: 1,
    } })
    await wrapper.get('input').trigger('keydown', { key: 'ArrowDown' })
    expect(wrapper.emitted('change')?.at(-1)).toEqual([-1])
    await wrapper.setProps({ modelValue: 13, emptyValue: 100, min: 10, max: 400 })
    await wrapper.get('input').trigger('keydown', { key: 'ArrowUp' })
    expect(wrapper.emitted('change')?.at(-1)).toEqual([14])
    await wrapper.setProps({ modelValue: 400 })
    expect(wrapper.get('[aria-label="增加数值"]').attributes('disabled')).toBeDefined()
    await wrapper.get('input').trigger('keydown', { key: 'ArrowUp' })
    expect(wrapper.emitted('change')?.at(-1)).toEqual([14])
    wrapper.unmount()
  })
})
