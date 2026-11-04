import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { ref } from 'vue'

describe('decimal input with a separate arrow step', () => {
  it('shows the consumer-rounded value after finishing a focused input', async () => {
    const value = ref(1)
    const wrapper = mount({
      components: { UiNumberField }, setup: () => ({ value }),
      template: '<UiNumberField :model-value="value" @update:model-value="value = Math.round($event)" />',
    })
    const input = wrapper.get('input')
    await input.trigger('focus')
    await input.setValue('1.2')
    expect(input.element.value).toBe('1.2')
    expect(value.value).toBe(1)
    await input.trigger('blur')
    expect(input.element.value).toBe('1')
    wrapper.unmount()
  })
  it('keeps partial bounded input while typing 150 and validates only on completion', async () => {
    const value = ref<number | null>(100)
    const wrapper = mount({
      components: { UiNumberField }, setup: () => ({ value }),
      template: '<UiNumberField v-model="value" :min="10" :max="400" />',
    })
    const input = wrapper.get('input')
    await input.trigger('focus')
    await input.setValue('1')
    expect(input.element.value).toBe('1')
    expect(value.value).toBe(100)
    await input.setValue('15')
    expect(value.value).toBe(15)
    await input.setValue('150')
    await input.trigger('blur')
    expect(input.element.value).toBe('150')
    expect(value.value).toBe(150)
    await input.trigger('focus')
    await input.setValue('1050')
    expect(input.element.value).toBe('1050')
    expect(value.value).toBe(150)
    await input.trigger('keydown', { key: 'Enter' })
    expect(input.element.value).toBe('400')
    expect(value.value).toBe(400)
    await input.setValue('1')
    await input.trigger('blur')
    expect(input.element.value).toBe('10')
    expect(value.value).toBe(10)
    wrapper.unmount()
  })
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
