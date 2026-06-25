import { afterEach, describe, expect, it } from 'vitest'
import { enableAutoUnmount, mount } from '@vue/test-utils'
import CustomSelect from '@/components/common/CustomSelect.vue'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.body.innerHTML = ''
})

describe('CustomSelect accessibility contract', () => {
  it('exposes combobox and listbox semantics with keyboard open and close', async () => {
    const wrapper = mount(CustomSelect, {
      attachTo: document.body,
      props: {
        modelValue: 'b',
        options: [
          { label: 'Option A', value: 'a' },
          { label: 'Option B', value: 'b' },
        ],
      },
    })

    const trigger = wrapper.get('[role="combobox"]')
    expect(trigger.attributes('aria-expanded')).toBe('false')
    expect(trigger.attributes('aria-haspopup')).toBe('listbox')

    await trigger.trigger('keydown', { key: 'Enter' })

    const listboxId = trigger.attributes('aria-controls')
    expect(listboxId).toBeTruthy()
    const listbox = document.getElementById(listboxId || '')
    expect(listbox?.getAttribute('role')).toBe('listbox')

    const options = Array.from(document.body.querySelectorAll('[role="option"]'))
    expect(options).toHaveLength(2)
    expect(options[1].getAttribute('aria-selected')).toBe('true')
    expect(trigger.attributes('aria-expanded')).toBe('true')

    await trigger.trigger('keydown', { key: 'Escape' })

    expect(document.getElementById(listboxId || '')).toBeNull()
    expect(trigger.attributes('aria-expanded')).toBe('false')
  })
})
