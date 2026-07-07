import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { enableAutoUnmount, mount } from '@vue/test-utils'
import UiCombobox from '@/components/ui/UiCombobox.vue'

enableAutoUnmount(afterEach)

afterEach(() => {
  document.body.innerHTML = ''
})

describe('UiCombobox accessibility contract', () => {
  it('exposes combobox and listbox semantics with keyboard open and close', async () => {
    const wrapper = mount(UiCombobox, {
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

  it('uses the shared icon primitive for its disclosure arrow', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/ui/UiCombobox.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/<svg[\s>]/)
    expect(source).toContain('<UiIcon name="chevron-down"')
  })

  it('binds public id and accessible name props to the combobox trigger', () => {
    const wrapper = mount(UiCombobox, {
      props: {
        inputId: 'settingsSourceLanguage',
        ariaLabel: '源语言',
        modelValue: 'japanese',
        options: [
          { label: '日语', value: 'japanese' },
        ],
      },
    })

    const trigger = wrapper.get('[role="combobox"]')
    expect(trigger.attributes('id')).toBe('settingsSourceLanguage')
    expect(trigger.attributes('aria-label')).toBe('源语言')
  })

  it('uses owner-prefixed state classes instead of bare state hooks', async () => {
    const wrapper = mount(UiCombobox, {
      attachTo: document.body,
      props: {
        modelValue: 'b',
        options: [
          { label: 'Option A', value: 'a' },
          { label: 'Option B', value: 'b' },
        ],
      },
    })

    await wrapper.get('[role="combobox"]').trigger('click')

    expect(wrapper.classes()).toContain('ui-combobox--open')
    expect(wrapper.classes()).not.toContain('open')
    expect(document.body.querySelector('.ui-combobox-option--selected')).toBeTruthy()
    expect(document.body.querySelector('.selected')).toBeNull()

    await wrapper.setProps({ disabled: true })
    expect(wrapper.classes()).toContain('ui-combobox--disabled')
    expect(wrapper.classes()).not.toContain('disabled')
  })
})
