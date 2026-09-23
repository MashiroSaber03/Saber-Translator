import { mount, flushPromises } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'
import UiSelect from '@/components/ui/UiSelect.vue'

describe('UiSelect teleport target', () => {
  it('opens inside a container created in the same render, including after remount', async () => {
    const Host = defineComponent({
      components: { UiSelect },
      template: `<div id="reader-select-host"><UiSelect teleport-to="#reader-select-host" :options="[{ value: 'chapter', label: '章节一' }]" /></div>`,
    })
    for (let run = 0; run < 2; run++) {
      const wrapper = mount(Host, { attachTo: document.body })
      try {
        await flushPromises()
        await wrapper.get('[role="combobox"]').trigger('click')
        await flushPromises()
        const option = document.querySelector('#reader-select-host [role="option"]') as HTMLElement
        expect(option?.textContent).toBe('章节一')
        let escapedParent = false
        wrapper.element.addEventListener('keydown', () => { escapedParent = true })
        await wrapper.get('[role="combobox"]').trigger('keydown', { key: 'Escape' })
        expect(escapedParent).toBe(false)
        expect(document.querySelector('[role="listbox"]')).toBeNull()
        await wrapper.get('[role="combobox"]').trigger('click')
        await flushPromises()
        ;(document.querySelector('#reader-select-host [role="option"]') as HTMLElement).click()
        await flushPromises()
        expect(wrapper.getComponent(UiSelect).emitted('change')?.[0]).toEqual(['chapter'])
        expect(document.querySelector('[role="listbox"]')).toBeNull()
      } finally {
        wrapper.unmount()
      }
    }
  })
})
