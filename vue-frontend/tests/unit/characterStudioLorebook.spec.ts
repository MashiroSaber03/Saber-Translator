import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LorebookTreeEditor from '@/components/insight/studio/LorebookTreeEditor.vue'

describe('LorebookTreeEditor import flow', () => {
  it('emits selected worldbook file to parent', async () => {
    const wrapper = mount(LorebookTreeEditor, {
      props: {
        entries: [],
        importing: false,
      },
    })

    const input = wrapper.find('input[type="file"]')
    expect(input.exists()).toBe(true)

    const file = new File(['{"entries":[]}'], 'worldbook.json', { type: 'application/json' })
    Object.defineProperty(input.element, 'files', {
      value: [file],
      configurable: true,
    })
    await input.trigger('change')

    const emitted = wrapper.emitted('import-worldbook')
    expect(emitted).toBeTruthy()
    expect(emitted?.[0]?.[0]).toBe(file)
  })
})
