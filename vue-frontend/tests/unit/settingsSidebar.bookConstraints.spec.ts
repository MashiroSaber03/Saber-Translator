import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

vi.mock('@/api/config', () => ({
  getFontList: async () => ({ fonts: [] }),
  uploadFont: async () => ({ success: true }),
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'options'],
    emits: ['change'],
    setup(props) {
      return () => h('select', { value: props.modelValue }, (props.options || []).map((option: any) => h('option', { value: option.value }, option.label)))
    },
  }),
}))

vi.mock('@/components/common/CollapsiblePanel.vue', () => ({
  default: defineComponent({
    props: ['title'],
    setup(props, { slots }) {
      return () => h('section', [h('h3', props.title), slots.default?.()])
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'

describe('SettingsSidebar book constraints entrypoints', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('shows disabled buttons when book constraints are unavailable', () => {
    const wrapper = mount(SettingsSidebar)
    const buttons = wrapper.findAll('button')
    const glossaryButton = buttons.find((button) => button.text().includes('术语表'))
    const nonTranslateButton = buttons.find((button) => button.text().includes('禁翻表'))

    expect(glossaryButton?.attributes('disabled')).toBeDefined()
    expect(nonTranslateButton?.attributes('disabled')).toBeDefined()
  })

  it('enables buttons and emits open events when in bookshelf mode', async () => {
    const constraintStore = useBookTranslationConstraintsStore()
    constraintStore.loadBookConstraints('book-1', {
      glossary: { enabled: false, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })

    const wrapper = mount(SettingsSidebar)
    const buttons = wrapper.findAll('button')
    const glossaryButton = buttons.find((button) => button.text().includes('术语表'))
    const nonTranslateButton = buttons.find((button) => button.text().includes('禁翻表'))

    expect(glossaryButton?.attributes('disabled')).toBeUndefined()
    expect(nonTranslateButton?.attributes('disabled')).toBeUndefined()

    await glossaryButton?.trigger('click')
    await nonTranslateButton?.trigger('click')

    expect(wrapper.emitted('openGlossary')).toBeTruthy()
    expect(wrapper.emitted('openNonTranslate')).toBeTruthy()
  })
})
