import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'

const { saveBookConstraintsMock, showToastMock } = vi.hoisted(() => ({
  saveBookConstraintsMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['modelValue'],
    emits: ['update:modelValue', 'close'],
    setup(_props, { slots }) {
      return () => h('div', [
        slots.default?.(),
        slots.footer?.(),
      ])
    },
  }),
}))

vi.mock('@/components/settings/shared/TranslationConstraintTable.vue', () => ({
  default: defineComponent({
    setup() {
      return () => h('div', { class: 'translation-constraint-table-stub' })
    },
  }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import BookGlossaryModal from '@/components/translate/BookGlossaryModal.vue'

describe('BookGlossaryModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    saveBookConstraintsMock.mockReset()
    showToastMock.mockReset()
  })

  it('loads and saves the auto extract settings with glossary constraints', async () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    })
    store.saveBookConstraints = saveBookConstraintsMock.mockResolvedValue(true)

    const wrapper = mount(BookGlossaryModal, {
      props: {
        modelValue: true,
      },
    })

    const checkboxes = wrapper.findAll('input[type="checkbox"]')
    expect(checkboxes).toHaveLength(2)
    expect((checkboxes[1]!.element as HTMLInputElement).checked).toBe(true)
    const promptTextarea = wrapper.find('textarea.auto-glossary-prompt')
    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe(DEFAULT_AUTO_GLOSSARY_PROMPT)

    await checkboxes[1]!.setValue(false)
    await promptTextarea.setValue('自定义提词提示词')
    await wrapper.find('.btn-primary').trigger('click')

    expect(saveBookConstraintsMock).toHaveBeenCalledWith(expect.objectContaining({
      glossary: expect.objectContaining({
        autoExtractEnabled: false,
        autoExtractPrompt: '自定义提词提示词',
      }),
    }))
  })

  it('resets the auto glossary prompt back to the default prompt', async () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: '自定义提示词',
        entries: [],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    })

    const wrapper = mount(BookGlossaryModal, {
      props: {
        modelValue: true,
      },
    })

    const promptTextarea = wrapper.find('textarea.auto-glossary-prompt')
    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe('自定义提示词')

    await wrapper.find('.reset-auto-glossary-prompt-btn').trigger('click')

    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe(DEFAULT_AUTO_GLOSSARY_PROMPT)
  })
})
