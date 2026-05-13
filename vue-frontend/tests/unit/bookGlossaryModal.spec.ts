import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'

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

  it('loads and saves the auto extract toggle with glossary constraints', async () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
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

    await checkboxes[1]!.setValue(false)
    await wrapper.find('.btn-primary').trigger('click')

    expect(saveBookConstraintsMock).toHaveBeenCalledWith(expect.objectContaining({
      glossary: expect.objectContaining({
        autoExtractEnabled: false,
      }),
    }))
  })
})
