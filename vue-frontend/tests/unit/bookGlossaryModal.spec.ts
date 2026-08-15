import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const { saveBookConstraintsMock, showToastMock } = vi.hoisted(() => ({
  saveBookConstraintsMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    name: 'BaseModalStub',
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
import BookNonTranslateModal from '@/components/translate/BookNonTranslateModal.vue'

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
      nonTranslate: {
        enabled: false,
        entries: [],
      },
    }, 1)
    store.saveBookConstraints = saveBookConstraintsMock.mockResolvedValue(undefined)

    const wrapper = mount(BookGlossaryModal, {
      props: {
        modelValue: true,
      },
    })

    const checkboxes = wrapper.findAll('input[type="checkbox"]')
    expect(checkboxes).toHaveLength(2)
    expect((checkboxes[1]!.element as HTMLInputElement).checked).toBe(true)
    const promptTextarea = wrapper.find('#autoGlossaryPrompt')
    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe(DEFAULT_AUTO_GLOSSARY_PROMPT)

    await checkboxes[1]!.setValue(false)
    await promptTextarea.setValue('自定义提词提示词')
    await wrapper.find('[data-testid="save-book-glossary-button"]').trigger('click')

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
      nonTranslate: {
        enabled: false,
        entries: [],
      },
    }, 1)

    const wrapper = mount(BookGlossaryModal, {
      props: {
        modelValue: true,
      },
    })

    const promptTextarea = wrapper.find('#autoGlossaryPrompt')
    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe('自定义提示词')

    await wrapper.find('.reset-auto-glossary-prompt-btn').trigger('click')

    expect((promptTextarea.element as HTMLTextAreaElement).value).toBe(DEFAULT_AUTO_GLOSSARY_PROMPT)
  })

  it('uses product dialog primitives for constraint modal forms and footers', () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      nonTranslate: {
        enabled: true,
        entries: [],
      },
    }, 1)

    const glossaryWrapper = mount(BookGlossaryModal, {
      props: {
        modelValue: true,
      },
    })
    const nonTranslateWrapper = mount(BookNonTranslateModal, {
      props: {
        modelValue: true,
      },
    })

    expect(glossaryWrapper.findAllComponents(ProductStatusBanner).length).toBeGreaterThanOrEqual(2)
    expect(nonTranslateWrapper.findAllComponents(ProductStatusBanner).length).toBeGreaterThanOrEqual(1)
    expect(glossaryWrapper.getComponent(UiField).props('label')).toBe('自动术语提取提示词')

    const glossaryActionRows = glossaryWrapper.findAllComponents(ProductActionRow)
    const promptActionRow = glossaryActionRows.find(row => row.props('ariaLabel') === '自动术语提取提示词操作')
    const glossaryFooterRow = glossaryActionRows.find(row => row.props('ariaLabel') === '术语表操作')
    expect(promptActionRow?.props('variant')).toBe('default')
    expect(glossaryFooterRow?.props('variant')).toBe('dialog')

    const nonTranslateFooterRow = nonTranslateWrapper
      .findAllComponents(ProductActionRow)
      .find(row => row.props('ariaLabel') === '禁翻表操作')
    expect(nonTranslateFooterRow?.props('variant')).toBe('dialog')
    expect(glossaryWrapper.find('.ui-checkbox-label').exists()).toBe(false)
    expect(nonTranslateWrapper.find('.ui-checkbox-label').exists()).toBe(false)
    expect(glossaryWrapper.find('.book-glossary-modal__prompt-field').exists()).toBe(false)
    expect(glossaryWrapper.getComponent(UiTextarea).props('variant')).toBe('panel')

    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/BookGlossaryModal.vue'), 'utf8')
    expect(source).not.toContain('class="auto-glossary-prompt"')
    expect(source).not.toContain('.auto-glossary-prompt')
  })

  it('uses the shared clone helper for book constraint modal drafts', () => {
    for (const file of [
      'src/components/translate/BookGlossaryModal.vue',
      'src/components/translate/BookNonTranslateModal.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).toContain("import { deepClone } from '@/utils/deepClone'")
      expect(source, file).not.toContain('JSON.parse(JSON.stringify(')
    }
  })

  it('forwards one close update without mirroring modal state locally', () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: false,
        autoExtractEnabled: false,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      nonTranslate: { enabled: false, entries: [] },
    }, 1)

    for (const component of [BookGlossaryModal, BookNonTranslateModal]) {
      const wrapper = mount(component, { props: { modelValue: true } })
      wrapper.getComponent({ name: 'BaseModalStub' }).vm.$emit('close')
      expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
    }
  })

  it('shows the backend constraint error instead of replacing it with a generic message', async () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: false,
        autoExtractEnabled: false,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      nonTranslate: { enabled: false, entries: [] },
    }, 1)
    store.saveBookConstraints = saveBookConstraintsMock.mockRejectedValue(
      new Error('约束文档已被其他请求更新'),
    )

    const wrapper = mount(BookGlossaryModal, { props: { modelValue: true } })
    await wrapper.find('[data-testid="save-book-glossary-button"]').trigger('click')

    expect(showToastMock).toHaveBeenCalledWith('约束文档已被其他请求更新', 'error')
    expect(wrapper.emitted('saved')).toBeUndefined()
  })
})
