import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { enableAutoUnmount, mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { UiSelectOption } from '@/components/ui/selectTypes'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'

vi.mock('@/api/v2/settings', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  listV2Fonts: async () => [],
  uploadV2Font: async () => ({ id: 'font-uploaded', assetUrl: '/api/v2/assets/font' }),
}))

vi.mock('@/components/ui/UiCombobox.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: [String, Number] as PropType<string | number | undefined>,
        default: undefined,
      },
      options: {
        type: Array as PropType<UiSelectOption[]>,
        default: () => [],
      },
    },
    emits: ['change'],
    setup(props) {
      return () => h('select', { value: props.modelValue }, props.options.map((option) => h('option', { value: option.value }, option.label)))
    },
  }),
}))

vi.mock('@/components/product/ProductCollapsibleSection.vue', () => ({
  default: defineComponent({
    props: {
      title: {
        type: String,
        default: '',
      },
    },
    setup(props, { slots }) {
      return () => h('section', [h('h3', props.title), slots.default?.()])
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import BookConstraintSection from '@/components/translate/settings-sidebar/BookConstraintSection.vue'
import NavigationButtons from '@/components/translate/settings-sidebar/NavigationButtons.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'

describe('SettingsSidebar book constraints entrypoints', () => {
  enableAutoUnmount(afterEach)

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
      glossary: {
        enabled: false,
        autoExtractEnabled: false,
        autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
        entries: [],
      },
      nonTranslate: { enabled: false, entries: [] },
    }, 1)

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

  it('uses product action rows and semantic tokens for settings sidebar action sections', () => {
    const bookWrapper = mount(BookConstraintSection, {
      props: {
        canUseBookConstraints: true,
      },
    })
    const navigationWrapper = mount(NavigationButtons, {
      props: {
        canGoNext: true,
        canGoPrevious: true,
      },
    })

    expect(bookWrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('书籍约束操作')
    expect(bookWrapper.getComponent(ProductFormSection).text()).toContain('书籍约束')
    expect(bookWrapper.find('.book-constraints-panel').exists()).toBe(false)
    expect(bookWrapper.find('.book-constraints-title').exists()).toBe(false)
    expect(navigationWrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('图片导航')
    expect([
      ...bookWrapper.findAllComponents(UiButton),
      ...navigationWrapper.findAllComponents(UiButton),
    ].map(button => button.props('variant'))).toEqual(['secondary', 'secondary', 'secondary', 'secondary'])
    expect(bookWrapper.find('.secondary-button').exists()).toBe(false)
    expect(navigationWrapper.find('.navigation-button').exists()).toBe(false)

    for (const file of [
      'src/components/translate/settings-sidebar/BookConstraintSection.vue',
      'src/components/translate/settings-sidebar/NavigationButtons.vue',
    ]) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
      expect(source).not.toContain('variant="toolbar"')
    }
  })

  it('does not keep legacy DOM id hooks for settings navigation buttons', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/NavigationButtons.vue'),
      'utf8',
    )

    expect(source).not.toContain('id="prevImageButton"')
    expect(source).not.toContain('id="nextImageButton"')
  })

  it('keeps book constraint and navigation hooks under their section owners', () => {
    const bookSource = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/BookConstraintSection.vue'),
      'utf8',
    )
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/settings-sidebar/NavigationButtons.vue'),
      'utf8',
    )

    expect(bookSource).toContain('book-constraint-section')
    expect(bookSource).toContain('book-constraint-section__actions')
    expect(bookSource).not.toContain('settings-sidebar__book-constraints')
    expect(source).toContain('settings-navigation-buttons')
    expect(source).toContain('settings-navigation-buttons__action')
    expect(source).not.toContain('settings-sidebar__navigation-actions')
    expect(source).not.toContain('settings-sidebar__navigation-action')
    expect(source).not.toMatch(/settings-sidebar-navigation-actions/)
    expect(source).not.toMatch(/settings-sidebar-navigation-action/)
  })

  it('uses product feedback for unavailable book constraints', () => {
    const wrapper = mount(BookConstraintSection, {
      props: {
        canUseBookConstraints: false,
      },
    })

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('neutral')
    expect(banner.text()).toContain('仅书架模式可用')
    expect(wrapper.find('.book-constraints-disabled-note').exists()).toBe(false)
  })
})
