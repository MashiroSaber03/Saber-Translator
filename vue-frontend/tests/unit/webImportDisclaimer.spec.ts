import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getV2SettingsMock,
  saveV2SettingsTransactionMock,
} = vi.hoisted(() => ({
  getV2SettingsMock: vi.fn(),
  saveV2SettingsTransactionMock: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => {
  return {
    getV2Settings: getV2SettingsMock,
    saveV2SettingsTransaction: saveV2SettingsTransactionMock,
  }
})

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import WebImportDisclaimer from '@/components/translate/WebImportDisclaimer.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import { useWebImportStore } from '@/stores/webImportStore'

const BaseModalStub = defineComponent({
  props: {
    modelValue: Boolean,
    backdrop: String,
    overlayLayer: String,
    backdropEffect: String,
  },
  setup(props, { slots }) {
    return () => props.modelValue
      ? h('section', {
          role: 'dialog',
          'data-backdrop': props.backdrop,
          'data-overlay-layer': props.overlayLayer,
          'data-backdrop-effect': props.backdropEffect,
        }, slots.default?.())
      : null
  },
})

describe('WebImportDisclaimer', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    getV2SettingsMock.mockReset()
    saveV2SettingsTransactionMock.mockReset()
    getV2SettingsMock.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
  })

  it('uses product form and action primitives while preserving typed confirmation', async () => {
    const store = useWebImportStore()
    store.disclaimerVisible = true

    const wrapper = mount(WebImportDisclaimer, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('网页导入免责声明操作')

    const confirmationField = wrapper.getComponent(UiField)
    expect(confirmationField.props('label')).toBe('确认输入')
    expect(confirmationField.props('controlId')).toBe('webImportDisclaimerConfirmation')

    const buttons = wrapper.findAllComponents(UiButton)
    expect(buttons.map(button => button.props('variant'))).toEqual(['secondary', 'primary'])

    const disabledConfirmButton = buttons.find(button => button.text().includes('确认并继续'))
    expect(disabledConfirmButton?.props('disabled')).toBe(true)

    await wrapper.get('#webImportDisclaimerConfirmation').setValue('我已阅读并同意')
    await flushPromises()

    const enabledConfirmButton = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text().includes('确认并继续'))
    expect(enabledConfirmButton?.props('disabled')).toBe(false)

    await enabledConfirmButton?.trigger('click')
    await flushPromises()

    expect(store.disclaimerAccepted).toBe(true)
    expect(store.modalVisible).toBe(true)
  })

  it('maps owner colors through semantic tokens instead of raw color values', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/WebImportDisclaimer.vue'), 'utf8')
    const style = source.match(/<style scoped>([\s\S]*?)<\/style>/)?.[1] ?? ''

    expect(style).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    for (const staleAlias of [
      '--web-import-disclaimer-header-background-end',
      '--web-import-disclaimer-section-background',
      '--web-import-disclaimer-confirmation-background-start',
      '--web-import-disclaimer-scrollbar-thumb-hover',
    ]) {
      expect(style).not.toContain(staleAlias)
    }
    expect(style).toContain('var(--color-status-warning-surface)')
    expect(style).toContain('var(--color-surface-raised)')
  })

  it('uses typed BaseModal overlay contracts instead of a global stylesheet', () => {
    const store = useWebImportStore()
    store.disclaimerVisible = true

    const wrapper = mount(WebImportDisclaimer, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.attributes('data-backdrop')).toBe('strong')
    expect(dialog.attributes('data-overlay-layer')).toBe('popover')
    expect(dialog.attributes('data-backdrop-effect')).toBe('blur-sm')

    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/WebImportDisclaimer.vue'), 'utf8')
    expect(source).not.toContain('WebImportDisclaimer.global.styles.css')
  })

  it('keeps disclaimer presentation hooks under the disclaimer owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/WebImportDisclaimer.vue'), 'utf8')

    for (const currentHook of [
      'web-import-disclaimer',
      'web-import-disclaimer__header',
      'web-import-disclaimer__warning-icon',
      'web-import-disclaimer__title',
      'web-import-disclaimer__content',
      'web-import-disclaimer__terms-heading',
      'web-import-disclaimer__section',
      'web-import-disclaimer__section-title',
      'web-import-disclaimer__paragraph',
      'web-import-disclaimer__list',
      'web-import-disclaimer__list-item',
      'web-import-disclaimer__emphasis',
      'web-import-disclaimer__confirmation',
      'web-import-disclaimer__required-code',
      'web-import-disclaimer__footer',
      'web-import-disclaimer__action',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'web-import-disclaimer-shell',
      'disclaimer-header',
      'warning-icon',
      'disclaimer-title',
      'disclaimer-content',
      'disclaimer-text',
      'section',
      'warning-section',
      'confirmation-area',
      'confirmation-prompt',
      'required-text',
      'disclaimer-footer',
      'disclaimer-action',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }

    expect(source).not.toMatch(/\.web-import-disclaimer__text\s+h3/)
    expect(source).not.toMatch(/\.web-import-disclaimer__section\s+(h4|p|ul|li|strong)/)
  })
})
