import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, watch } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'

const { saveToStorageMock, saveToBackendMock, saveDefaultsMock } = vi.hoisted(() => ({
  saveToStorageMock: vi.fn(),
  saveToBackendMock: vi.fn(),
  saveDefaultsMock: vi.fn(),
}))

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => ({
    saveToStorage: saveToStorageMock,
    saveToBackend: saveToBackendMock,
  }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'mobilePresentation', 'headerVariant'],
    emits: ['update:modelValue', 'close', 'open'],
    setup(props, { slots }) {
      return () => h('div', {
        'data-header-variant': props.headerVariant,
        'data-mobile-presentation': props.mobilePresentation,
      }, [
        h('div', { class: 'modal-body-stub' }, slots.default ? slots.default() : []),
        h('div', { class: 'modal-footer-stub' }, slots.footer ? slots.footer() : []),
      ])
    },
  }),
}))

vi.mock('@/components/settings/OcrSettings.vue', () => ({
  default: defineComponent({ name: 'OcrSettings', setup: () => () => h('div', 'OcrSettings stub') }),
}))
vi.mock('@/components/settings/TranslationSettings.vue', () => ({
  default: defineComponent({ name: 'TranslationSettings', setup: () => () => h('div', 'TranslationSettings stub') }),
}))
vi.mock('@/components/settings/DetectionSettings.vue', () => ({
  default: defineComponent({ name: 'DetectionSettings', setup: () => () => h('div', 'DetectionSettings stub') }),
}))
vi.mock('@/components/settings/HqTranslationSettings.vue', () => ({
  default: defineComponent({ name: 'HqTranslationSettings', setup: () => () => h('div', 'HqTranslationSettings stub') }),
}))
vi.mock('@/components/settings/ProofreadingSettings.vue', () => ({
  default: defineComponent({ name: 'ProofreadingSettings', setup: () => () => h('div', 'ProofreadingSettings stub') }),
}))
vi.mock('@/components/settings/PromptLibrary.vue', () => ({
  default: defineComponent({ name: 'PromptLibrary', setup: () => () => h('div', 'PromptLibrary stub') }),
}))
vi.mock('@/components/settings/PluginManager.vue', () => ({
  default: defineComponent({ name: 'PluginManager', setup: () => () => h('div', 'PluginManager stub') }),
}))
vi.mock('@/components/settings/MoreSettings.vue', () => ({
  default: defineComponent({ name: 'MoreSettings', setup: () => () => h('div', 'MoreSettings stub') }),
}))
vi.mock('@/components/settings/TextStyleDefaultsSettings.vue', () => ({
  default: defineComponent({
    name: 'TextStyleDefaultsSettings',
    props: {
      isOpen: Boolean,
      saveRequestId: {
        type: Number,
        default: 0,
      },
    },
    emits: ['save-complete'],
    setup(props, { emit }) {
      watch(() => props.saveRequestId, async (requestId, previousRequestId) => {
        if (requestId === previousRequestId || requestId === 0) return
        emit('save-complete', await saveDefaultsMock())
      })
      return () => h('div', 'TextStyleDefaultsSettings stub')
    },
  }),
}))

import SettingsModal from '@/components/settings/SettingsModal.vue'

describe('SettingsModal', () => {
  beforeEach(() => {
    saveToStorageMock.mockReset()
    saveToBackendMock.mockReset()
    saveDefaultsMock.mockReset()

    saveToBackendMock.mockResolvedValue(true)
    saveDefaultsMock.mockResolvedValue({ success: true, changed: true })
  })

  it('emits textDefaultsChanged when text default values are saved', async () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    const saveButton = wrapper.findAll('button').find(button => button.text().includes('保存设置'))
    expect(saveButton).toBeTruthy()

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      await saveButton!.trigger('click')
      await flushPromises()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(saveDefaultsMock).toHaveBeenCalledTimes(1)
    expect(saveToStorageMock).toHaveBeenCalledTimes(1)
    expect(saveToBackendMock).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('save')?.[0]?.[0]).toEqual({ textDefaultsChanged: true })
  })

  it('does not render glossary and non-translate tabs', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    const tabTexts = wrapper.findAll('[role="tab"]').map(tab => tab.text())
    expect(tabTexts).not.toContain('术语表')
    expect(tabTexts).not.toContain('禁翻表')
    expect(wrapper.find('.settings-tabs').exists()).toBe(false)
    expect(wrapper.find('.product-segmented-tabs').exists()).toBe(true)
  })

  it('uses BaseModal mobile presentation instead of a single-use global stylesheet', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    expect(wrapper.get('[data-mobile-presentation]').attributes('data-mobile-presentation')).toBe('fullscreen')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')
    expect(source).not.toContain('SettingsModal.global.styles.css')
  })

  it('uses a typed BaseModal brand header variant instead of raw header visual props', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    expect(wrapper.get('[data-header-variant]').attributes('data-header-variant')).toBe('brand')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')
    expect(source).toContain('header-variant="brand"')
    expect(source).not.toMatch(/header-(background|color)=/)
    expect(source).not.toMatch(/title-(color|font-size)=/)
    expect(source).not.toMatch(/close-(color|font-size|hover-color|hover-background)=/)
  })

  it('uses the product underline tab appearance for modal navigation', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    expect(wrapper.getComponent(ProductSegmentedTabs).props('appearance')).toBe('underline')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')
    expect(source).toContain('appearance="underline"')
  })

  it('does not keep a hard-coded active pane class from the old local tab shell', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')

    expect(source).not.toContain('class="settings-tab-pane active"')
    expect(source).not.toContain('.settings-tab-pane.active')
    expect(source).not.toContain('@keyframes settingsModalFadeIn')
  })

  it('keeps settings modal shell hooks under the modal owner contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')

    expect(source).toContain('class="settings-modal__title"')
    expect(source).toContain('class="settings-modal__tabs"')
    expect(source).toContain('class="settings-modal__tab-content"')
    expect(source).toContain('class="settings-modal__tab-pane"')

    expect(source).not.toContain('class="settings-modal-title"')
    expect(source).not.toContain('class="settings-modal-tabs"')
    expect(source).not.toContain('class="settings-tab-content"')
    expect(source).not.toContain('class="settings-tab-pane"')
    expect(source).not.toContain('.settings-modal-title')
    expect(source).not.toContain('.settings-modal-tabs')
    expect(source).not.toContain('.settings-tab-content')
    expect(source).not.toContain('.settings-tab-pane')
  })

  it('requests text default saves through a typed child result event', () => {
    const modalSource = readFileSync(resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'), 'utf8')
    const textDefaultsSource = readFileSync(
      resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'),
      'utf8',
    )

    expect(modalSource).toContain(':save-request-id="textDefaultsSaveRequestId"')
    expect(modalSource).toContain('@save-complete="handleTextDefaultsSaveComplete"')
    expect(modalSource).not.toContain('textStyleDefaultsRef')
    expect(modalSource).not.toContain('TextStyleDefaultsSettingsExposed')
    expect(modalSource).not.toContain('saveDefaults()')

    expect(textDefaultsSource).toContain('saveRequestId')
    expect(textDefaultsSource).toContain('save-complete')
    expect(textDefaultsSource).not.toContain('defineExpose')
  })

  it('uses the product dialog action row for modal footer actions', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('应用设置操作')
  })
})
