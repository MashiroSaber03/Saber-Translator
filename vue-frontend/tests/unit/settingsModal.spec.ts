import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'

const { loadFromBackendMock, saveToBackendMock, settingsStoreState, translationSettingsSetupMock } =
  vi.hoisted(() => ({
    loadFromBackendMock: vi.fn(),
    saveToBackendMock: vi.fn(),
    translationSettingsSetupMock: vi.fn(),
    settingsStoreState: {
      settings: {
        textStyle: {
          fontSize: 16,
        },
      },
      textStyleDefaults: {
        fontSize: 16,
      },
      providerConfigs: {},
    },
  }))

vi.mock('@/stores/settings', async () => {
  const { reactive } = await import('vue')
  settingsStoreState.settings = reactive(settingsStoreState.settings)
  settingsStoreState.textStyleDefaults = reactive(settingsStoreState.textStyleDefaults)
  settingsStoreState.providerConfigs = reactive(settingsStoreState.providerConfigs)
  return {
    useSettingsStore: () => ({
      backendError: null,
      isBackendReady: true,
      settings: settingsStoreState.settings,
      textStyleDefaults: settingsStoreState.textStyleDefaults,
      providerConfigs: settingsStoreState.providerConfigs,
      loadFromBackend: loadFromBackendMock,
      saveToBackend: saveToBackendMock,
    }),
  }
})

vi.mock('@/stores/runtimeStore', () => ({
  useRuntimeStore: () => ({
    capabilities: { features: { plugins: true } },
  }),
}))

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: [
      'modelValue',
      'mobilePresentation',
      'headerVariant',
      'showCloseButton',
      'closeOnOverlay',
      'closeOnEsc',
    ],
    emits: ['update:modelValue', 'close'],
    setup(props, { slots }) {
      return () =>
        h(
          'div',
          {
            'data-header-variant': props.headerVariant,
            'data-mobile-presentation': props.mobilePresentation,
            'data-show-close-button': String(props.showCloseButton),
            'data-close-on-overlay': String(props.closeOnOverlay),
            'data-close-on-esc': String(props.closeOnEsc),
          },
          [
            h('div', { class: 'modal-body-stub' }, slots.default ? slots.default() : []),
            h('div', { class: 'modal-footer-stub' }, slots.footer ? slots.footer() : []),
          ]
        )
    },
  }),
}))

vi.mock('@/components/settings/OcrSettings.vue', () => ({
  default: defineComponent({
    name: 'OcrSettings',
    setup: () => () => h('div', 'OcrSettings stub'),
  }),
}))
vi.mock('@/components/settings/TranslationSettings.vue', () => ({
  default: defineComponent({
    name: 'TranslationSettings',
    setup: () => {
      translationSettingsSetupMock()
      return () => h('div', 'TranslationSettings stub')
    },
  }),
}))
vi.mock('@/components/settings/DetectionSettings.vue', () => ({
  default: defineComponent({
    name: 'DetectionSettings',
    setup: () => () => h('div', 'DetectionSettings stub'),
  }),
}))
vi.mock('@/components/settings/HqTranslationSettings.vue', () => ({
  default: defineComponent({
    name: 'HqTranslationSettings',
    setup: () => () => h('div', 'HqTranslationSettings stub'),
  }),
}))
vi.mock('@/components/settings/ProofreadingSettings.vue', () => ({
  default: defineComponent({
    name: 'ProofreadingSettings',
    setup: () => () => h('div', 'ProofreadingSettings stub'),
  }),
}))
vi.mock('@/components/settings/PromptLibrary.vue', () => ({
  default: defineComponent({
    name: 'PromptLibrary',
    setup: () => () => h('div', 'PromptLibrary stub'),
  }),
}))
vi.mock('@/components/settings/PluginManager.vue', () => ({
  default: defineComponent({
    name: 'PluginManager',
    setup: () => () => h('div', 'PluginManager stub'),
  }),
}))
vi.mock('@/components/settings/MoreSettings.vue', () => ({
  default: defineComponent({
    name: 'MoreSettings',
    setup: () => () => h('div', 'MoreSettings stub'),
  }),
}))
vi.mock('@/components/settings/TextStyleDefaultsSettings.vue', () => ({
  default: defineComponent({
    name: 'TextStyleDefaultsSettings',
    setup: () => () => h('div', 'TextStyleDefaultsSettings stub'),
  }),
}))

import SettingsModal from '@/components/settings/SettingsModal.vue'

enableAutoUnmount(afterEach)

describe('SettingsModal', () => {
  beforeEach(() => {
    loadFromBackendMock.mockReset()
    loadFromBackendMock.mockResolvedValue(true)
    saveToBackendMock.mockReset()
    saveToBackendMock.mockResolvedValue(true)
    translationSettingsSetupMock.mockReset()
    settingsStoreState.settings.textStyle.fontSize = 16
    settingsStoreState.textStyleDefaults.fontSize = 16
  })

  it('automatically saves a settings change without a save button', async () => {
    vi.useFakeTimers()
    try {
      const wrapper = mount(SettingsModal, {
        props: {
          modelValue: true,
        },
      })
      await flushPromises()
      settingsStoreState.textStyleDefaults.fontSize = 18
      await wrapper.vm.$nextTick()

      expect(wrapper.findAll('button').some(button => button.text().includes('保存设置'))).toBe(
        false
      )
      expect(wrapper.text()).toContain('修改后自动保存')

      await vi.advanceTimersByTimeAsync(450)
      await flushPromises()

      expect(saveToBackendMock).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps every close path disabled while the save transaction is pending', async () => {
    vi.useFakeTimers()
    let resolveSave!: (value: boolean) => void
    saveToBackendMock.mockReturnValueOnce(
      new Promise<boolean>(resolve => {
        resolveSave = resolve
      })
    )
    const wrapper = mount(SettingsModal, {
      props: { modelValue: true },
    })
    await flushPromises()
    settingsStoreState.settings.textStyle.fontSize = 18
    await wrapper.vm.$nextTick()
    await vi.advanceTimersByTimeAsync(450)
    await wrapper.vm.$nextTick()

    const buttons = wrapper.findAll('button')
    const doneButton = buttons.find(button => button.text() === '完成')
    expect(doneButton).toBeTruthy()

    expect(saveToBackendMock).toHaveBeenCalledOnce()
    expect(doneButton!.attributes('disabled')).toBeDefined()
    const modal = wrapper.get('[data-close-on-overlay]')
    expect(modal.attributes('data-show-close-button')).toBe('false')
    expect(modal.attributes('data-close-on-overlay')).toBe('false')
    expect(modal.attributes('data-close-on-esc')).toBe('false')
    await doneButton!.trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()

    resolveSave(true)
    await flushPromises()
    await doneButton!.trigger('click')
    await flushPromises()
    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
    vi.useRealTimers()
  })

  it('leaves the modal usable when automatic persistence rejects unexpectedly', async () => {
    vi.useFakeTimers()
    try {
      saveToBackendMock.mockRejectedValueOnce(new Error('写入失败'))
      const wrapper = mount(SettingsModal, {
        props: { modelValue: true },
      })
      await flushPromises()

      settingsStoreState.settings.textStyle.fontSize = 19
      await wrapper.vm.$nextTick()
      await vi.advanceTimersByTimeAsync(450)
      await flushPromises()

      const doneButton = wrapper.findAll('button').find(button => button.text() === '完成')
      expect(doneButton?.attributes('disabled')).toBeUndefined()
      expect(wrapper.text()).toContain('修改后自动保存')
    } finally {
      vi.useRealTimers()
    }
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

    expect(wrapper.get('[data-mobile-presentation]').attributes('data-mobile-presentation')).toBe(
      'fullscreen'
    )

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )
    expect(source).not.toContain('SettingsModal.global.styles.css')
  })

  it('uses a typed BaseModal brand header variant instead of raw header visual props', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    expect(wrapper.get('[data-header-variant]').attributes('data-header-variant')).toBe('brand')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )
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

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )
    expect(source).toContain('appearance="underline"')
  })

  it('does not keep a hard-coded active pane class from the old local tab shell', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )

    expect(source).not.toContain('class="settings-tab-pane active"')
    expect(source).not.toContain('.settings-tab-pane.active')
    expect(source).not.toContain('@keyframes settingsModalFadeIn')
  })

  it('keeps settings modal shell hooks under the modal owner contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )

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

  it('keeps global text defaults in their isolated draft and one save transaction', () => {
    const modalSource = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )
    const textDefaultsSource = readFileSync(
      resolve(process.cwd(), 'src/components/settings/TextStyleDefaultsSettings.vue'),
      'utf8'
    )

    expect(modalSource).toContain('<TextStyleDefaultsSettings />')
    expect(modalSource).toContain('settingsStore.saveToBackend()')
    expect(modalSource).not.toContain('save-request-id')
    expect(modalSource).not.toContain('save-complete')
    expect(modalSource).not.toContain('textStyleDefaultsRef')
    expect(modalSource).not.toContain('TextStyleDefaultsSettingsExposed')
    expect(modalSource).not.toContain('saveDefaults()')

    expect(textDefaultsSource).toContain('settingsStore.textStyleDefaults = normalized')
    expect(textDefaultsSource).not.toContain('saveRequestId')
    expect(textDefaultsSource).not.toContain('save-complete')
    expect(textDefaultsSource).not.toContain('defineExpose')
  })

  it('does not keep the old save and cancel snapshot machinery', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )

    expect(source).toContain('scheduleAutoSave()')
    expect(source).toContain('hasUnsavedChanges')
    expect(source).not.toContain('settingsSnapshot')
    expect(source).not.toContain('providerSnapshot')
    expect(source).not.toContain('保存设置')
  })

  it('mounts setting forms only after the latest backend load completes', async () => {
    let resolveLoad!: (value: boolean) => void
    loadFromBackendMock.mockReturnValueOnce(
      new Promise<boolean>(resolve => {
        resolveLoad = resolve
      })
    )

    const wrapper = mount(SettingsModal, {
      props: {
        initialTab: 'translate',
        modelValue: true,
      },
    })
    await flushPromises()

    expect(translationSettingsSetupMock).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('正在读取后端设置')

    resolveLoad(true)
    await flushPromises()

    expect(translationSettingsSetupMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('TranslationSettings stub')
  })

  it('disables all settings writes while backend settings are unavailable', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/settings/SettingsModal.vue'),
      'utf8'
    )

    expect(source).toContain('title="设置加载失败"')
    expect(source).toContain(':disabled="!settingsStore.isBackendReady || isSaving"')
    expect(source).toContain('settingsStore.backendError ||')
  })

  it('uses the product dialog action row for modal footer actions', () => {
    const wrapper = mount(SettingsModal, {
      props: {
        modelValue: true,
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('设置状态')
  })
})
