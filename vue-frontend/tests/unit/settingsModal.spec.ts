import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

const { saveToStorageMock, saveToBackendMock, saveDefaultsMock } = vi.hoisted(() => ({
  saveToStorageMock: vi.fn(),
  saveToBackendMock: vi.fn(),
  saveDefaultsMock: vi.fn(),
}))

vi.mock('@/stores/settingsStore', () => ({
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
    props: ['modelValue'],
    emits: ['update:modelValue', 'close', 'open'],
    setup(_props, { slots }) {
      return () => h('div', [
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
    setup(_props, { expose }) {
      expose({
        saveDefaults: saveDefaultsMock,
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

    await saveButton!.trigger('click')
    await flushPromises()

    expect(saveDefaultsMock).toHaveBeenCalledTimes(1)
    expect(saveToStorageMock).toHaveBeenCalledTimes(1)
    expect(saveToBackendMock).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('save')?.[0]?.[0]).toEqual({ textDefaultsChanged: true })
  })
})
