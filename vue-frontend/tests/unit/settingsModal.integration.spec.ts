import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'

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

vi.mock('@/components/settings/OcrSettings.vue', () => ({
  default: { name: 'OcrSettings', template: '<div>OcrSettings stub</div>' },
}))
vi.mock('@/components/settings/TranslationSettings.vue', () => ({
  default: { name: 'TranslationSettings', template: '<div>TranslationSettings stub</div>' },
}))
vi.mock('@/components/settings/DetectionSettings.vue', () => ({
  default: { name: 'DetectionSettings', template: '<div>DetectionSettings stub</div>' },
}))
vi.mock('@/components/settings/HqTranslationSettings.vue', () => ({
  default: { name: 'HqTranslationSettings', template: '<div>HqTranslationSettings stub</div>' },
}))
vi.mock('@/components/settings/ProofreadingSettings.vue', () => ({
  default: { name: 'ProofreadingSettings', template: '<div>ProofreadingSettings stub</div>' },
}))
vi.mock('@/components/settings/GlossarySettings.vue', () => ({
  default: {
    name: 'GlossarySettings',
    template: '<div>GlossarySettings stub</div>',
    methods: {
      validateSettings: () => ({ success: true }),
    },
  },
}))
vi.mock('@/components/settings/NonTranslateSettings.vue', () => ({
  default: {
    name: 'NonTranslateSettings',
    template: '<div>NonTranslateSettings stub</div>',
    methods: {
      validateSettings: () => ({ success: true }),
    },
  },
}))
vi.mock('@/components/settings/PromptLibrary.vue', () => ({
  default: { name: 'PromptLibrary', template: '<div>PromptLibrary stub</div>' },
}))
vi.mock('@/components/settings/PluginManager.vue', () => ({
  default: { name: 'PluginManager', template: '<div>PluginManager stub</div>' },
}))
vi.mock('@/components/settings/MoreSettings.vue', () => ({
  default: { name: 'MoreSettings', template: '<div>MoreSettings stub</div>' },
}))
vi.mock('@/components/settings/TextStyleDefaultsSettings.vue', () => ({
  default: {
    name: 'TextStyleDefaultsSettings',
    props: ['isOpen'],
    template: '<div>TextStyleDefaultsSettings stub</div>',
    methods: {
      saveDefaults: saveDefaultsMock,
    },
  },
}))

import SettingsModal from '@/components/settings/SettingsModal.vue'

const mountedWrappers: VueWrapper[] = []

function getOverlay(): HTMLDivElement {
  const overlay = document.body.querySelector('.modal-overlay')
  expect(overlay).toBeTruthy()
  return overlay as HTMLDivElement
}

function dispatchMouseEvent(target: Element, type: 'mousedown' | 'mouseup' | 'click') {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }))
}

afterEach(() => {
  while (mountedWrappers.length > 0) {
    mountedWrappers.pop()?.unmount()
  }
  document.body.innerHTML = ''
  document.body.style.overflow = ''
})

describe('SettingsModal integration', () => {
  it('propagates a complete overlay close through BaseModal and hides itself', async () => {
    const wrapper = mount(SettingsModal, {
      attachTo: document.body,
      props: {
        modelValue: true,
      },
    })
    mountedWrappers.push(wrapper)

    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')
    dispatchMouseEvent(overlay, 'click')

    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
    expect(wrapper.emitted('update:modelValue')?.some(([value]) => value === false)).toBe(true)
    expect(document.body.querySelector('.modal-overlay')).toBeNull()
  })
})
