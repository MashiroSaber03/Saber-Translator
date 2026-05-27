import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SettingsSidebar from './SettingsSidebar.vue'

const apiMocks = vi.hoisted(() => ({
  getFontList: vi.fn(),
  getTranslateWorkflowPreferences: vi.fn(),
  saveTranslateWorkflowPreferences: vi.fn(),
  uploadFont: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getFontList: apiMocks.getFontList,
  getTranslateWorkflowPreferences: apiMocks.getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences: apiMocks.saveTranslateWorkflowPreferences,
  uploadFont: apiMocks.uploadFont,
}))

describe('SettingsSidebar defaults', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    apiMocks.getFontList.mockResolvedValue({ fonts: [] })
    apiMocks.getTranslateWorkflowPreferences.mockRejectedValue(new Error('offline'))
    apiMocks.saveTranslateWorkflowPreferences.mockResolvedValue({ success: true })
    apiMocks.uploadFont.mockResolvedValue({ success: true, fontPath: 'fonts/custom.ttf' })
  })

  it('defaults remember workflow mode to enabled before remote preferences load', async () => {
    const wrapper = mount(SettingsSidebar, {
      global: {
        plugins: [createPinia()],
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            props: ['modelValue'],
            template: '<div class="custom-select-stub">{{ modelValue }}</div>',
          },
          CollapsiblePanel: {
            name: 'CollapsiblePanel',
            props: ['title'],
            template: '<section><slot /></section>',
          },
          PageSelectionModal: true,
        },
      },
    })

    const checkbox = wrapper.get('#rememberWorkflowModeCheckbox')
    expect((checkbox.element as HTMLInputElement).checked).toBe(true)
  })
})
