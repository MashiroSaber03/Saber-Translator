import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import SettingsSidebar from './SettingsSidebar.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

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
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('defaults remember workflow mode to disabled before remote preferences load', async () => {
    const wrapper = mount(SettingsSidebar, {
      global: {
        plugins: [createPinia()],
        stubs: {
          UiCombobox: {
            name: 'UiCombobox',
            props: ['modelValue'],
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
          },
          ProductCollapsibleSection: {
            name: 'ProductCollapsibleSection',
            props: ['title', 'expanded'],
            template: '<section><slot /></section>',
          },
          PageSelectionModal: true,
        },
      },
    })

    const rememberToggle = wrapper.findAllComponents(UiCheckbox)
      .find(toggle => toggle.props('label') === '记住操作模式')
    expect(rememberToggle?.props('modelValue')).toBe(false)
  })

  it('maps parent shell colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/SettingsSidebar.vue'),
      'utf8'
    )

    expect(source).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })

  it('keeps sidebar shell hooks under the settings-sidebar owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/SettingsSidebar.vue'),
      'utf8'
    )

    expect(source).toContain('settings-sidebar__card')
    expect(source).toContain('settings-sidebar__title')
    expect(source).not.toMatch(/class="settings-card"/)
    expect(source).not.toMatch(/class="sidebar-title"/)
    expect(source).not.toMatch(/\.settings-card\b/)
    expect(source).not.toMatch(/\.sidebar-title\b/)
  })
})
