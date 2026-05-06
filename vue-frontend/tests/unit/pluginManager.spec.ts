import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

const {
  getPluginsMock,
  getPluginDefaultStatesMock,
  refreshPluginsMock,
  enablePluginMock,
  disablePluginMock,
  deletePluginMock,
  getPluginConfigSchemaMock,
  getPluginConfigMock,
  savePluginConfigMock,
  setPluginDefaultStateMock,
  toastSuccessMock,
  toastErrorMock,
  toastWarningMock,
} = vi.hoisted(() => ({
  getPluginsMock: vi.fn(),
  getPluginDefaultStatesMock: vi.fn(),
  refreshPluginsMock: vi.fn(),
  enablePluginMock: vi.fn(),
  disablePluginMock: vi.fn(),
  deletePluginMock: vi.fn(),
  getPluginConfigSchemaMock: vi.fn(),
  getPluginConfigMock: vi.fn(),
  savePluginConfigMock: vi.fn(),
  setPluginDefaultStateMock: vi.fn(),
  toastSuccessMock: vi.fn(),
  toastErrorMock: vi.fn(),
  toastWarningMock: vi.fn(),
}))

vi.mock('@/api/plugin', () => ({
  getPlugins: getPluginsMock,
  getPluginDefaultStates: getPluginDefaultStatesMock,
  refreshPlugins: refreshPluginsMock,
  enablePlugin: enablePluginMock,
  disablePlugin: disablePluginMock,
  deletePlugin: deletePluginMock,
  getPluginConfigSchema: getPluginConfigSchemaMock,
  getPluginConfig: getPluginConfigMock,
  savePluginConfig: savePluginConfigMock,
  setPluginDefaultState: setPluginDefaultStateMock,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
    warning: toastWarningMock,
  }),
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    name: 'CustomSelect',
    setup: () => () => h('div', 'CustomSelect stub'),
  }),
}))

import PluginManager from '@/components/settings/PluginManager.vue'

describe('PluginManager', () => {
  beforeEach(() => {
    getPluginsMock.mockReset()
    getPluginDefaultStatesMock.mockReset()
    refreshPluginsMock.mockReset()
    enablePluginMock.mockReset()
    disablePluginMock.mockReset()
    deletePluginMock.mockReset()
    getPluginConfigSchemaMock.mockReset()
    getPluginConfigMock.mockReset()
    savePluginConfigMock.mockReset()
    setPluginDefaultStateMock.mockReset()
    toastSuccessMock.mockReset()
    toastErrorMock.mockReset()
    toastWarningMock.mockReset()

    getPluginsMock.mockResolvedValue({
      success: true,
      plugins: [{
        id: 'plugin_one',
        display_name: 'Plugin One',
        description: 'desc',
        version: '1.0.0',
        enabled: false,
        default_enabled: false,
        has_config: false,
        supported_steps: ['ocr'],
        supported_modes: ['standard'],
      }],
    })
    getPluginDefaultStatesMock.mockResolvedValue({
      success: true,
      default_states: {
        plugin_one: false,
      },
    })
  })

  it('refreshes plugins from the refresh button and warns on partial success', async () => {
    refreshPluginsMock.mockResolvedValue({
      success: true,
      partial_success: true,
      plugins: [{
        id: 'plugin_two',
        display_name: 'Plugin Two',
        description: 'updated',
        version: '2.0.0',
        enabled: true,
        default_enabled: true,
        has_config: false,
        supported_steps: ['ocr'],
        supported_modes: ['standard'],
      }],
      default_states: {
        plugin_two: true,
      },
      summary: {
        added: 1,
        reloaded: 0,
        removed: 0,
        failed: 1,
      },
      failures: [{
        plugin_name: 'broken_plugin',
        error: 'bad import',
      }],
    })

    const wrapper = mount(PluginManager)
    await flushPromises()

    const refreshButton = wrapper.findAll('button').find(
      button => button.text().includes('刷新插件')
    )
    expect(refreshButton).toBeTruthy()

    await refreshButton!.trigger('click')
    await flushPromises()

    expect(refreshPluginsMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('Plugin Two')
    expect(toastWarningMock).toHaveBeenCalledWith(expect.stringContaining('部分插件刷新失败'))
  })
})
