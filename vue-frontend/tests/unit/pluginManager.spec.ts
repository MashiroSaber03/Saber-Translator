/* eslint-disable vue/one-component-per-file */
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
  exportPluginMock,
  importPluginMock,
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
  exportPluginMock: vi.fn(),
  importPluginMock: vi.fn(),
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
  exportPlugin: exportPluginMock,
  importPlugin: importPluginMock,
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

vi.mock('@/components/settings/PluginAgentModal.vue', () => ({
  default: defineComponent({
    name: 'PluginAgentModal',
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
    },
    setup(props) {
      return () => h(
        'div',
        {
          class: 'plugin-agent-modal-stub',
          'data-open': String(Boolean(props.modelValue)),
        },
        'PluginAgentModal stub',
      )
    },
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
    exportPluginMock.mockReset()
    importPluginMock.mockReset()
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
    exportPluginMock.mockResolvedValue({
      blob: new Blob(['zip-bytes'], { type: 'application/zip' }),
      filename: 'plugin_one.zip',
    })
    importPluginMock.mockResolvedValue({
      success: true,
      plugin: {
        id: 'plugin_imported',
        display_name: 'Imported Plugin',
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

  it('opens the plugin agent modal from the auto-generate button', async () => {
    const wrapper = mount(PluginManager)
    await flushPromises()

    const agentButton = wrapper.findAll('button').find(
      button => button.text().includes('自动生成插件'),
    )
    expect(agentButton).toBeTruthy()

    await agentButton!.trigger('click')
    await flushPromises()

    expect(wrapper.find('.plugin-agent-modal-stub').attributes('data-open')).toBe('true')
  })

  it('exports a plugin package from the export button', async () => {
    const objectUrlSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:plugin-one')
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    const wrapper = mount(PluginManager)
    await flushPromises()

    const exportButton = wrapper.findAll('button').find(button => button.attributes('title') === '导出')
    expect(exportButton).toBeTruthy()

    await exportButton!.trigger('click')
    await flushPromises()

    expect(exportPluginMock).toHaveBeenCalledWith('plugin_one')
    expect(objectUrlSpy).toHaveBeenCalledTimes(1)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(revokeSpy).toHaveBeenCalledWith('blob:plugin-one')
    expect(toastSuccessMock).toHaveBeenCalledWith('已导出 Plugin One')

    clickSpy.mockRestore()
    revokeSpy.mockRestore()
    objectUrlSpy.mockRestore()
  })

  it('retries import with replace after conflict confirmation', async () => {
    importPluginMock
      .mockRejectedValueOnce({
        message: '插件已存在',
        status: 409,
        details: {
          plugin_id: 'plugin_one',
        },
      })
      .mockResolvedValueOnce({
        success: true,
        plugin: {
          id: 'plugin_one',
          display_name: 'Plugin One',
        },
      })

    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const wrapper = mount(PluginManager)
    await flushPromises()

    const fileInput = wrapper.find('input[type="file"]')
    expect(fileInput.exists()).toBe(true)

    const file = new File(['zip-bytes'], 'plugin_one.zip', { type: 'application/zip' })
    Object.defineProperty(fileInput.element, 'files', {
      value: [file],
      configurable: true,
    })
    await fileInput.trigger('change')
    await flushPromises()

    expect(importPluginMock).toHaveBeenNthCalledWith(1, file, false)
    expect(confirmSpy).toHaveBeenCalled()
    expect(importPluginMock).toHaveBeenNthCalledWith(2, file, true)
    expect(refreshPluginsMock).toHaveBeenCalledTimes(1)
    expect(toastSuccessMock).toHaveBeenCalledWith('插件导入成功')

    confirmSpy.mockRestore()
  })
})
