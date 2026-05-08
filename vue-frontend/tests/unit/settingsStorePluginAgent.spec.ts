import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const { getUserSettingsMock, saveUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn(),
  saveUserSettingsMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock,
  saveUserSettings: saveUserSettingsMock,
}))

import { useSettingsStore } from '@/stores/settingsStore'

describe('settings store plugin agent configuration', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    getUserSettingsMock.mockReset()
    saveUserSettingsMock.mockReset()
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        settingsSchemaVersion: 3,
        translation: {
          modelName: 'backend-translation-model',
        },
        pluginAgent: {
          provider: 'siliconflow',
          modelName: 'backend-agent-model',
        },
        providerConfigs: {
          pluginAgent: {
            siliconflow: {
              modelName: 'backend-agent-model',
            },
          },
        },
      },
    })
    saveUserSettingsMock.mockResolvedValue({
      success: true,
    })
  })

  it('keeps plugin agent credentials isolated per provider', () => {
    const store = useSettingsStore()

    expect(store.settings.pluginAgent.openaiOptions.execution.transportRetries).toBe(10)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(10)

    store.updatePluginAgent({
      apiKey: 'sf-key',
      modelName: 'sf-model',
      customBaseUrl: 'https://sf.example/v1',
    })

    store.setPluginAgentProvider('deepseek')

    expect(store.providerConfigs.pluginAgent.siliconflow).toEqual(
      expect.objectContaining({
        apiKey: 'sf-key',
        modelName: 'sf-model',
        customBaseUrl: 'https://sf.example/v1',
      }),
    )
    expect(store.settings.pluginAgent.provider).toBe('deepseek')
    expect(store.settings.pluginAgent.apiKey).toBe('')
    expect(store.settings.pluginAgent.modelName).toBe('')
  })

  it('updates nested openai options through plugin agent helpers', () => {
    const store = useSettingsStore()

    store.updatePluginAgent({
      rpmLimit: 11,
      transportRetries: 2,
      businessRetries: 4,
      forceJsonOutput: true,
      useStream: false,
      extraBody: { reasoning_effort: 'low' },
    })

    expect(store.settings.pluginAgent.openaiOptions.execution.rpmLimit).toBe(11)
    expect(store.settings.pluginAgent.openaiOptions.execution.transportRetries).toBe(2)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(4)
    expect(store.settings.pluginAgent.openaiOptions.request.forceJsonOutput).toBe(true)
    expect(store.settings.pluginAgent.openaiOptions.execution.useStream).toBe(false)
    expect(store.settings.pluginAgent.openaiOptions.request.extraBody).toEqual({
      reasoning_effort: 'low',
    })
    expect((store.settings.pluginAgent as Record<string, unknown>).rpmLimit).toBeUndefined()
    expect((store.settings.pluginAgent as Record<string, unknown>).useStream).toBeUndefined()
  })

  it('saves only plugin agent settings to backend payload', async () => {
    const store = useSettingsStore()

    store.settings.translation.modelName = 'unsaved-local-translation-change'
    store.updatePluginAgent({
      apiKey: 'agent-key',
      modelName: 'agent-model',
      customBaseUrl: 'https://agent.example/v1',
    })

    const success = await store.savePluginAgentSettings()

    expect(success).toBe(true)
    expect(getUserSettingsMock).toHaveBeenCalledTimes(1)
    expect(saveUserSettingsMock).toHaveBeenCalledTimes(1)

    const payload = saveUserSettingsMock.mock.calls[0]?.[0] as Record<string, any>
    expect(payload.translation.modelName).toBe('backend-translation-model')
    expect(payload.pluginAgent.modelName).toBe('agent-model')
    expect(payload.providerConfigs.pluginAgent.siliconflow).toEqual(
      expect.objectContaining({
        apiKey: 'agent-key',
        modelName: 'agent-model',
        customBaseUrl: 'https://agent.example/v1',
      }),
    )
    expect(payload.pluginAgent.rpmLimit).toBeUndefined()
    expect(payload.pluginAgent.useStream).toBeUndefined()
    expect(payload.settingsSchemaVersion).toBe(3)
  })

  it('resets plugin agent openai options to defaults when switching to uncached provider', () => {
    const store = useSettingsStore()

    store.updatePluginAgent({
      rpmLimit: 23,
      businessRetries: 5,
      forceJsonOutput: true,
      useStream: false,
      extraBody: { reasoning_effort: 'high' },
    })

    store.setPluginAgentProvider('deepseek')

    expect(store.settings.pluginAgent.openaiOptions.execution.rpmLimit).toBe(7)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(10)
    expect(store.settings.pluginAgent.openaiOptions.execution.useStream).toBe(true)
    expect(store.settings.pluginAgent.openaiOptions.request.forceJsonOutput).toBe(false)
    expect(store.settings.pluginAgent.openaiOptions.request.extraBody).toBeUndefined()
  })
})
