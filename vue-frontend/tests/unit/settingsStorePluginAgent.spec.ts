import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settingsStore'

describe('settings store plugin agent configuration', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('keeps plugin agent credentials isolated per provider', () => {
    const store = useSettingsStore()

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
  })
})
