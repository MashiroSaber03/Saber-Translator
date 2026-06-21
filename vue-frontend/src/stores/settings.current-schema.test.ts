import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const configMocks = vi.hoisted(() => ({
  getUserSettings: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: configMocks.getUserSettings,
}))

import { useSettingsStore } from './settings'

describe('useSettingsStore backend schema loading', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    configMocks.getUserSettings.mockReset()
  })

  it('ignores backend settings without the current schema version', async () => {
    const oldProviderSettingsKey = 'provider' + 'Settings'
    configMocks.getUserSettings.mockResolvedValue({
      success: true,
      settings: {
        modelProvider: 'siliconflow',
        hqTranslateProvider: 'siliconflow',
        [oldProviderSettingsKey]: {
          modelProvider: {
            siliconflow: {},
          },
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.providerConfigs.translation.siliconflow).toBeUndefined()
  })

  it('loads current schema settings and provider configs', async () => {
    configMocks.getUserSettings.mockResolvedValue({
      success: true,
      settings: {
        settingsSchemaVersion: 3,
        translation: {
          provider: 'custom',
          apiKey: 'translation-key',
          modelName: 'translation-model',
          customBaseUrl: 'https://translation.example.com/v1',
          openaiOptions: {
            request: { forceJsonOutput: true },
            execution: {
              useStream: true,
              rpmLimit: 12,
              transportRetries: 2,
              businessRetries: 4,
            },
          },
        },
        providerConfigs: {
          translation: {
            custom: {
              apiKey: 'cached-key',
              modelName: 'cached-model',
              customBaseUrl: 'https://cached.example.com/v1',
              openaiOptions: {
                request: { forceJsonOutput: true },
                execution: {
                  useStream: true,
                  rpmLimit: 10,
                  transportRetries: 3,
                  businessRetries: 5,
                },
              },
            },
          },
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.settings.translation.provider).toBe('custom')
    expect(store.settings.translation.openaiOptions.execution.businessRetries).toBe(4)
    expect(store.providerConfigs.translation.custom?.modelName).toBe('cached-model')
  })
})
