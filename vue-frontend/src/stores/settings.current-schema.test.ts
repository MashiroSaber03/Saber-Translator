import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const settingsApiMocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: settingsApiMocks.getV2Settings,
  saveV2SettingsTransaction: settingsApiMocks.saveV2SettingsTransaction,
}))

import { useSettingsStore } from './settings'
import { createDefaultSettings } from './settings/defaults'

describe('useSettingsStore backend-first loading', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('blocks backend writes while authoritative settings are unavailable', async () => {
    settingsApiMocks.getV2Settings.mockRejectedValue(new Error('offline'))

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(false)
    expect(await store.saveToBackend()).toBe(false)
    expect(store.backendError).toContain('后端设置尚未加载')
  })

  it('loads provider memory without hydrating stored secrets into the browser', async () => {
    const settings = createDefaultSettings()
    settings.translation = {
      ...settings.translation,
      provider: 'custom',
      apiKey: '',
      modelName: 'translation-model',
      customBaseUrl: 'https://translation.example.com/v1',
    }
    localStorage.setItem('translationSettings', JSON.stringify({ apiKey: 'retired-key' }))
    localStorage.setItem('providerConfigs', JSON.stringify({ apiKey: 'retired-key' }))

    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [{
        domain: 'translation',
        revision: 4,
        schemaVersion: 3,
        payload: settings,
      }],
      bookSettings: [],
      providerSettings: [{
        domain: 'translation',
        provider: 'custom',
        revision: 2,
        schemaVersion: 1,
        credentialVersionId: 'version-1',
        payload: {
          modelName: 'cached-model',
          customBaseUrl: 'https://cached.example.com/v1',
          openaiOptions: settings.translation.openaiOptions,
        },
      }],
      credentials: [{
        credentialId: 'credential-1',
        credentialVersionId: 'version-1',
        currentVersion: 1,
        domain: 'translation',
        hasKey: true,
        provider: 'custom',
        revision: 1,
      }],
    })

    const store = useSettingsStore()
    store.initSettings()
    expect(await store.loadFromBackend()).toBe(true)

    expect(store.settings.translation.provider).toBe('custom')
    expect(store.settings.translation.apiKey).toBe('')
    expect(store.providerConfigs.translation.custom?.modelName).toBe('cached-model')
    expect(store.credentialSummaries[0]?.hasKey).toBe(true)
    expect(localStorage.getItem('translationSettings')).toBeNull()
    expect(localStorage.getItem('providerConfigs')).toBeNull()
  })
})
