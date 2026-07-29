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
  })

  it('saves a new API key, reloads only its summary, and keeps it usable', async () => {
    const initialSettings = createDefaultSettings()
    initialSettings.translation = {
      ...initialSettings.translation,
      provider: 'deepseek',
      apiKey: '',
      modelName: 'deepseek-chat',
    }
    const initialDocument = {
      settings: [{
        domain: 'translation',
        revision: 0,
        schemaVersion: 3,
        payload: initialSettings,
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    }
    const persistedSettings = createDefaultSettings()
    persistedSettings.translation = {
      ...persistedSettings.translation,
      provider: 'deepseek',
      apiKey: '',
      modelName: 'deepseek-chat',
    }
    const persistedDocument = {
      settings: [{
        domain: 'translation',
        revision: 1,
        schemaVersion: 3,
        payload: persistedSettings,
      }],
      bookSettings: [],
      providerSettings: [{
        domain: 'translation',
        provider: 'deepseek',
        revision: 1,
        schemaVersion: 1,
        credentialVersionId: 'version-1',
        payload: {
          modelName: 'deepseek-chat',
          customBaseUrl: '',
          openaiOptions: persistedSettings.translation.openaiOptions,
        },
      }],
      credentials: [{
        credentialId: 'credential-1',
        credentialVersionId: 'version-1',
        currentVersion: 1,
        domain: 'translation',
        hasKey: true,
        provider: 'deepseek',
        revision: 1,
      }],
    }
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(initialDocument)
      .mockResolvedValueOnce(persistedDocument)
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.updateTranslationService({
      apiKey: 'sk-new-secret',
      modelName: 'deepseek-chat',
    })

    expect(await store.saveToBackend()).toBe(true)
    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    expect(transaction.credentialEdits).toContainEqual(expect.objectContaining({
      domain: 'translation',
      provider: 'deepseek',
      secret: { api_key: 'sk-new-secret' },
    }))
    expect(transaction.providerSettings).toContainEqual(expect.objectContaining({
      domain: 'translation',
      provider: 'deepseek',
      credentialEditRef: 'credential:translation:deepseek',
    }))
    expect(store.settings.translation.apiKey).toBe('')
    expect(store.hasCredential('translation', 'deepseek')).toBe(true)
  })

  it('does not restore a stale chapter snapshot after saving parallel mode', async () => {
    const initialSettings = createDefaultSettings()
    const persistedSettings = createDefaultSettings()
    persistedSettings.parallel.enabled = true
    persistedSettings.parallel.deepLearningLockSize = 2
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce({
        settings: [{
          domain: 'translation',
          revision: 0,
          schemaVersion: 3,
          payload: initialSettings,
        }],
        bookSettings: [],
        providerSettings: [],
        credentials: [],
      })
      .mockResolvedValueOnce({
        settings: [{
          domain: 'translation',
          revision: 1,
          schemaVersion: 3,
          payload: persistedSettings,
        }],
        bookSettings: [],
        providerSettings: [],
        credentials: [],
      })
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.hydrateChapterWorkState('chapter-1', {
      parallel: { enabled: false, deepLearningLockSize: 1 },
    })).toBe(true)

    store.updateSettings({
      parallel: { enabled: true, deepLearningLockSize: 2 },
    })

    expect(await store.saveToBackend()).toBe(true)
    expect(store.settings.parallel).toEqual({
      enabled: true,
      deepLearningLockSize: 2,
    })
  })

  it('does not submit a partial Baidu OCR credential replacement', async () => {
    const settings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [{
        domain: 'translation',
        revision: 0,
        schemaVersion: 3,
        payload: settings,
      }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.updateBaiduOcr({ apiKey: 'replacement-only', secretKey: '' })

    expect(await store.saveToBackend()).toBe(false)
    expect(store.backendError).toContain('必须同时填写')
    expect(settingsApiMocks.saveV2SettingsTransaction).not.toHaveBeenCalled()
  })
})
