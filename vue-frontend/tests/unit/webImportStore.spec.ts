import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import {
  WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
  useWebImportStore,
} from '@/stores/webImportStore'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
} from '@/stores/settings/modules/webImport'

const { getWebImportSettingsMock, saveWebImportSettingsMock } = vi.hoisted(() => ({
  getWebImportSettingsMock: vi.fn(),
  saveWebImportSettingsMock: vi.fn(),
}))

vi.mock('@/api/webImport', async () => {
  const actual = await vi.importActual<typeof import('@/api/webImport')>('@/api/webImport')
  return {
    ...actual,
    getWebImportSettings: getWebImportSettingsMock,
    saveWebImportSettings: saveWebImportSettingsMock,
  }
})

describe('webImportStore settings workflow', () => {
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    localStorageMock = {}
    setActivePinia(createPinia())

    getWebImportSettingsMock.mockReset()
    saveWebImportSettingsMock.mockReset()
    getWebImportSettingsMock.mockResolvedValue({
      success: true,
      settings: {},
      providerConfigs: { agent: {} },
    })
    saveWebImportSettingsMock.mockResolvedValue({ success: true })

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => {
      return localStorageMock[key] || null
    })

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })

    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps agent credentials isolated per provider while editing drafts', () => {
    const store = useWebImportStore()

    store.beginSettingsEdit()
    store.setAgentApiKey('openai-key')
    store.setAgentModelName('gpt-4o-mini-custom')
    store.setAgentBaseUrl('https://openai.example/v1')

    store.setAgentProvider('deepseek')
    expect(store.draftSettings.agent.provider).toBe('deepseek')
    expect(store.draftSettings.agent.apiKey).toBe('')
    expect(store.draftSettings.agent.modelName).toBe('')
    expect(store.draftSettings.agent.customBaseUrl).toBe('')

    store.setAgentApiKey('deepseek-key')
    store.setAgentModelName('deepseek-chat')
    store.setAgentBaseUrl('https://deepseek.example/v1')

    store.setAgentProvider('openai')

    expect(store.draftSettings.agent.provider).toBe('openai')
    expect(store.draftSettings.agent.apiKey).toBe('openai-key')
    expect(store.draftSettings.agent.modelName).toBe('gpt-4o-mini-custom')
    expect(store.draftSettings.agent.customBaseUrl).toBe('https://openai.example/v1')
  })

  it('discards unsaved draft changes without mutating committed settings', () => {
    const store = useWebImportStore()

    expect(store.settings.agent.modelName).toBe('gpt-4o-mini')

    store.beginSettingsEdit()
    store.setAgentModelName('draft-model')

    expect(store.draftSettings.agent.modelName).toBe('draft-model')
    expect(store.settings.agent.modelName).toBe('gpt-4o-mini')
    expect(store.hasUnsavedSettings).toBe(true)

    store.discardSettingsChanges()

    expect(store.draftSettings.agent.modelName).toBe('gpt-4o-mini')
    expect(store.settings.agent.modelName).toBe('gpt-4o-mini')
    expect(store.hasUnsavedSettings).toBe(false)
  })

  it('commits saved drafts to store, localStorage, and backend payload together', async () => {
    const store = useWebImportStore()

    store.beginSettingsEdit()
    store.setFirecrawlApiKey('fc-123')
    store.setAgentProvider('deepseek')
    store.setAgentApiKey('deepseek-key')
    store.setAgentModelName('deepseek-chat')
    store.setAgentBaseUrl('https://deepseek.example/v1')

    const success = await store.saveSettings()

    expect(success).toBe(true)
    expect(store.settings.agent.provider).toBe('deepseek')
    expect(store.settings.agent.apiKey).toBe('deepseek-key')
    expect(store.settings.agent.modelName).toBe('deepseek-chat')
    expect(store.settings.firecrawl.apiKey).toBe('fc-123')
    expect(store.hasUnsavedSettings).toBe(false)

    expect(saveWebImportSettingsMock).toHaveBeenCalledTimes(1)
    expect(saveWebImportSettingsMock).toHaveBeenCalledWith({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: store.settings,
      providerConfigs: store.providerConfigs,
    })

    const stored = JSON.parse(localStorageMock[STORAGE_KEY_WEB_IMPORT_SETTINGS] || '{}')
    expect(stored.webImportSettingsSchemaVersion).toBe(WEB_IMPORT_SETTINGS_SCHEMA_VERSION)
    expect(stored.settings.agent.provider).toBe('deepseek')
    expect(stored.providerConfigs.agent.deepseek.modelName).toBe('deepseek-chat')
  })

  it('loads complete current backend data', async () => {
    const settings = createDefaultWebImportSettings()
    settings.agent.provider = 'custom'
    settings.agent.apiKey = 'custom-key'
    settings.agent.modelName = 'custom-model'
    settings.agent.customBaseUrl = 'https://custom.example/v1'
    const providerConfigs = createDefaultWebImportProviderConfigs()
    providerConfigs.agent.custom = {
      apiKey: 'custom-key',
      modelName: 'custom-model',
      customBaseUrl: 'https://custom.example/v1',
    }

    getWebImportSettingsMock.mockResolvedValue({
      success: true,
      settings,
      providerConfigs,
    })

    const store = useWebImportStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.settings.agent.provider).toBe('custom')
    expect(store.settings.agent.apiKey).toBe('custom-key')
    expect(store.settings.download.timeout).toBe(30)
    expect(store.providerConfigs.agent.custom).toBeDefined()
    expect(store.providerConfigs.agent.custom?.modelName).toBe('custom-model')
  })

  it('keeps current payload parsing in the WebImport settings helper', async () => {
    const {
      buildWebImportSettingsPayload,
      parseLocalWebImportSettingsPayload,
      parseWebImportSettingsPayload,
    } = await import('@/stores/webImportSettingsPayload')
    const settings = createDefaultWebImportSettings()
    settings.agent.provider = 'deepseek'
    settings.agent.apiKey = 'deepseek-key'
    settings.agent.modelName = 'deepseek-chat'
    const providerConfigs = createDefaultWebImportProviderConfigs()
    providerConfigs.agent.deepseek = {
      apiKey: 'deepseek-key',
      modelName: 'deepseek-chat',
      customBaseUrl: '',
    }

    const payload = buildWebImportSettingsPayload(settings, providerConfigs)

    expect(parseWebImportSettingsPayload({
      settings: payload.settings,
      providerConfigs: payload.providerConfigs,
    })).toEqual(payload)
    expect(parseLocalWebImportSettingsPayload(payload)).toEqual(payload)
    expect(parseWebImportSettingsPayload({
      settings: {
        agent: {
          provider: 'deepseek',
        },
      },
      providerConfigs,
    })).toBeNull()
  })

  it('ignores incomplete backend data instead of filling missing fields', async () => {
    getWebImportSettingsMock.mockResolvedValue({
      success: true,
      hasStoredSettings: true,
      settings: {
        agent: {
          provider: 'custom',
          apiKey: 'custom-key',
          modelName: 'custom-model',
          customBaseUrl: 'https://custom.example/v1',
        },
      },
      providerConfigs: {
        agent: {
          custom: {
            apiKey: 'custom-key',
            modelName: 'custom-model',
            customBaseUrl: 'https://custom.example/v1',
          },
        },
      },
    })

    const store = useWebImportStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.agent.provider).toBe('openai')
    expect(store.providerConfigs.agent.custom).toBeUndefined()
  })

  it('preserves localStorage settings when backend has no stored payload yet', async () => {
    const settings = createDefaultWebImportSettings()
    settings.firecrawl.apiKey = 'fc-local'
    settings.agent.provider = 'deepseek'
    settings.agent.apiKey = 'deepseek-local'
    settings.agent.modelName = 'deepseek-chat'
    settings.agent.customBaseUrl = 'https://deepseek.local/v1'
    const providerConfigs = createDefaultWebImportProviderConfigs()
    providerConfigs.agent.deepseek = {
      apiKey: 'deepseek-local',
      modelName: 'deepseek-chat',
      customBaseUrl: 'https://deepseek.local/v1',
    }

    localStorageMock[STORAGE_KEY_WEB_IMPORT_SETTINGS] = JSON.stringify({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings,
      providerConfigs,
    })

    getWebImportSettingsMock.mockResolvedValue({
      success: true,
      hasStoredSettings: false,
      settings: {},
      providerConfigs: { agent: {} },
    })

    const store = useWebImportStore()
    store.loadFromStorage()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.firecrawl.apiKey).toBe('fc-local')
    expect(store.settings.agent.provider).toBe('deepseek')
    expect(store.settings.agent.modelName).toBe('deepseek-chat')
    expect(store.providerConfigs.agent.deepseek?.apiKey).toBe('deepseek-local')
  })

  it('ignores localStorage payloads without the current web import schema marker', () => {
    localStorageMock[STORAGE_KEY_WEB_IMPORT_SETTINGS] = JSON.stringify({
      settings: {
        firecrawl: { apiKey: 'fc-local' },
        agent: {
          provider: 'deepseek',
          apiKey: 'deepseek-local',
          modelName: 'deepseek-chat',
          customBaseUrl: 'https://deepseek.local/v1',
        },
      },
      providerConfigs: {
        agent: {
          deepseek: {
            apiKey: 'deepseek-local',
            modelName: 'deepseek-chat',
            customBaseUrl: 'https://deepseek.local/v1',
          },
        },
      },
    })

    const store = useWebImportStore()
    store.loadFromStorage()

    expect(store.settings.agent.provider).toBe('openai')
    expect(store.providerConfigs.agent.deepseek).toBeUndefined()
  })

  it('clamps download progress percentage to the progressbar range', () => {
    const store = useWebImportStore()

    store.updateDownloadProgress(8, 4)
    expect(store.downloadProgressPercent).toBe(100)

    store.updateDownloadProgress(-1, 4)
    expect(store.downloadProgressPercent).toBe(0)
  })
})
