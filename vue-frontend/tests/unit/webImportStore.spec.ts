import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import { useWebImportStore } from '@/stores/webImportStore'

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
      settings: store.settings,
      providerConfigs: store.providerConfigs,
    })

    const stored = JSON.parse(localStorageMock[STORAGE_KEY_WEB_IMPORT_SETTINGS] || '{}')
    expect(stored.settings.agent.provider).toBe('deepseek')
    expect(stored.providerConfigs.agent.deepseek.modelName).toBe('deepseek-chat')
  })

  it('merges backend data with defaults and normalizes legacy provider ids', async () => {
    getWebImportSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        agent: {
          provider: 'custom_openai',
          apiKey: 'custom-key',
          modelName: 'custom-model',
          customBaseUrl: 'https://custom.example/v1',
        },
      },
      providerConfigs: {
        agent: {
          custom_openai: {
            apiKey: 'custom-key',
            modelName: 'custom-model',
            customBaseUrl: 'https://custom.example/v1',
          },
        },
      },
    })

    const store = useWebImportStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.settings.agent.provider).toBe('custom')
    expect(store.settings.agent.apiKey).toBe('custom-key')
    expect(store.settings.download.timeout).toBe(30)
    expect(store.providerConfigs.agent.custom).toBeDefined()
    expect(store.providerConfigs.agent.custom?.modelName).toBe('custom-model')
    expect(store.providerConfigs.agent.custom_openai).toBeUndefined()
  })

  it('preserves localStorage settings when backend has no stored payload yet', async () => {
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
})
