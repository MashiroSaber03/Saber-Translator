import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWebImportStore } from '@/stores/webImportStore'
import { createDefaultWebImportSettings } from '@/stores/settings/modules/webImport'

const mocks = vi.hoisted(() => ({
  getSettings: vi.fn(),
  saveSettings: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: mocks.getSettings,
  saveV2SettingsTransaction: mocks.saveSettings,
}))

function settingsDocument(
  provider = 'openai',
  modelName = 'gpt-4o-mini',
) {
  const settings = createDefaultWebImportSettings()
  settings.agent.provider = provider
  settings.agent.modelName = modelName
  return {
    credentials: [{
      credentialId: 'credential-1',
      credentialVersionId: 'credential-version-1',
      domain: 'web_import_agent',
      hasKey: true,
      provider,
      revision: 1,
    }],
    providerSettings: [{
      credentialVersionId: 'credential-version-1',
      domain: 'web_import_agent',
      payload: {
        customBaseUrl: '',
        modelName,
      },
      provider,
      revision: 1,
    }],
    settings: [{
      domain: 'web_import',
      payload: settings,
      revision: 2,
      schemaVersion: 1,
    }],
  }
}

describe('webImportStore backend settings workflow', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    localStorage.clear()
    mocks.getSettings.mockResolvedValue(settingsDocument())
    mocks.saveSettings.mockResolvedValue({
      credentials: [],
      providerSettings: [],
      settings: [],
    })
  })

  it('keeps agent credentials isolated per provider while editing drafts', () => {
    const store = useWebImportStore()

    store.setAgentApiKey('openai-key')
    store.setAgentModelName('gpt-custom')
    store.setAgentProvider('deepseek')
    expect(store.draftSettings.agent.apiKey).toBe('')
    store.setAgentApiKey('deepseek-key')
    store.setAgentModelName('deepseek-chat')
    store.setAgentProvider('openai')

    expect(store.draftSettings.agent.apiKey).toBe('openai-key')
    expect(store.draftSettings.agent.modelName).toBe('gpt-custom')
  })

  it('loads configuration and credential availability from the backend', async () => {
    mocks.getSettings.mockResolvedValue(settingsDocument('deepseek', 'deepseek-chat'))
    const store = useWebImportStore()

    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings.agent.provider).toBe('deepseek')
    expect(store.settings.agent.modelName).toBe('deepseek-chat')
    expect(store.settings.agent.apiKey).toBe('')
    expect(store.hasCredential('web_import_agent', 'deepseek')).toBe(true)
  })

  it('rejects a missing or empty backend settings fact instead of using browser defaults', async () => {
    const store = useWebImportStore()
    mocks.getSettings.mockResolvedValue({
      credentials: [],
      providerSettings: [],
      settings: [{
        domain: 'web_import',
        payload: {},
        revision: 1,
        schemaVersion: 1,
      }],
    })

    expect(await store.loadFromBackend()).toBe(false)
    await store.acceptDisclaimer()
    expect(store.modalVisible).toBe(false)
    expect(store.error).toBe('网页导入设置加载失败')
  })

  it('saves settings, provider values, and secret edits in one backend transaction', async () => {
    const store = useWebImportStore()
    await store.loadFromBackend()
    store.setAgentProvider('deepseek')
    store.setAgentModelName('deepseek-chat')
    store.setAgentApiKey('deepseek-key')
    mocks.getSettings.mockResolvedValue(settingsDocument('deepseek', 'deepseek-chat'))

    expect(await store.saveSettings()).toBe(true)

    expect(mocks.saveSettings).toHaveBeenCalledWith(expect.objectContaining({
      credentialEdits: expect.arrayContaining([
        expect.objectContaining({
          domain: 'web_import_agent',
          provider: 'deepseek',
          secret: { api_key: 'deepseek-key' },
        }),
      ]),
      providerSettings: expect.arrayContaining([
        expect.objectContaining({
          domain: 'web_import_agent',
          provider: 'deepseek',
          payload: expect.objectContaining({ modelName: 'deepseek-chat' }),
        }),
      ]),
      settings: [
        expect.objectContaining({
          baseRevision: 2,
          domain: 'web_import',
        }),
      ],
    }))
    expect(store.settings.agent.apiKey).toBe('')
  })

})
