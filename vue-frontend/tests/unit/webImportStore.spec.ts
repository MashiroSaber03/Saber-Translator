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

function settingsDocument(provider = 'openai', modelName = 'gpt-4o-mini') {
  const settings = createDefaultWebImportSettings()
  settings.agent.provider = provider
  settings.agent.modelName = modelName
  const backendSettings = structuredClone(settings)
  delete (backendSettings.firecrawl as Partial<typeof backendSettings.firecrawl>).apiKey
  delete (backendSettings.agent as Partial<typeof backendSettings.agent>).apiKey
  delete (
    backendSettings.advanced as Partial<typeof backendSettings.advanced>
  ).customCookie
  delete (
    backendSettings.advanced as Partial<typeof backendSettings.advanced>
  ).customHeaders
  return {
    credentials: [
      {
        credentialId: 'credential-1',
        credentialVersionId: 'credential-version-1',
        currentVersion: 1,
        domain: 'web_import_agent',
        hasKey: true,
        provider,
        revision: 1,
        secret: { api_key: `${provider}-key` },
      },
    ],
    providerSettings: [
      {
        credentialVersionId: 'credential-version-1',
        domain: 'web_import_agent',
        payload: {
          customBaseUrl: '',
          modelName,
        },
        provider,
        revision: 1,
      },
    ],
    settings: [
      {
        domain: 'web_import',
        payload: backendSettings,
        revision: 2,
        schemaVersion: 1,
      },
    ],
  }
}

describe('webImportStore backend settings workflow', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    localStorage.clear()
    mocks.getSettings.mockResolvedValue(settingsDocument())
    mocks.saveSettings.mockResolvedValue({
      bookSettings: [],
      credentials: [
        {
          credentialId: 'credential-deepseek',
          credentialVersionId: 'credential-version-deepseek',
          currentVersion: 1,
          domain: 'web_import_agent',
          hasKey: true,
          provider: 'deepseek',
          revision: 1,
          secret: { api_key: 'deepseek-key' },
        },
      ],
      prompts: [],
      providerSettings: [
        { domain: 'web_import_agent', provider: 'openai', revision: 2 },
        { domain: 'web_import_agent', provider: 'deepseek', revision: 1 },
        { domain: 'web_import_firecrawl', provider: 'firecrawl', revision: 1 },
        { domain: 'web_import_http', provider: 'headers', revision: 1 },
      ],
      settings: [{ domain: 'web_import', revision: 3 }],
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
    const document = settingsDocument('deepseek', 'deepseek-chat')
    document.credentials.push(
      {
        credentialId: 'credential-firecrawl',
        credentialVersionId: 'credential-version-firecrawl',
        currentVersion: 1,
        domain: 'web_import_firecrawl',
        hasKey: true,
        provider: 'firecrawl',
        revision: 1,
        secret: { api_key: 'firecrawl-key' },
      },
      {
        credentialId: 'credential-http',
        credentialVersionId: 'credential-version-http',
        currentVersion: 1,
        domain: 'web_import_http',
        hasKey: true,
        provider: 'headers',
        revision: 1,
        secret: { cookie: 'session=value', headers: { Referer: 'https://example.com' } },
      },
    )
    mocks.getSettings.mockResolvedValue(document)
    const store = useWebImportStore()

    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings.agent.provider).toBe('deepseek')
    expect(store.settings.agent.modelName).toBe('deepseek-chat')
    expect(store.settings.agent.apiKey).toBe('deepseek-key')
    expect(store.settings.firecrawl.apiKey).toBe('firecrawl-key')
    expect(store.settings.advanced.customCookie).toBe('session=value')
    expect(store.settings.advanced.customHeaders).toBe(
      '{\n  "Referer": "https://example.com"\n}',
    )
  })

  it('rejects a missing or empty backend settings fact instead of using browser defaults', async () => {
    const store = useWebImportStore()
    mocks.getSettings.mockResolvedValue({
      credentials: [],
      providerSettings: [],
      settings: [
        {
          domain: 'web_import',
          payload: {},
          revision: 1,
          schemaVersion: 1,
        },
      ],
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

    expect(await store.saveSettings()).toBe(true)

    expect(mocks.saveSettings).toHaveBeenCalledWith(
      expect.objectContaining({
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
      })
    )
    expect(store.settings.agent.apiKey).toBe('deepseek-key')
    expect(store.draftSettings.agent.apiKey).toBe('deepseek-key')
    expect(mocks.getSettings).toHaveBeenCalledTimes(1)
  })

  it('uses the committed transaction revisions without a follow-up settings read', async () => {
    const store = useWebImportStore()
    await store.loadFromBackend()
    store.setAgentProvider('deepseek')
    store.setAgentModelName('deepseek-chat')

    expect(await store.saveSettings()).toBe(true)
    store.setDownloadConcurrency(5)
    expect(await store.saveSettings()).toBe(true)

    const secondPayload = mocks.saveSettings.mock.calls[1]?.[0]
    expect(secondPayload.settings[0].baseRevision).toBe(3)
    expect(secondPayload.providerSettings).toContainEqual(
      expect.objectContaining({
        domain: 'web_import_agent',
        provider: 'deepseek',
        baseRevision: 1,
      })
    )
    expect(mocks.getSettings).toHaveBeenCalledTimes(1)
  })

  it('rejects undocumented or non-string custom header formats before saving', async () => {
    const store = useWebImportStore()
    await store.loadFromBackend()

    store.setCustomHeaders('X-Test: value')
    expect(await store.saveSettings()).toBe(false)
    expect(store.settingsSaveError).toBe('自定义 Headers 必须是有效的 JSON 对象')
    expect(mocks.saveSettings).not.toHaveBeenCalled()

    store.setCustomHeaders('{"X-Retry": 3}')
    expect(await store.saveSettings()).toBe(false)
    expect(store.settingsSaveError).toBe('自定义 Headers 的名称和值必须是非空字符串')
    expect(mocks.saveSettings).not.toHaveBeenCalled()
  })
})
