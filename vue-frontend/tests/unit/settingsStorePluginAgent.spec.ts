import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import type { V2SettingsDocument, V2SettingsTransaction } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const settingsApiMocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: settingsApiMocks.getV2Settings,
  saveV2SettingsTransaction: settingsApiMocks.saveV2SettingsTransaction,
}))

function backendDocument(
  agentModel = 'backend-agent-model',
  revision = 5,
  providerRevision = 3,
): V2SettingsDocument {
  const settings = createDefaultSettings()
  settings.translation.modelName = 'backend-translation-model'
  settings.pluginAgent.provider = 'siliconflow'
  settings.pluginAgent.modelName = agentModel
  return {
    settings: [
      {
        domain: 'translation',
        payload: settings as unknown as Record<string, unknown>,
        revision,
      },
      {
        domain: 'text_style_defaults',
        payload: settings.textStyle as unknown as Record<string, unknown>,
        revision,
      },
      {
        domain: 'workflow_preferences',
        payload: {
          rememberWorkflowModeEnabled: false,
          lastWorkflowMode: 'translate-current',
        },
        revision,
      },
      {
        domain: 'export_preferences',
        payload: { preserveOriginalFilenames: false },
        revision,
      },
    ],
    bookSettings: [],
    providerSettings: [{
      domain: 'plugin_agent',
      provider: 'siliconflow',
      payload: {
        modelName: agentModel,
        customBaseUrl: '',
        openaiOptions: settings.pluginAgent.openaiOptions,
      },
      revision: providerRevision,
      credentialVersionId: 'credential-version-1',
    }],
    credentials: [{
      credentialId: 'credential-1',
      credentialVersionId: 'credential-version-1',
      currentVersion: 1,
      domain: 'plugin_agent',
      hasKey: true,
      provider: 'siliconflow',
      revision: 2,
      secret: { api_key: 'stored-agent-key' },
    }, {
      credentialId: 'translation-credential',
      credentialVersionId: 'translation-credential-version',
      currentVersion: 1,
      domain: 'translation',
      hasKey: true,
      provider: 'siliconflow',
      revision: 4,
      secret: { api_key: 'stored-translation-key' },
    }],
  }
}

function pluginAgentDocument(): V2SettingsDocument {
  const document = backendDocument()
  return {
    ...document,
    settings: document.settings.filter(entry => entry.domain === 'translation'),
    credentials: document.credentials.filter(entry => entry.domain === 'plugin_agent'),
  }
}

describe('settings store plugin agent configuration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
    settingsApiMocks.getV2Settings.mockResolvedValue(backendDocument())
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'translation', revision: 6 }],
      bookSettings: [],
      providerSettings: [{
        domain: 'plugin_agent',
        provider: 'siliconflow',
        revision: 4,
      }],
      credentials: [{
        credentialId: 'credential-1',
        credentialVersionId: 'credential-version-2',
        currentVersion: 2,
        domain: 'plugin_agent',
        hasKey: true,
        provider: 'siliconflow',
        revision: 3,
        secret: { api_key: 'agent-key' },
      }],
      prompts: [],
    })
  })

  it('keeps plugin agent credentials isolated per provider', () => {
    const store = useSettingsStore()

    expect(store.settings.pluginAgent.openaiOptions.execution.transportRetries).toBe(3)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(3)

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

    expect(store.settings.pluginAgent.openaiOptions.execution).toMatchObject({
      rpmLimit: 11,
      transportRetries: 2,
      businessRetries: 4,
      useStream: false,
    })
    expect(store.settings.pluginAgent.openaiOptions.request).toMatchObject({
      forceJsonOutput: true,
      extraBody: { reasoning_effort: 'low' },
    })
    expect((store.settings.pluginAgent as Record<string, unknown>).rpmLimit).toBeUndefined()
    expect((store.settings.pluginAgent as Record<string, unknown>).useStream).toBeUndefined()
  })

  it('saves only plugin agent settings against a fresh backend revision', async () => {
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(backendDocument())
      .mockResolvedValueOnce(pluginAgentDocument())

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.translation.modelName = 'unsaved-local-translation-change'
    store.updatePluginAgent({
      apiKey: 'agent-key',
      modelName: 'agent-model',
      customBaseUrl: 'https://agent.example/v1',
    })

    expect(await store.savePluginAgentSettings()).toBe(true)

    expect(settingsApiMocks.getV2Settings).toHaveBeenNthCalledWith(
      2,
      ['translation', 'plugin_agent'],
    )
    expect(settingsApiMocks.getV2Settings).toHaveBeenCalledTimes(2)
    const transaction = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    ) as V2SettingsTransaction
    expect(transaction.settings).toHaveLength(1)
    expect(transaction.settings?.[0]).toMatchObject({
      domain: 'translation',
      baseRevision: 5,
    })
    expect(transaction.settings?.[0]?.payload).toMatchObject({
      translation: { modelName: 'backend-translation-model' },
      pluginAgent: {
        modelName: 'agent-model',
        customBaseUrl: 'https://agent.example/v1',
      },
    })
    expect(
      (transaction.settings?.[0]?.payload.pluginAgent as Record<string, unknown>)
        .apiKey,
    ).toBeUndefined()
    expect(transaction.providerSettings).toEqual([
      expect.objectContaining({
        domain: 'plugin_agent',
        provider: 'siliconflow',
        baseRevision: 3,
        credentialEditRef: 'credential:plugin_agent:siliconflow',
        payload: expect.objectContaining({
          modelName: 'agent-model',
          customBaseUrl: 'https://agent.example/v1',
        }),
      }),
    ])
    expect(transaction.credentialEdits).toEqual([{
      domain: 'plugin_agent',
      provider: 'siliconflow',
      secret: { api_key: 'agent-key' },
      baseRevision: 2,
      credentialId: 'credential-1',
      clientRef: 'credential:plugin_agent:siliconflow',
    }])
    expect(store.settings.translation.modelName).toBe('unsaved-local-translation-change')
    expect(store.settings.pluginAgent.modelName).toBe('agent-model')
    expect(store.settings.pluginAgent.apiKey).toBe('agent-key')
    expect(store.credentialSummaries).toContainEqual(expect.objectContaining({
      domain: 'translation',
      provider: 'siliconflow',
      secret: { api_key: 'stored-translation-key' },
    }))
  })

  it('resets plugin agent openai options to defaults for an uncached provider', () => {
    const store = useSettingsStore()
    store.updatePluginAgent({
      rpmLimit: 23,
      businessRetries: 5,
      forceJsonOutput: true,
      useStream: false,
      extraBody: { reasoning_effort: 'high' },
    })

    store.setPluginAgentProvider('deepseek')

    expect(store.settings.pluginAgent.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(3)
    expect(store.settings.pluginAgent.openaiOptions.execution.useStream).toBe(true)
    expect(store.settings.pluginAgent.openaiOptions.request.forceJsonOutput).toBe(false)
    expect(store.settings.pluginAgent.openaiOptions.request.extraBody).toBeUndefined()
  })

  it('does not save before loading settings', async () => {
    const store = useSettingsStore()
    expect(await store.savePluginAgentSettings()).toBe(false)
    expect(settingsApiMocks.saveV2SettingsTransaction).not.toHaveBeenCalled()
  })

  it('leaves an untouched plugin agent from another page unchanged', async () => {
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(backendDocument())
      .mockResolvedValueOnce(backendDocument('other-page-model', 6, 4))
    const store = useSettingsStore()
    await store.loadFromBackend()

    expect(await store.savePluginAgentSettings()).toBe(true)
    expect(settingsApiMocks.saveV2SettingsTransaction).not.toHaveBeenCalled()
  })

  it('rejects stale plugin agent edits without overwriting another page', async () => {
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(backendDocument())
      .mockResolvedValueOnce(backendDocument('other-page-model', 6, 4))
    settingsApiMocks.saveV2SettingsTransaction.mockRejectedValueOnce(new Error('设置已被其他页面更新'))
    const store = useSettingsStore()
    await store.loadFromBackend()
    store.updatePluginAgent({ rpmLimit: 17 })

    expect(await store.savePluginAgentSettings()).toBe(false)
    expect(store.backendError).toContain('其他页面')
    expect(store.settings.pluginAgent.openaiOptions.execution.rpmLimit).toBe(17)
    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0] as V2SettingsTransaction
    expect(transaction.settings?.[0]?.baseRevision).toBe(5)
    expect(transaction.providerSettings?.[0]?.baseRevision).toBe(3)
  })

  it('keeps the loaded provider revision when only its key is edited', async () => {
    const current = pluginAgentDocument()
    current.providerSettings[0]!.revision = 4
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(backendDocument())
      .mockResolvedValueOnce(current)
    settingsApiMocks.saveV2SettingsTransaction.mockRejectedValueOnce(new Error('provider setting conflict'))
    const store = useSettingsStore()
    await store.loadFromBackend()
    store.updatePluginAgent({ apiKey: '' })

    expect(await store.savePluginAgentSettings()).toBe(false)
    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0] as V2SettingsTransaction
    expect(transaction.providerSettings?.[0]?.baseRevision).toBe(3)
    expect(transaction.credentialEdits?.[0]?.secret).toEqual({ api_key: '' })
    expect(store.settings.pluginAgent.apiKey).toBe('')
  })

  it('does not refresh the global revision after saving only the plugin key', async () => {
    const current = pluginAgentDocument()
    current.settings[0]!.revision = 6
    settingsApiMocks.getV2Settings
      .mockResolvedValueOnce(backendDocument())
      .mockResolvedValueOnce(current)
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValueOnce({
      settings: [], bookSettings: [], providerSettings: [{ domain: 'plugin_agent', provider: 'siliconflow', revision: 4 }],
      credentials: [], prompts: [],
    })
    const store = useSettingsStore()
    await store.loadFromBackend()
    store.updatePluginAgent({ apiKey: '' })
    expect(await store.savePluginAgentSettings()).toBe(true)

    store.settings.translation.modelName = 'pending-model'
    await store.saveToBackend()
    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[1]?.[0] as V2SettingsTransaction
    expect(transaction.settings?.find(row => row.domain === 'translation')?.baseRevision).toBe(5)
  })
})
