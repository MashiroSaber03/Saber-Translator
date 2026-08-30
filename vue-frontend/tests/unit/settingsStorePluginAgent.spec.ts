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
        schemaVersion: 9,
      },
      {
        domain: 'text_style_defaults',
        payload: settings.textStyle as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 2,
      },
      {
        domain: 'workflow_preferences',
        payload: {
          rememberWorkflowModeEnabled: false,
          lastWorkflowMode: 'translate-current',
        },
        revision,
        schemaVersion: 1,
      },
      {
        domain: 'export_preferences',
        payload: { preserveOriginalFilenames: false },
        revision,
        schemaVersion: 1,
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
      schemaVersion: 1,
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

    expect(store.settings.pluginAgent.openaiOptions.execution.transportRetries).toBe(1)
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(0)

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

  it('keeps Browser DOM Agent provider state independent from Plugin Agent', () => {
    const store = useSettingsStore()
    store.updatePluginAgent({ modelName: 'plugin-model', apiKey: 'plugin-key' })
    store.updateBrowserDomAgent({
      modelName: 'dom-model',
      apiKey: 'dom-key',
      customBaseUrl: 'https://dom.example/v1',
    })

    store.setBrowserDomAgentProvider('deepseek')

    expect(store.providerConfigs.browserDomAgent.siliconflow).toEqual(
      expect.objectContaining({
        modelName: 'dom-model',
        apiKey: 'dom-key',
        customBaseUrl: 'https://dom.example/v1',
      }),
    )
    expect(store.settings.browserDomAgent.provider).toBe('deepseek')
    expect(store.settings.browserDomAgent.modelName).toBe('')
    expect(store.settings.pluginAgent.modelName).toBe('plugin-model')
    expect(store.settings.pluginAgent.apiKey).toBe('plugin-key')
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
      schemaVersion: 9,
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
    expect(store.settings.pluginAgent.openaiOptions.execution.businessRetries).toBe(0)
    expect(store.settings.pluginAgent.openaiOptions.execution.useStream).toBe(true)
    expect(store.settings.pluginAgent.openaiOptions.request.forceJsonOutput).toBe(false)
    expect(store.settings.pluginAgent.openaiOptions.request.extraBody).toBeUndefined()
  })
})
