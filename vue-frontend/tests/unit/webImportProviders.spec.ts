import { ref } from 'vue'
import { describe, expect, it } from 'vitest'

import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
  useWebImportSettings,
} from '@/stores/settings/modules/webImport'
import {
  parseWebImportSettingsPayload,
  WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
} from '@/stores/webImportSettingsPayload'

describe('web import provider settings', () => {
  it('exposes canonical custom provider in the selector list', () => {
    const providers = WEB_IMPORT_AGENT_PROVIDERS.map(provider => provider.value)

    expect(providers).toContain('openai')
    expect(providers).toContain('siliconflow')
    expect(providers).toContain('deepseek')
    expect(providers).toContain('volcano')
    expect(providers).toContain('gemini')
    expect(providers).toContain('custom')
    expect(providers).not.toContain('qwen')
  })

  it('uses the current custom provider id when saving settings', () => {
    const settings = ref(createDefaultWebImportSettings())
    const providerConfigs = ref(createDefaultWebImportProviderConfigs())
    const { setAgentProvider } = useWebImportSettings(settings, providerConfigs)

    setAgentProvider('custom')

    expect(settings.value.agent.provider).toBe('custom')
  })

  it('stores agent credentials per provider when switching', () => {
    const settings = ref(createDefaultWebImportSettings())
    const providerConfigs = ref(createDefaultWebImportProviderConfigs())
    const {
      setAgentApiKey,
      setAgentBaseUrl,
      setAgentModelName,
      setAgentProvider,
    } = useWebImportSettings(settings, providerConfigs)

    setAgentApiKey('openai-key')
    setAgentModelName('gpt-web-import')
    setAgentBaseUrl('https://openai.example/v1')

    setAgentProvider('deepseek')

    expect(providerConfigs.value.agent.openai).toEqual({
      apiKey: 'openai-key',
      modelName: 'gpt-web-import',
      customBaseUrl: 'https://openai.example/v1',
    })
    expect(settings.value.agent.apiKey).toBe('')
    expect(settings.value.agent.modelName).toBe('')
    expect(settings.value.agent.customBaseUrl).toBe('')
  })

  it('accepts only the current web-import payload schema and providers', () => {
    const current = {
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: createDefaultWebImportSettings(),
      providerConfigs: createDefaultWebImportProviderConfigs(),
    }
    expect(parseWebImportSettingsPayload(current)).not.toBeNull()

    expect(parseWebImportSettingsPayload({
      settings: current.settings,
      providerConfigs: current.providerConfigs,
    })).toBeNull()
    expect(parseWebImportSettingsPayload({
      ...current,
      webImportSettingsSchemaVersion: 0,
    })).toBeNull()
    expect(parseWebImportSettingsPayload({
      ...current,
      removedField: true,
    })).toBeNull()
    expect(parseWebImportSettingsPayload({
      ...current,
      providerConfigs: {
        agent: {
          retired: {
            apiKey: '',
            modelName: 'retired-model',
            customBaseUrl: '',
          },
        },
      },
    })).toBeNull()
  })

  it('uses the same numeric boundaries as the backend without arbitrary size gates', () => {
    const providerConfigs = createDefaultWebImportProviderConfigs()
    const fractional = createDefaultWebImportSettings()
    fractional.download.concurrency = 1.5
    expect(parseWebImportSettingsPayload({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: fractional,
      providerConfigs,
    })).toBeNull()

    const negative = createDefaultWebImportSettings()
    negative.download.retries = -1
    expect(parseWebImportSettingsPayload({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: negative,
      providerConfigs,
    })).toBeNull()

    const fractionalTimeouts = createDefaultWebImportSettings()
    fractionalTimeouts.agent.timeout = 120.5
    fractionalTimeouts.download.timeout = 30.25
    expect(parseWebImportSettingsPayload({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: fractionalTimeouts,
      providerConfigs,
    })).not.toBeNull()

    const largeDimensions = createDefaultWebImportSettings()
    largeDimensions.imagePreprocess.compression.maxWidth = 1_000_000
    largeDimensions.imagePreprocess.compression.maxHeight = 1_000_000
    expect(parseWebImportSettingsPayload({
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: largeDimensions,
      providerConfigs,
    })).not.toBeNull()
  })
})
