import { ref } from 'vue'
import { describe, expect, it } from 'vitest'

import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
  useWebImportSettings,
} from '@/stores/settings/modules/webImport'

describe('web import provider compatibility', () => {
  it('exposes canonical custom provider in the selector list', () => {
    const providers = WEB_IMPORT_AGENT_PROVIDERS.map(provider => provider.value)

    expect(providers).toContain('openai')
    expect(providers).toContain('siliconflow')
    expect(providers).toContain('deepseek')
    expect(providers).toContain('volcano')
    expect(providers).toContain('gemini')
    expect(providers).toContain('custom')
    expect(providers).not.toContain('qwen')
    expect(providers).not.toContain('custom_openai')
  })

  it('normalizes legacy custom provider ids before saving settings', () => {
    const settings = ref(createDefaultWebImportSettings())
    const providerConfigs = ref(createDefaultWebImportProviderConfigs())
    const { setAgentProvider } = useWebImportSettings(settings, providerConfigs)

    setAgentProvider('custom_openai')

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
})
