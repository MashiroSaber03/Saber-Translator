import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import { createDefaultWebImportSettings, useWebImportSettings } from '@/stores/settings/modules/webImport'

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
    const saveToStorage = vi.fn()
    const { setAgentProvider } = useWebImportSettings(settings, saveToStorage)

    setAgentProvider('custom_openai')

    expect(settings.value.agent.provider).toBe('custom')
    expect(saveToStorage).toHaveBeenCalledTimes(1)
  })
})
