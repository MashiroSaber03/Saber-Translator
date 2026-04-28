import { describe, expect, it } from 'vitest'

import {
  getProviderOptionsForCapability,
  getProviderDefaultModel,
  providerSupportsRpmLimit,
  normalizeProviderId,
  providerRequiresBaseUrl,
  providerSupportsCapability
} from '@/config/aiProviders'

describe('translation page AI provider manifest', () => {
  it('normalizes legacy custom provider ids to custom', () => {
    expect(normalizeProviderId('custom_openai')).toBe('custom')
    expect(normalizeProviderId('custom_openai_vision')).toBe('custom')
    expect(normalizeProviderId('custom')).toBe('custom')
  })

  it('derives provider capabilities from a shared manifest', () => {
    expect(providerSupportsCapability('custom', 'translation')).toBe(true)
    expect(providerSupportsCapability('custom', 'hqTranslation')).toBe(true)
    expect(providerSupportsCapability('custom', 'visionOcr')).toBe(true)
    expect(providerRequiresBaseUrl('custom')).toBe(true)
  })

  it('does not expose deepseek in AI vision OCR options', () => {
    const options = getProviderOptionsForCapability('visionOcr')
    expect(options.map(option => option.value)).not.toContain('deepseek')
    expect(options.map(option => option.value)).toContain('custom')
  })

  it('derives RPM-limit visibility from the shared provider manifest', () => {
    expect(providerSupportsRpmLimit('siliconflow')).toBe(true)
    expect(providerSupportsRpmLimit('custom')).toBe(true)
    expect(providerSupportsRpmLimit('ollama')).toBe(false)
    expect(providerSupportsRpmLimit('caiyun')).toBe(false)
  })

  it('keeps frontend default chat models aligned with the shared manifest contract', () => {
    expect(getProviderDefaultModel('openai', 'chat')).toBe('gpt-4o')
    expect(getProviderDefaultModel('qwen', 'chat')).toBe('qwen-plus')
  })
})
