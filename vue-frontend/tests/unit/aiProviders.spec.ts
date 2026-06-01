import { describe, expect, it } from 'vitest'

import {
  AI_PROVIDER_MANIFEST,
  getProviderOptionsForCapability,
  getProviderDefaultModel,
  providerSupportsRpmLimit,
  normalizeProviderId,
  providerRequiresBaseUrl,
  providerRequiresModel,
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
    expect(providerSupportsRpmLimit('ollama')).toBe(true)
    expect(providerSupportsRpmLimit('caiyun')).toBe(false)
  })

  it('exposes ollama for HQ translation and AI vision OCR once declared in the shared manifest', () => {
    expect(getProviderOptionsForCapability('hqTranslation').map(option => option.value)).toContain('ollama')
    expect(getProviderOptionsForCapability('visionOcr').map(option => option.value)).toContain('ollama')
  })

  it('keeps frontend default chat models aligned with the shared manifest contract', () => {
    expect(getProviderDefaultModel('openai', 'chat')).toBe('gpt-4o')
    expect(getProviderDefaultModel('qwen', 'chat')).toBe('qwen-plus')
  })

  it('treats gpt2api as a base-url-driven image generation adapter', () => {
    expect(providerSupportsCapability('gpt2api', 'imageGen')).toBe(true)
    expect(providerSupportsCapability('gpt2api', 'modelFetch')).toBe(false)
    expect(providerRequiresBaseUrl('gpt2api')).toBe(true)
  })

  it('treats newapi as a base-url-driven image generation adapter without a default model', () => {
    expect(providerSupportsCapability('newapi', 'imageGen')).toBe(true)
    expect(providerSupportsCapability('newapi', 'modelFetch')).toBe(false)
    expect(providerRequiresBaseUrl('newapi')).toBe(true)
    expect(providerRequiresModel('newapi')).toBe(true)
    expect(getProviderDefaultModel('newapi', 'imageGen')).toBe('')
  })

  it('does not expose removed reasoning-control manifest fields', () => {
    for (const entry of AI_PROVIDER_MANIFEST) {
      expect(entry).not.toHaveProperty('supportsReasoningControl')
    }
  })
})
