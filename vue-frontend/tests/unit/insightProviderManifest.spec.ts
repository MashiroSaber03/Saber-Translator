import { describe, expect, it } from 'vitest'

import {
  IMAGE_GEN_PROVIDER_OPTIONS,
  RERANKER_PROVIDER_OPTIONS,
  VLM_PROVIDER_OPTIONS,
} from '@/components/insight/settings/types'

describe('insight provider manifest', () => {
  it('filters image generation providers by capability', () => {
    const providers = IMAGE_GEN_PROVIDER_OPTIONS.map(option => option.value)

    expect(providers).toContain('openai')
    expect(providers).toContain('qwen')
    expect(providers).toContain('custom')
    expect(providers).not.toContain('gemini')
    expect(providers).not.toContain('deepseek')
  })

  it('filters reranker providers by capability', () => {
    const providers = RERANKER_PROVIDER_OPTIONS.map(option => option.value)

    expect(providers).toContain('qwen')
    expect(providers).toContain('siliconflow')
    expect(providers).toContain('volcano')
    expect(providers).toContain('custom')
    expect(providers).not.toContain('openai')
    expect(providers).not.toContain('gemini')
  })

  it('keeps vlm providers available while excluding rerank-only vendors', () => {
    const providers = VLM_PROVIDER_OPTIONS.map(option => option.value)

    expect(providers).toContain('openai')
    expect(providers).toContain('gemini')
    expect(providers).toContain('qwen')
    expect(providers).toContain('custom')
    expect(providers).not.toContain('jina')
    expect(providers).not.toContain('cohere')
  })
})
