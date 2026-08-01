import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useInsightStore } from './insightStore'
import type { ProviderConfigsCache } from './insight/useInsightConfigManager'
import {
  normalizeInsightImageGenConfig,
  normalizeInsightRerankerConfig,
} from './insight/insightConfigDefaults'
import { applyInsightProviderSettingsFromApi } from './insight/insightProviderSettingsHydration'

describe('useInsightStore imageGen config', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('preserves an explicitly selected image generation provider instead of forcing gpt2api', () => {
    const store = useInsightStore()

    store.updateImageGenConfig({
      provider: 'future-image-provider',
      apiKey: 'future-key',
      model: 'future-image-model',
      baseUrl: 'https://future.example.com/v1',
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })

    expect(store.config.imageGen).toEqual({
      provider: 'future-image-provider',
      apiKey: 'future-key',
      model: 'future-image-model',
      baseUrl: 'https://future.example.com/v1',
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })
  })

  it('restores per-provider imageGen runtime settings when switching providers', () => {
    const store = useInsightStore()

    store.updateImageGenConfig({
      provider: 'gpt2api',
      apiKey: 'first-key',
      model: 'gpt-image-2',
      baseUrl: 'https://first.example.com/v1',
      transportRetries: 4,
      businessRetries: 5,
      timeoutSeconds: 6,
    })

    store.switchImageGenProviderDraft({
      ...store.config.imageGen,
      provider: 'future-image-provider',
    })
    store.updateImageGenConfig({
      provider: 'future-image-provider',
      apiKey: 'second-key',
      model: 'future-image-model',
      baseUrl: 'https://second.example.com/v1',
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })
    store.switchImageGenProviderDraft({
      ...store.config.imageGen,
      provider: 'gpt2api',
    })

    expect(store.config.imageGen).toEqual({
      provider: 'gpt2api',
      apiKey: 'first-key',
      model: 'gpt-image-2',
      baseUrl: 'https://first.example.com/v1',
      transportRetries: 4,
      businessRetries: 5,
      timeoutSeconds: 6,
    })
  })

  it('does not backfill gpt-image-2 when switching to newapi without a cached model', () => {
    const store = useInsightStore()

    store.switchImageGenProviderDraft({
      ...store.config.imageGen,
      provider: 'newapi',
    })

    expect(store.config.imageGen.provider).toBe('newapi')
    expect(store.config.imageGen.model).toBe('')
    expect(store.config.imageGen.baseUrl).toBe('')
  })

  it('preserves explicit zero business_retries when mapping imageGen API payloads', () => {
    const store = useInsightStore()

    store.setConfigFromApi({
      image_gen: {
        provider: 'gpt2api',
        api_key: 'image-key',
        model: 'gpt-image-2',
        base_url: 'https://image.example.com/v1',
        business_retries: 0,
      },
      provider_settings: {
        imageGenProvider: {
          gpt2api: {
            api_key: 'image-key',
            model: 'gpt-image-2',
            base_url: 'https://image.example.com/v1',
            business_retries: 0,
          },
        },
      },
    })

    expect(store.config.imageGen.businessRetries).toBe(0)
  })

  it('normalizes provider defaults through the insight config defaults helper', () => {
    expect(normalizeInsightRerankerConfig()).toMatchObject({
      provider: 'jina',
      model: 'jina-reranker-v2-base-multilingual',
      topK: 5,
      transportRetries: 10,
      businessRetries: 10,
      timeoutSeconds: 0,
    })

    expect(normalizeInsightImageGenConfig({ provider: 'newapi' })).toMatchObject({
      provider: 'newapi',
      model: '',
      baseUrl: '',
      transportRetries: 10,
      businessRetries: 10,
      timeoutSeconds: 0,
    })
  })

  it('hydrates provider settings through the insight provider settings helper', () => {
    const providerConfigs: ProviderConfigsCache = {
      vlm: {},
      llm: {},
      embedding: {},
      reranker: {},
      imageGen: {},
    }

    expect(applyInsightProviderSettingsFromApi(providerConfigs, {
      vlmProvider: {
        gemini: {
          api_key: 'vlm-key',
          model: 'gemini-2.0-flash',
          base_url: '',
          openai_options: {
            request: { force_json_output: true, temperature: 0.2 },
            execution: {
              use_stream: false,
              rpm_limit: 11,
              transport_retries: 2,
              business_retries: 3,
            },
          },
          image_max_size: 1024,
        },
      },
      rerankerProvider: {
        jina: {
          api_key: 'reranker-key',
          model: 'jina-reranker-v2-base-multilingual',
          top_k: 0,
        },
      },
      imageGenProvider: {
        gpt2api: {
          api_key: 'image-key',
          model: 'gpt-image-2',
          business_retries: 0,
        },
      },
    })).toBe(true)

    expect(providerConfigs.vlm.gemini?.openaiOptions?.request.forceJsonOutput).toBe(true)
    expect(providerConfigs.vlm.gemini?.openaiOptions?.execution.rpmLimit).toBe(11)
    expect(providerConfigs.vlm.gemini?.imageMaxSize).toBe(1024)
    expect(providerConfigs.reranker.jina?.topK).toBe(0)
    expect(providerConfigs.imageGen.gpt2api?.businessRetries).toBe(0)
  })
})

