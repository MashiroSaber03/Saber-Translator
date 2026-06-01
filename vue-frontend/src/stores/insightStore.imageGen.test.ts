import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useInsightStore } from './insightStore'
import { useInsightConfigManager, type ProviderConfigsCache } from './insight/useInsightConfigManager'
import { ref } from 'vue'

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

  it('migrates legacy local imageGen maxRetries into businessRetries', () => {
    localStorage.setItem('manga_insight_config', JSON.stringify({
      imageGen: {
        provider: 'gpt2api',
        apiKey: 'legacy-key',
        model: 'gpt-image-2',
        baseUrl: 'https://legacy.example.com/v1',
        maxRetries: 6,
      },
    }))

    const store = useInsightStore()
    store.loadConfigFromStorage()

    expect(store.config.imageGen.transportRetries).toBe(10)
    expect(store.config.imageGen.businessRetries).toBe(6)
    expect(store.config.imageGen.timeoutSeconds).toBe(0)
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

    store.setImageGenProvider('future-image-provider')
    store.updateImageGenConfig({
      provider: 'future-image-provider',
      apiKey: 'second-key',
      model: 'future-image-model',
      baseUrl: 'https://second.example.com/v1',
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })
    store.setImageGenProvider('gpt2api')

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

    store.setImageGenProvider('newapi')

    expect(store.config.imageGen.provider).toBe('newapi')
    expect(store.config.imageGen.model).toBe('')
    expect(store.config.imageGen.baseUrl).toBe('')
  })

  it('fills missing reranker runtime fields when loading legacy provider cache', () => {
    const providerConfigs = ref<ProviderConfigsCache>({
      vlm: {},
      llm: {},
      embedding: {},
      reranker: {},
      imageGen: {},
    })
    localStorage.setItem('insight_provider_configs', JSON.stringify({
      reranker: {
        jina: {
          apiKey: 'legacy-key',
          model: 'jina-reranker-v2-base-multilingual',
          baseUrl: 'https://rerank.example.com/v1',
          topK: 5,
        },
      },
    }))

    const manager = useInsightConfigManager(providerConfigs)
    manager.loadFromStorage()

    expect(providerConfigs.value.reranker.jina).toEqual({
      apiKey: 'legacy-key',
      model: 'jina-reranker-v2-base-multilingual',
      baseUrl: 'https://rerank.example.com/v1',
      topK: 5,
      transportRetries: 10,
      businessRetries: 10,
      timeoutSeconds: 0,
    })
  })

  it('preserves legacy zero max_retries when mapping imageGen API payloads', () => {
    const store = useInsightStore()

    store.setConfigFromApi({
      image_gen: {
        provider: 'gpt2api',
        api_key: 'image-key',
        model: 'gpt-image-2',
        base_url: 'https://image.example.com/v1',
        max_retries: 0,
      },
      providerSettings: {
        imageGenProvider: {
          gpt2api: {
            api_key: 'image-key',
            model: 'gpt-image-2',
            base_url: 'https://image.example.com/v1',
            max_retries: 0,
          },
        },
      },
    })

    expect(store.config.imageGen.businessRetries).toBe(0)
  })
})

