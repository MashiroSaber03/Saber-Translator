import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useInsightStore } from './insightStore'
import {
  normalizeInsightImageGenConfig,
  normalizeInsightRerankerConfig,
} from './insight/insightConfigDefaults'

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

  it('preserves explicit zero business retries in settings snapshots', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()
    snapshot.config.imageGen = {
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://image.example.com/v1',
      transportRetries: 10,
      businessRetries: 0,
      timeoutSeconds: 0,
    }

    store.setConfigFromApi(snapshot)

    expect(store.config.imageGen.businessRetries).toBe(0)
  })

  it('normalizes provider defaults through the insight config defaults helper', () => {
    expect(normalizeInsightRerankerConfig()).toMatchObject({
      provider: 'jina',
      model: 'jina-reranker-v2-base-multilingual',
      transportRetries: 1,
      businessRetries: 0,
      timeoutSeconds: 0,
    })

    expect(normalizeInsightImageGenConfig({ provider: 'newapi' })).toMatchObject({
      provider: 'newapi',
      model: '',
      baseUrl: '',
      transportRetries: 1,
      businessRetries: 0,
      timeoutSeconds: 0,
    })
  })

  it('round-trips current provider drafts without a second transport shape', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()
    snapshot.providerDrafts = {
      vlm: {
        gemini: {
          apiKey: 'vlm-key',
          model: 'gemini-2.0-flash',
          baseUrl: '',
          openaiOptions: {
            request: { forceJsonOutput: true, temperature: 0.2 },
            execution: {
              useStream: false,
              rpmLimit: 11,
              transportRetries: 2,
              businessRetries: 3,
            },
          },
          imageMaxSize: 1024,
        },
      },
      llm: {},
      embedding: {},
      reranker: {
        jina: {
          apiKey: 'reranker-key',
          model: 'jina-reranker-v2-base-multilingual',
          baseUrl: '',
          transportRetries: 10,
          businessRetries: 10,
          timeoutSeconds: 0,
        },
      },
      imageGen: {
        gpt2api: {
          apiKey: 'image-key',
          model: 'gpt-image-2',
          baseUrl: '',
          transportRetries: 10,
          businessRetries: 0,
          timeoutSeconds: 0,
        },
      },
    }
    store.setConfigFromApi(snapshot)
    const restored = store.getConfigForApi().providerDrafts

    expect(restored.vlm.gemini?.openaiOptions.request.forceJsonOutput).toBe(true)
    expect(restored.vlm.gemini?.openaiOptions.execution.rpmLimit).toBe(11)
    expect(restored.vlm.gemini?.imageMaxSize).toBe(1024)
    expect(restored.reranker.jina?.transportRetries).toBe(10)
    expect(restored.imageGen.gpt2api?.businessRetries).toBe(0)
  })

  it('preserves an explicitly empty cached base URL when switching back to a provider', () => {
    const store = useInsightStore()
    store.updateImageGenConfig({
      ...store.config.imageGen,
      baseUrl: '',
    })

    store.switchImageGenProviderDraft({ ...store.config.imageGen, provider: 'newapi' })
    store.switchImageGenProviderDraft({ ...store.config.imageGen, provider: 'gpt2api' })

    expect(store.config.imageGen.baseUrl).toBe('')
  })

  it('does not retain references owned by an applied settings snapshot', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()

    store.setConfigFromApi(snapshot)
    snapshot.config.imageGen.model = 'mutated-after-apply'

    expect(store.config.imageGen.model).not.toBe('mutated-after-apply')
  })
})
