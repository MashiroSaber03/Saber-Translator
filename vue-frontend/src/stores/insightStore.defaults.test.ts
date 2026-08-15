import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useInsightStore } from './insightStore'

describe('useInsightStore factory defaults', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('uses the expected source defaults for insight configuration', () => {
    const store = useInsightStore()

    expect(store.config.vlm.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.config.vlm.openaiOptions.execution.transportRetries).toBe(1)
    expect(store.config.vlm.openaiOptions.execution.businessRetries).toBe(0)
    expect(store.config.vlm.imageMaxSize).toBe(0)

    expect(store.config.llm.useSameAsVlm).toBe(false)
    expect(store.config.llm.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.config.llm.openaiOptions.execution.transportRetries).toBe(1)
    expect(store.config.llm.openaiOptions.execution.businessRetries).toBe(0)

    expect(store.config.embedding.transportRetries).toBe(1)
    expect(store.config.embedding.businessRetries).toBe(0)
    expect(store.config.embedding.timeoutSeconds).toBe(0)
    expect(store.config.reranker.transportRetries).toBe(1)
    expect(store.config.reranker.businessRetries).toBe(0)
    expect(store.config.reranker.timeoutSeconds).toBe(0)
    expect(store.config.imageGen.transportRetries).toBe(1)
    expect(store.config.imageGen.businessRetries).toBe(0)
    expect(store.config.imageGen.timeoutSeconds).toBe(0)

    expect(store.config.batch.contextBatchCount).toBe(3)
  })

  it('preserves an explicit zero imageMaxSize from API payloads', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()
    snapshot.config.vlm.imageMaxSize = 0

    store.setConfigFromApi(snapshot)

    expect(store.config.vlm.imageMaxSize).toBe(0)
  })

  it('preserves a zero-unit custom summary layer from API payloads', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()
    snapshot.config.batch = {
      pagesPerBatch: 5,
      contextBatchCount: 3,
      architecturePreset: 'custom',
      customLayers: [
        { name: '批量分析', units: 5, align: false },
        { name: '全书总结', units: 0, align: false },
      ],
    }

    store.setConfigFromApi(snapshot)

    expect(store.config.batch.customLayers[1]).toEqual({
      name: '全书总结',
      units: 0,
      align: false,
    })
  })

  it('maps reranker and imageGen runtime settings from API payloads', () => {
    const store = useInsightStore()
    const snapshot = store.getConfigForApi()
    snapshot.config.reranker = {
      provider: 'jina',
      apiKey: 'rerank-key',
      model: 'jina-reranker-v2-base-multilingual',
      baseUrl: 'https://rerank.example.com/v1',
      transportRetries: 4,
      businessRetries: 5,
      timeoutSeconds: 0,
    }
    snapshot.config.imageGen = {
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://image.example.com/v1',
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    }

    store.setConfigFromApi(snapshot)

    expect(store.config.reranker.transportRetries).toBe(4)
    expect(store.config.reranker.businessRetries).toBe(5)
    expect(store.config.reranker.timeoutSeconds).toBe(0)
    expect(store.config.imageGen.transportRetries).toBe(7)
    expect(store.config.imageGen.businessRetries).toBe(8)
    expect(store.config.imageGen.timeoutSeconds).toBe(9)
  })
})
