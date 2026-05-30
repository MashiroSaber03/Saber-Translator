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
    expect(store.config.vlm.openaiOptions.execution.transportRetries).toBe(10)
    expect(store.config.vlm.openaiOptions.execution.businessRetries).toBe(10)
    expect(store.config.vlm.imageMaxSize).toBe(1280)

    expect(store.config.llm.useSameAsVlm).toBe(false)
    expect(store.config.llm.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.config.llm.openaiOptions.execution.transportRetries).toBe(10)
    expect(store.config.llm.openaiOptions.execution.businessRetries).toBe(10)

    expect(store.config.embedding.transportRetries).toBe(10)
    expect(store.config.embedding.businessRetries).toBe(10)
    expect(store.config.embedding.timeoutSeconds).toBe(0)
    expect(store.config.reranker.transportRetries).toBe(10)
    expect(store.config.reranker.businessRetries).toBe(10)
    expect(store.config.reranker.timeoutSeconds).toBe(0)
    expect(store.config.imageGen.transportRetries).toBe(10)
    expect(store.config.imageGen.businessRetries).toBe(10)
    expect(store.config.imageGen.timeoutSeconds).toBe(0)

    expect(store.config.batch.contextBatchCount).toBe(3)
  })

  it('preserves an explicit zero imageMaxSize from API payloads', () => {
    const store = useInsightStore()

    store.setConfigFromApi({
      vlm: {
        provider: 'gemini',
        api_key: '',
        model: 'gemini-2.0-flash',
        base_url: '',
        openai_options: {
          request: {
            force_json_output: false,
            temperature: 0.3,
          },
          execution: {
            use_stream: true,
            rpm_limit: 0,
            transport_retries: 10,
            business_retries: 10,
          },
        },
        image_max_size: 0,
      },
    })

    expect(store.config.vlm.imageMaxSize).toBe(0)
  })

  it('maps reranker and imageGen runtime settings from API payloads', () => {
    const store = useInsightStore()

    store.setConfigFromApi({
      reranker: {
        provider: 'jina',
        api_key: 'rerank-key',
        model: 'jina-reranker-v2-base-multilingual',
        base_url: 'https://rerank.example.com/v1',
        top_k: 6,
        transport_retries: 4,
        business_retries: 5,
        timeout_seconds: 0,
      },
      image_gen: {
        provider: 'gpt2api',
        api_key: 'image-key',
        model: 'gpt-image-2',
        base_url: 'https://image.example.com/v1',
        transport_retries: 7,
        business_retries: 8,
        timeout_seconds: 9,
      },
    })

    expect(store.config.reranker.transportRetries).toBe(4)
    expect(store.config.reranker.businessRetries).toBe(5)
    expect(store.config.reranker.timeoutSeconds).toBe(0)
    expect(store.config.imageGen.transportRetries).toBe(7)
    expect(store.config.imageGen.businessRetries).toBe(8)
    expect(store.config.imageGen.timeoutSeconds).toBe(9)
  })
})
