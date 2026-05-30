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
})
