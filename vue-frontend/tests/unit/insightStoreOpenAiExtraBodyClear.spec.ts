import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useInsightStore } from '@/stores/insightStore'

describe('insight store clears OpenAI extraBody', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('clears VLM and LLM extraBody and does not restore them after reload', () => {
    const store = useInsightStore()

    store.updateVlmConfig({
      extraBody: {
        thinking: {
          type: 'disabled',
        },
      },
    })
    store.updateLlmConfig({
      extraBody: {
        thinking: {
          type: 'disabled',
        },
      },
    })

    store.updateVlmConfig({ extraBody: undefined })
    store.updateLlmConfig({ extraBody: undefined })

    expect(store.config.vlm.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.config.llm.openaiOptions.request.extraBody).toBeUndefined()

    setActivePinia(createPinia())
    const reloadedStore = useInsightStore()
    reloadedStore.loadConfigFromStorage()

    expect(reloadedStore.config.vlm.openaiOptions.request.extraBody).toBeUndefined()
    expect(reloadedStore.config.llm.openaiOptions.request.extraBody).toBeUndefined()
  })
})
