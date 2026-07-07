import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useInsightStore } from '@/stores/insightStore'
import {
  buildInsightConfigStoragePayload,
  parseInsightConfigStorage,
} from '@/stores/insight/insightConfigStorage'

describe('insight store clears OpenAI extraBody', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('ignores unversioned local config from storage', () => {
    localStorage.setItem('manga_insight_config', JSON.stringify({
      vlm: {
        provider: 'ignored-top-level-provider',
        apiKey: 'ignored-top-level-key',
        model: 'ignored-top-level-model',
        baseUrl: 'https://ignored.example.com/v1',
      },
    }))
    const store = useInsightStore()

    store.loadConfigFromStorage()

    expect(store.config.vlm).toMatchObject({
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
    })
  })

  it('ignores versioned local config when the current schema is incomplete', () => {
    localStorage.setItem('manga_insight_config', JSON.stringify({
      insightConfigSchemaVersion: 1,
      config: {
        vlm: {
          provider: 'custom',
          apiKey: 'partial-key',
          model: 'partial-model',
          baseUrl: 'https://partial.example.com/v1',
        },
      },
    }))
    const store = useInsightStore()

    store.loadConfigFromStorage()

    expect(store.config.vlm).toMatchObject({
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
    })
  })

  it('clears VLM and LLM extraBody and does not restore them after reload', () => {
    const store = useInsightStore()

    store.updateVlmConfig({
      openaiOptions: {
        request: {
          ...store.config.vlm.openaiOptions.request,
          extraBody: {
            thinking: {
              type: 'disabled',
            },
          },
        },
        execution: { ...store.config.vlm.openaiOptions.execution },
      },
    })
    store.updateLlmConfig({
      openaiOptions: {
        request: {
          ...store.config.llm.openaiOptions.request,
          extraBody: {
            thinking: {
              type: 'disabled',
            },
          },
        },
        execution: { ...store.config.llm.openaiOptions.execution },
      },
    })

    store.updateVlmConfig({
      openaiOptions: {
        request: {
          ...store.config.vlm.openaiOptions.request,
          extraBody: undefined,
        },
        execution: { ...store.config.vlm.openaiOptions.execution },
      },
    })
    store.updateLlmConfig({
      openaiOptions: {
        request: {
          ...store.config.llm.openaiOptions.request,
          extraBody: undefined,
        },
        execution: { ...store.config.llm.openaiOptions.execution },
      },
    })

    expect(store.config.vlm.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.config.llm.openaiOptions.request.extraBody).toBeUndefined()

    setActivePinia(createPinia())
    const reloadedStore = useInsightStore()
    reloadedStore.loadConfigFromStorage()

    expect(reloadedStore.config.vlm.openaiOptions.request.extraBody).toBeUndefined()
    expect(reloadedStore.config.llm.openaiOptions.request.extraBody).toBeUndefined()
  })

  it('ignores top-level OpenAI mirror fields on VLM and LLM updates', () => {
    const store = useInsightStore()
    const topLevelVlmPatch = {
      extraBody: { stale: true },
      rpmLimit: 77,
      temperature: 0.9,
      useStream: false,
    } as unknown as Parameters<typeof store.updateVlmConfig>[0]
    const topLevelLlmPatch = {
      extraBody: { stale: true },
      useStream: false,
    } as unknown as Parameters<typeof store.updateLlmConfig>[0]

    store.updateVlmConfig(topLevelVlmPatch)
    store.updateLlmConfig(topLevelLlmPatch)

    expect(store.config.vlm.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.config.vlm.openaiOptions.request.temperature).toBe(0.3)
    expect(store.config.vlm.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.config.vlm.openaiOptions.execution.useStream).toBe(true)
    expect(store.config.llm.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.config.llm.openaiOptions.execution.useStream).toBe(true)
  })

  it('parses only current insight config storage payloads through the storage helper', () => {
    const store = useInsightStore()
    const payload = buildInsightConfigStoragePayload(store.config)

    expect(parseInsightConfigStorage({ config: store.config })).toBeNull()
    expect(parseInsightConfigStorage({ insightConfigSchemaVersion: 1, config: { vlm: store.config.vlm } })).toBeNull()
    expect(parseInsightConfigStorage(payload)).toEqual(store.config)
  })
})
