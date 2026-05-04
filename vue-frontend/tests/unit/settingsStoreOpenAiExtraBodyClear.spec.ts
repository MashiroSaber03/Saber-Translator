import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settingsStore'

describe('settings store clears OpenAI extraBody', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('clears translation extraBody and does not restore it after reload', () => {
    const store = useSettingsStore()
    store.updateTranslationService({
      extraBody: {
        thinking: {
          type: 'disabled',
        },
      },
    })

    store.updateTranslationService({
      extraBody: undefined,
    })

    expect(store.settings.translation.openaiOptions.request.extraBody).toBeUndefined()

    setActivePinia(createPinia())
    const reloadedStore = useSettingsStore()
    reloadedStore.loadFromStorage()

    expect(reloadedStore.settings.translation.openaiOptions.request.extraBody).toBeUndefined()
  })

  it('clears HQ translation, AI vision OCR, and proofreading extraBody values', () => {
    const store = useSettingsStore()

    store.updateHqTranslation({
      extraBody: {
        thinking: {
          type: 'disabled',
        },
      },
    })
    store.updateAiVisionOcr({
      extraBody: {
        thinking: {
          type: 'disabled',
        },
      },
    })
    store.settings.proofreading.rounds = [
      {
        name: '第1轮',
        provider: 'siliconflow',
        apiKey: 'proof-key',
        modelName: 'proof-model',
        customBaseUrl: '',
        prompt: 'proof',
        batchSize: 2,
        openaiOptions: {
          request: {
            forceJsonOutput: false,
            extraBody: {
              thinking: {
                type: 'disabled',
              },
            },
          },
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 1,
            businessRetries: 1,
          },
        },
      },
    ]

    store.updateHqTranslation({ extraBody: undefined })
    store.updateAiVisionOcr({ extraBody: undefined })
    store.updateProofreadingRound(0, { extraBody: undefined })

    expect(store.settings.hqTranslation.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.settings.aiVisionOcr.openaiOptions.request.extraBody).toBeUndefined()
    expect(store.settings.proofreading.rounds[0]?.openaiOptions.request.extraBody).toBeUndefined()
  })
})
