import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { executeAiTranslate } from '@/composables/translation/core/steps/aiTranslate'
import { useSettingsStore } from '@/stores/settingsStore'

const { hqTranslateBatchMock } = vi.hoisted(() => ({
  hqTranslateBatchMock: vi.fn(),
}))

vi.mock('@/api/translate', () => ({
  hqTranslateBatch: hqTranslateBatchMock,
}))

describe('executeAiTranslate', () => {
  const storageState: Record<string, string> = {}

  beforeEach(() => {
    setActivePinia(createPinia())
    for (const key of Object.keys(storageState)) {
      delete storageState[key]
    }

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => storageState[key] ?? null)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      storageState[key] = value
    })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete storageState[key]
    })

    hqTranslateBatchMock.mockReset()
    hqTranslateBatchMock.mockResolvedValue({
      success: true,
      results: [
        {
          imageIndex: 0,
          bubbles: [{ translated: '译文' }],
        },
      ],
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('preserves empty provider and zero max retries for HQ translation requests', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.hqTranslation.provider = '' as any
    settingsStore.settings.hqTranslation.apiKey = 'hq-key'
    settingsStore.settings.hqTranslation.modelName = 'hq-model'
    settingsStore.settings.hqTranslation.maxRetries = 0

    await executeAiTranslate({
      mode: 'hq',
      tasks: [
        {
          imageIndex: 0,
          image: {
            originalDataURL: 'data:image/png;base64,abc',
          } as any,
          originalTexts: ['こんにちは'],
          autoDirections: ['vertical'],
        },
      ],
    })

    expect(hqTranslateBatchMock).toHaveBeenCalledTimes(1)
    expect(hqTranslateBatchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: '',
        max_retries: 0,
      }),
    )
  })

  it('uses per-round proofreading maxRetries including explicit zero', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.proofreading.maxRetries = 5
    settingsStore.settings.proofreading.rounds = [
      {
        name: '第1轮',
        provider: 'custom',
        apiKey: 'proof-key-1',
        modelName: 'proof-model-1',
        customBaseUrl: 'https://proof-1.example.com/v1',
        batchSize: 1,
        sessionReset: 1,
        rpmLimit: 0,
        maxRetries: 0,
        lowReasoning: false,
        noThinkingMethod: 'gemini',
        forceJsonOutput: false,
        useStream: true,
        prompt: '请校对译文',
      },
      {
        name: '第2轮',
        provider: 'custom',
        apiKey: 'proof-key-2',
        modelName: 'proof-model-2',
        customBaseUrl: 'https://proof-2.example.com/v1',
        batchSize: 1,
        sessionReset: 1,
        rpmLimit: 0,
        maxRetries: 0,
        lowReasoning: false,
        noThinkingMethod: 'gemini',
        forceJsonOutput: false,
        useStream: true,
        prompt: '再次校对译文',
      },
    ] as any

    await executeAiTranslate({
      mode: 'proofread',
      tasks: [
        {
          imageIndex: 0,
          image: {
            originalDataURL: 'data:image/png;base64,abc',
            translatedDataURL: null,
            bubbleStates: [
              {
                originalText: '原文',
                translatedText: '初始译文',
                textDirection: 'vertical',
                autoTextDirection: 'vertical',
              },
            ],
          } as any,
        },
      ],
    })

    expect(hqTranslateBatchMock).toHaveBeenCalledTimes(2)
    expect(hqTranslateBatchMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        provider: 'custom',
        max_retries: 0,
      }),
    )
    expect(hqTranslateBatchMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        provider: 'custom',
        max_retries: 0,
      }),
    )
  })
})
