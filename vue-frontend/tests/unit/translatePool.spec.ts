import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  hqTranslateBatchMock,
  settingsStoreMock,
} = vi.hoisted(() => ({
  hqTranslateBatchMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      enableVerboseLogs: false,
      useTextboxPrompt: false,
      hqTranslation: {
        provider: 'custom',
        apiKey: 'hq-key',
        modelName: 'hq-model',
        customBaseUrl: 'https://hq.example.com/v1',
        prompt: '高质量翻译提示词',
        lowReasoning: false,
        forceJsonOutput: false,
        noThinkingMethod: 'gemini',
        useStream: true,
        maxRetries: 0,
        batchSize: 1,
      },
      proofreading: {
        maxRetries: 5,
        rounds: [
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
        ],
      },
    },
  },
}))

vi.mock('@/api/translate', () => ({
  hqTranslateBatch: hqTranslateBatchMock,
  translateSingleText: vi.fn(),
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelTranslate: vi.fn(),
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

describe('TranslatePool retry mapping', () => {
  beforeEach(() => {
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

  it('preserves zero max retries for HQ batches', async () => {
    const { TranslatePool } = await import('@/composables/translation/parallel/pools/TranslatePool')

    const pool = new TranslatePool(
      null,
      { updatePool: vi.fn() } as any,
    )
    pool.setMode('hq', 1, null)

    const task = {
      id: 'task-1',
      imageIndex: 0,
      status: 'pending',
      imageData: {
        originalDataURL: 'data:image/png;base64,abc',
      },
      ocrResult: {
        originalTexts: ['原文'],
      },
      detectionResult: {
        autoDirections: ['vertical'],
      },
    } as any

    await (pool as any).process(task)

    expect(hqTranslateBatchMock).toHaveBeenCalledOnce()
    expect(hqTranslateBatchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'custom',
        max_retries: 0,
      }),
    )
  })

  it('preserves zero max retries for every proofreading round', async () => {
    const { TranslatePool } = await import('@/composables/translation/parallel/pools/TranslatePool')

    const pool = new TranslatePool(
      null,
      { updatePool: vi.fn() } as any,
    )
    pool.setMode('proofread', 1, null)

    const task = {
      id: 'task-2',
      imageIndex: 0,
      status: 'pending',
      imageData: {
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
      },
    } as any

    await (pool as any).process(task)

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
