import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import {
  executeAiTranslate,
  type AiTranslateOutput,
  type AiTranslateTask,
} from '@/composables/translation/core/steps/aiTranslate'
import { useSettingsStore } from '@/stores/settings'
import type { BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import type { HqTranslationProvider, ProofreadingRound } from '@/types/settings'
import { createBubbleState } from '@/utils/bubbleFactory'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import { createDefaultOpenAiOptions } from '@/utils/openaiOptions'

const { hqTranslateBatchMock } = vi.hoisted(() => ({
  hqTranslateBatchMock: vi.fn(),
}))

vi.mock('@/api/translate', () => ({
  hqTranslateBatch: hqTranslateBatchMock,
}))

const DATA_URL = 'data:image/png;base64,abc'

function createTestImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    id: 'image-1',
    fileName: 'page-1.png',
    width: 100,
    height: 100,
    originalDataURL: DATA_URL,
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    fontSize: 28,
    autoFontSize: true,
    fontFamily: 'Arial',
    layoutDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#ffffff',
    inpaintMethod: 'solid',
    strokeEnabled: false,
    strokeColor: '#ffffff',
    strokeWidth: 0,
    lineSpacing: 1,
    textAlign: 'center',
    hasUnsavedChanges: false,
    ...overrides,
  }
}

function createTranslationTask(overrides: Partial<AiTranslateTask> = {}): AiTranslateTask {
  return {
    imageIndex: 0,
    image: createTestImage(),
    originalTexts: ['こんにちは'],
    autoDirections: ['vertical'],
    ...overrides,
  }
}

function createProofreadBubble(overrides: Partial<BubbleState> = {}): BubbleState {
  return createBubbleState({
    originalText: '原文',
    translatedText: '初始译文',
    textDirection: 'vertical',
    autoTextDirection: 'vertical',
    ...overrides,
  })
}

function createProofreadingRound(
  index: number,
  overrides: Partial<ProofreadingRound> = {},
): ProofreadingRound {
  return {
    name: `第${index}轮`,
    provider: 'custom',
    apiKey: `proof-key-${index}`,
    modelName: `proof-model-${index}`,
    customBaseUrl: `https://proof-${index}.example.com/v1`,
    batchSize: 1,
    openaiOptions: createDefaultOpenAiOptions({
      request: {
        forceJsonOutput: false,
      },
      execution: {
        useStream: true,
        rpmLimit: index === 1 ? 4 : 6,
        transportRetries: 1,
        businessRetries: 0,
      },
    }),
    prompt: index === 1 ? '请校对译文' : '再次校对译文',
    ...overrides,
  }
}

function getPayload(callIndex: number): Record<string, unknown> {
  return hqTranslateBatchMock.mock.calls[callIndex]?.[0] as Record<string, unknown>
}

describe('executeAiTranslate', () => {
  const storageState: Record<string, string> = {}

  it('keeps AI translation fixtures typed to the current contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/aiTranslate.spec.ts'), 'utf8')

    expect(source).not.toContain('as ' + 'any')
    expect(source).not.toMatch(new RegExp(':\\s*' + 'any\\b'))
    expect(source).not.toContain('any' + '[]')
  })

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
    settingsStore.settings.hqTranslation.provider = '' as HqTranslationProvider
    settingsStore.settings.hqTranslation.apiKey = 'hq-key'
    settingsStore.settings.hqTranslation.modelName = 'hq-model'
    const constraints = createEmptyBookTranslationConstraints()
    constraints.glossary.enabled = true
    constraints.glossary.entries = [
      { source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' },
    ]
    constraints.non_translate.enabled = true
    constraints.non_translate.entries = [
      { pattern: '<keep>', note: '', matchMode: 'text' },
    ]
    settingsStore.settings.hqTranslation.openaiOptions.execution.rpmLimit = 13
    settingsStore.settings.hqTranslation.openaiOptions.execution.businessRetries = 0

    await executeAiTranslate({
      mode: 'hq',
      tasks: [createTranslationTask()],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: constraints,
      isBookshelfMode: true,
    })

    expect(hqTranslateBatchMock).toHaveBeenCalledTimes(1)
    expect(hqTranslateBatchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: '',
        glossary_settings: constraints.glossary,
        non_translate_settings: constraints.non_translate,
        openai_options: expect.objectContaining({
          execution: expect.objectContaining({
            rpm_limit: 13,
            business_retries: 0,
          }),
        }),
      }),
    )
    const payload = getPayload(0)
    expect(payload).not.toHaveProperty('low_reasoning')
    expect(payload).not.toHaveProperty('no_thinking_method')
  })

  it('uses per-round proofreading maxRetries including explicit zero', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.proofreading.maxRetries = 5
    settingsStore.settings.proofreading.rounds = [
      createProofreadingRound(1),
      createProofreadingRound(2),
    ]

    await executeAiTranslate({
      mode: 'proofread',
      tasks: [
        createTranslationTask({
          image: createTestImage({
            bubbleStates: [createProofreadBubble()],
          }),
        }),
      ],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      isBookshelfMode: false,
    })

    expect(hqTranslateBatchMock).toHaveBeenCalledTimes(2)
    expect(hqTranslateBatchMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        provider: 'custom',
        openai_options: expect.objectContaining({
          execution: expect.objectContaining({
            rpm_limit: 4,
            business_retries: 0,
          }),
        }),
      }),
    )
    expect(hqTranslateBatchMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        provider: 'custom',
        openai_options: expect.objectContaining({
          execution: expect.objectContaining({
            rpm_limit: 6,
            business_retries: 0,
          }),
        }),
      }),
    )
    const firstPayload = getPayload(0)
    const secondPayload = getPayload(1)
    expect(firstPayload).not.toHaveProperty('low_reasoning')
    expect(firstPayload).not.toHaveProperty('no_thinking_method')
    expect(secondPayload).not.toHaveProperty('low_reasoning')
    expect(secondPayload).not.toHaveProperty('no_thinking_method')
  })

  it('normalizes a single-image JSON response without routine console output', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.hqTranslation.apiKey = 'hq-key'
    settingsStore.settings.hqTranslation.modelName = 'hq-model'
    settingsStore.settings.hqTranslation.openaiOptions.request.forceJsonOutput = true
    hqTranslateBatchMock.mockResolvedValueOnce({
      success: true,
      content: JSON.stringify({
        imageIndex: 0,
        bubbles: [{ translated: '单图译文' }],
      }),
    })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    let result: AiTranslateOutput | undefined
    try {
      result = await executeAiTranslate({
        mode: 'hq',
        tasks: [createTranslationTask()],
        settingsSnapshot: settingsStore.settings,
        bookTranslationConstraints: createEmptyBookTranslationConstraints(),
        isBookshelfMode: false,
      })
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(result?.results).toEqual([
      {
        imageIndex: 0,
        translatedTexts: ['单图译文'],
        textboxTexts: [],
        warnings: [],
      },
    ])
  })

  it('ignores malformed HQ result payloads instead of throwing', async () => {
    const settingsStore = useSettingsStore()
    settingsStore.settings.hqTranslation.apiKey = 'hq-key'
    settingsStore.settings.hqTranslation.modelName = 'hq-model'
    hqTranslateBatchMock.mockResolvedValueOnce({
      success: true,
      results: [
        {
          imageIndex: 0,
          bubbles: 'not-an-array',
        },
      ],
    })

    const result = await executeAiTranslate({
      mode: 'hq',
      tasks: [createTranslationTask()],
      settingsSnapshot: settingsStore.settings,
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      isBookshelfMode: false,
    })

    expect(result.results).toEqual([
      {
        imageIndex: 0,
        translatedTexts: [''],
        textboxTexts: [],
        warnings: [],
      },
    ])
  })
})
