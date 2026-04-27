import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { createBubbleState, createBubbleStatesFromResponse } from '@/utils/bubbleFactory'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { executeOcr } from '@/composables/translation/core/steps/ocr'

const { parallelOcrMock } = vi.hoisted(() => ({
  parallelOcrMock: vi.fn(async () => ({
    success: true,
    original_texts: ['こんにちは'],
    ocr_results: [
      {
        text: 'こんにちは',
        confidence: 0.82,
        confidenceSupported: true,
        engine: 'manga_ocr',
        primaryEngine: 'manga_ocr',
        fallbackUsed: false
      }
    ]
  }))
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => ({
    settings: {
      ocrEngine: 'manga_ocr',
      sourceLanguage: 'japanese',
      baiduOcr: {},
      paddleOcrVl: { sourceLanguage: 'japanese' },
      aiVisionOcr: {
        provider: 'custom_openai_vision',
        apiKey: 'vision-key',
        modelName: 'vision-model',
        prompt: 'ocr prompt',
        customBaseUrl: 'https://vision.example.com/v1',
        isJsonMode: true
      },
      hybridOcr: {
        enabled: true,
        secondaryEngine: '48px_ocr',
        confidenceThreshold: 0.2
      }
    }
  })
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelOcr: parallelOcrMock
}))

describe('OCR result integration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    parallelOcrMock.mockClear()
  })

  it('createBubbleState should initialize ocrResult as null', () => {
    const state = createBubbleState()
    expect(state.ocrResult).toBeNull()
    expect(state.textlines).toEqual([])
  })

  it('createBubbleStatesFromResponse should hydrate ocrResult and keep originalText in sync', () => {
    const states = createBubbleStatesFromResponse({
      bubble_coords: [[0, 0, 100, 100]],
      ocr_results: [
        {
          text: 'テスト',
          confidence: 0.73,
          confidenceSupported: true,
          engine: 'manga_ocr',
          primaryEngine: 'manga_ocr',
          fallbackUsed: false
        }
      ],
      textlines_per_bubble: [
        [
          {
            polygon: [[0, 0], [10, 0], [10, 20], [0, 20]],
            direction: 'v',
            confidence: 0.5
          }
        ]
      ]
    } as any)

    expect(states).toHaveLength(1)
    expect(states[0]?.originalText).toBe('テスト')
    expect(states[0]?.ocrResult?.confidence).toBe(0.73)
    expect(states[0]?.ocrResult?.engine).toBe('manga_ocr')
    expect(states[0]?.textlines).toEqual([
      {
        polygon: [[0, 0], [10, 0], [10, 20], [0, 20]],
        direction: 'v',
        confidence: 0.5
      }
    ])
  })

  it('createBubbleStatesFromResponse should fall back to legacy textlines when bubble state textlines are empty', () => {
    const states = createBubbleStatesFromResponse({
      bubble_coords: [[0, 0, 100, 100]],
      bubble_states: [
        {
          coords: [0, 0, 100, 100],
          textlines: []
        }
      ],
      textlines_per_bubble: [
        [
          {
            polygon: [[1, 1], [9, 1], [9, 9], [1, 9]],
            direction: 'h',
            confidence: 0.4
          }
        ]
      ]
    } as any)

    expect(states[0]?.textlines).toEqual([
      {
        polygon: [[1, 1], [9, 1], [9, 9], [1, 9]],
        direction: 'h',
        confidence: 0.4
      }
    ])
  })

  it('executeOcr should return both ocrResults and legacy originalTexts', async () => {
    const result = await executeOcr({
      imageIndex: 0,
      image: {
        id: 'img-1',
        fileName: 'test.png',
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        translationStatus: 'pending',
        translationFailed: false,
        fontSize: 16,
        autoFontSize: false,
        fontFamily: 'fonts/STSONG.TTF',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#ffffff',
        inpaintMethod: 'solid',
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1,
        textAlign: 'start',
        hasUnsavedChanges: false
      } as any,
      bubbleCoords: [[0, 0, 10, 10]],
      textlinesPerBubble: []
    })

    expect(result.originalTexts).toEqual(['こんにちは'])
    expect(result.ocrResults).toHaveLength(1)
    expect(result.ocrResults[0]?.confidence).toBe(0.82)
    expect(parallelOcrMock).toHaveBeenCalledWith(
      expect.objectContaining({
        enable_hybrid_ocr: true,
        secondary_ocr_engine: '48px_ocr',
        hybrid_ocr_threshold: 0.2,
        use_json_format_for_ai_vision: true
      })
    )
  })

  it('imageStore should preserve ocrResults when loading legacy-compatible images', () => {
    const store = useImageStore()
    store.setImages([
      {
        id: 'img-ocr',
        fileName: 'ocr.png',
        width: 0,
        height: 0,
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        ocrResults: [
          {
            text: '保存测试',
            confidence: 0.66,
            confidenceSupported: true,
            engine: '48px_ocr',
            primaryEngine: '48px_ocr',
            fallbackUsed: false
          }
        ],
        translationStatus: 'pending',
        translationFailed: false,
        hasUnsavedChanges: false
      } as any
    ])

    expect(store.currentImage?.ocrResults?.[0]?.text).toBe('保存测试')
    expect(store.currentImage?.ocrResults?.[0]?.confidence).toBe(0.66)
  })

  it('bubbleStore sync should keep image-level OCR mirror fields aligned', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.setImages([
      {
        id: 'img-sync',
        fileName: 'sync.png',
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        translationStatus: 'pending',
        translationFailed: false,
        hasUnsavedChanges: false
      } as any
    ])
    imageStore.setCurrentImageIndex(0)

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [1, 2, 3, 4],
        rotationAngle: 15,
        originalText: '原文',
        translatedText: '译文',
        textboxText: '提示词',
        textlines: [
          {
            polygon: [[1, 2], [3, 2], [3, 4], [1, 4]],
            direction: 'h',
            confidence: 0.9
          }
        ],
        ocrResult: {
          text: '原文',
          confidence: 0.6,
          confidenceSupported: true,
          engine: '48px_ocr',
          primaryEngine: '48px_ocr',
          fallbackUsed: false
        }
      })
    ])

    expect(imageStore.currentImage?.originalTexts).toEqual(['原文'])
    expect(imageStore.currentImage?.bubbleTexts).toEqual(['译文'])
    expect(imageStore.currentImage?.textboxTexts).toEqual(['提示词'])
    expect(imageStore.currentImage?.ocrResults?.[0]?.text).toBe('原文')
    expect(imageStore.currentImage?.bubbleCoords).toEqual([[1, 2, 3, 4]])
    expect(imageStore.currentImage?.bubbleAngles).toEqual([15])
    expect(imageStore.currentImage?.textlinesPerBubble).toEqual([
      [
        {
          polygon: [[1, 2], [3, 2], [3, 4], [1, 4]],
          direction: 'h',
          confidence: 0.9
        }
      ]
    ])
  })
})
