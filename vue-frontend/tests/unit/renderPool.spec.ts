import { describe, it, expect, vi, beforeEach } from 'vitest'

const { executeRenderMock, imageStoreMock, bubbleStoreMock, settingsStoreMock, saveTranslatedImageMock } = vi.hoisted(() => ({
  executeRenderMock: vi.fn(async () => ({
    finalImage: 'rendered-image',
    bubbleStates: [
      {
        coords: [0, 0, 10, 10],
        polygon: [],
        originalText: '原文',
        translatedText: '译文',
        textboxText: '',
        fontSize: 16,
        fontFamily: 'fonts/STSONG.TTF',
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        textColor: '#000000',
        fillColor: '#ffffff',
        rotationAngle: 0,
        position: { x: 0, y: 0 },
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        textlines: [
          {
            polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
            direction: 'h',
            confidence: 0.9
          }
        ],
        ocrResult: null
      }
    ]
  })),
  imageStoreMock: {
    updateImageByIndex: vi.fn(),
    currentImageIndex: 0
  },
  bubbleStoreMock: {
    setBubbles: vi.fn()
  },
  settingsStoreMock: {
    settings: {
      textStyle: {
        fontSize: 16,
        autoFontSize: false,
        fontFamily: 'fonts/STSONG.TTF',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#ffffff',
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        useAutoTextColor: false
      }
    }
  },
  saveTranslatedImageMock: vi.fn()
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeRender: executeRenderMock
}))

vi.mock('@/stores/imageStore', () => ({
  useImageStore: () => imageStoreMock
}))

vi.mock('@/stores/bubbleStore', () => ({
  useBubbleStore: () => bubbleStoreMock
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock
}))

vi.mock('@/composables/translation/core/saveStep', () => ({
  shouldEnableAutoSave: () => false,
  saveTranslatedImage: saveTranslatedImageMock
}))

vi.mock('@/composables/translation/parallel/useParallelTranslation', () => ({
  useParallelTranslation: () => ({
    progress: { value: {} }
  })
}))

describe('RenderPool standard mode', () => {
  beforeEach(() => {
    executeRenderMock.mockClear()
    imageStoreMock.updateImageByIndex.mockClear()
    bubbleStoreMock.setBubbles.mockClear()
    saveTranslatedImageMock.mockClear()
  })

  it('should render a standard translation task without referencing an undefined imageData variable', async () => {
    const { RenderPool } = await import('@/composables/translation/parallel/pools/RenderPool')

    const pool = new RenderPool(
      { incrementCompleted: vi.fn() } as any,
      { add: vi.fn() } as any
    )

    const task = {
      imageIndex: 0,
      imageData: {
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        userMask: null,
        bubbleStates: null,
        fontSize: 16,
        autoFontSize: false,
        fontFamily: 'fonts/STSONG.TTF',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#ffffff',
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        useAutoTextColor: false
      },
      status: 'processing',
      detectionResult: {
        bubbleCoords: [[0, 0, 10, 10]],
        bubbleAngles: [0],
        autoDirections: ['vertical'],
        textlinesPerBubble: [[{
          polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
          direction: 'h',
          confidence: 0.9
        }]]
      },
      ocrResult: {
        originalTexts: ['原文'],
        ocrResults: [{
          text: '原文',
          confidence: 0.8,
          confidenceSupported: true,
          engine: '48px_ocr',
          primaryEngine: '48px_ocr',
          fallbackUsed: false
        }]
      },
      colorResult: {
        colors: [{
          textColor: '#000000',
          bgColor: '#ffffff'
        }]
      },
      translateResult: {
        translatedTexts: ['译文'],
        textboxTexts: ['']
      },
      inpaintResult: {
        cleanImage: 'clean-image'
      }
    } as any

    await expect((pool as any).process(task)).resolves.toBe(task)
    expect(executeRenderMock).toHaveBeenCalledOnce()
  })
})
