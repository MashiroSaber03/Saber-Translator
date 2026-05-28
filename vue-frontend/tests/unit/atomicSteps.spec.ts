import { beforeEach, describe, expect, it, vi } from 'vitest'

const { executeOcrMock } = vi.hoisted(() => ({
  executeOcrMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: vi.fn(),
  executeOcr: executeOcrMock,
  executeColor: vi.fn(),
  executeAutoGlossary: vi.fn(),
  executeTranslate: vi.fn(),
  executeAiTranslate: vi.fn(),
  executeInpaint: vi.fn(),
  executeRender: vi.fn(),
}))

describe('executeAtomicStep', () => {
  beforeEach(() => {
    executeOcrMock.mockReset()
  })

  it('merges OCR output into bubble states so remove-text mode keeps original text metadata', async () => {
    const ocrResult = {
      text: '縦書き原文',
      confidence: 0.91,
      confidenceSupported: true,
      engine: '48px_ocr',
      primaryEngine: '48px_ocr',
      fallbackUsed: false,
    }
    executeOcrMock.mockResolvedValue({
      originalTexts: ['縦書き原文'],
      ocrResults: [ocrResult],
    })

    const { executeAtomicStep } = await import('@/composables/translation/core/atomicSteps')
    const result = await executeAtomicStep('ocr', {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'removeText',
      sourceImage: {
        originalDataURL: 'data:image/png;base64,original',
        userMask: null,
      },
      status: 'processing',
      bubbleCoords: [[0, 0, 100, 80]],
      bubbleAngles: [0],
      bubblePolygons: [[]],
      autoDirections: ['vertical'],
      textlinesPerBubble: [[]],
      originalTexts: [''],
      ocrResults: [],
      colors: [],
      translatedTexts: [],
      textboxTexts: [],
      warnings: [],
      autoGlossaryStats: {
        added: 0,
        duplicates: 0,
        failedPages: 0,
      },
      bubbleStates: [{
        originalText: '',
        translatedText: '',
        textboxText: '',
        coords: [0, 0, 100, 80],
        polygon: [],
        fontSize: 18,
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
        textlines: [],
        ocrResult: null,
      }],
      persisted: false,
    } as any, {
      mode: 'removeText',
      settingsSnapshot: {} as any,
      bookTranslationConstraints: {} as any,
      savedTextStyles: null,
      autoSaveEnabled: false,
      isBookshelfMode: false,
      sessionPath: null,
      bookId: null,
      chapterId: null,
    })

    expect(result.originalTexts).toEqual(['縦書き原文'])
    expect(result.ocrResults).toEqual([ocrResult])
    expect(result.bubbleStates?.[0]?.originalText).toBe('縦書き原文')
    expect(result.bubbleStates?.[0]?.ocrResult).toEqual(ocrResult)
  })
})
