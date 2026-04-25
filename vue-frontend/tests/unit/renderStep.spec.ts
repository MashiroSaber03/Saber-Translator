import { beforeEach, describe, expect, it, vi } from 'vitest'

const { parallelRenderMock, settingsStoreMock } = vi.hoisted(() => ({
  parallelRenderMock: vi.fn(),
  settingsStoreMock: {
    settings: {
      textStyle: {
        fontSize: 18,
        autoFontSize: false,
        fontFamily: 'fonts/STSONG.TTF',
        layoutDirection: 'auto',
        textColor: '#111111',
        fillColor: '#ffffff',
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1.1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        useAutoTextColor: false
      }
    }
  }
}))

vi.mock('@/api/parallelTranslate', () => ({
  parallelRender: parallelRenderMock
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => settingsStoreMock
}))

describe('executeRender', () => {
  beforeEach(() => {
    parallelRenderMock.mockReset()
  })

  it('preserves local textlines and ocrResult when backend returns legacy bubble states', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered',
      bubble_states: [
        {
          coords: [0, 0, 20, 20],
          originalText: '原文',
          translatedText: '译文',
          textboxText: '',
          fontSize: 18,
          fontFamily: 'fonts/STSONG.TTF',
          textDirection: 'vertical',
          autoTextDirection: 'vertical',
          textColor: '#111111',
          fillColor: '#ffffff',
          rotationAngle: 0,
          position: { x: 0, y: 0 },
          strokeEnabled: false,
          strokeColor: '#000000',
          strokeWidth: 1,
          lineSpacing: 1.1,
          textAlign: 'start',
          inpaintMethod: 'solid'
        }
      ]
    })

    const { executeRender } = await import('@/composables/translation/core/steps/render')
    const result = await executeRender({
      imageIndex: 0,
      cleanImage: 'clean-image',
      bubbleCoords: [[0, 0, 20, 20]],
      bubbleAngles: [0],
      autoDirections: ['vertical'],
      textlinesPerBubble: [[{
        polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
        direction: 'h',
        confidence: 0.9
      }]],
      existingBubbleStates: [{
        coords: [0, 0, 20, 20],
        polygon: [],
        originalText: '原文',
        translatedText: '旧译文',
        textboxText: '',
        fontSize: 18,
        fontFamily: 'fonts/STSONG.TTF',
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        textColor: '#111111',
        fillColor: '#ffffff',
        rotationAngle: 0,
        position: { x: 0, y: 0 },
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1.1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        autoFgColor: null,
        autoBgColor: null,
        colorConfidence: 0,
        textlines: [{
          polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
          direction: 'h',
          confidence: 0.9
        }],
        ocrResult: {
          text: '原文',
          confidence: 0.88,
          confidenceSupported: true,
          engine: '48px_ocr',
          primaryEngine: '48px_ocr',
          fallbackUsed: false
        }
      }],
      originalTexts: ['原文'],
      ocrResults: [{
        text: '原文',
        confidence: 0.88,
        confidenceSupported: true,
        engine: '48px_ocr',
        primaryEngine: '48px_ocr',
        fallbackUsed: false
      }],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      colors: [{
        textColor: '#111111',
        bgColor: '#ffffff'
      }],
      savedTextStyles: null,
      currentMode: 'standard'
    })

    expect(result.finalImage).toBe('rendered')
    expect(result.bubbleStates[0]?.textlines).toEqual([
      {
        polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
        direction: 'h',
        confidence: 0.9
      }
    ])
    expect(result.bubbleStates[0]?.ocrResult?.engine).toBe('48px_ocr')
    expect(result.bubbleStates[0]?.ocrResult?.confidence).toBe(0.88)
  })

  it('uses backend bubble state fields when backend returns complete structured state', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered-complete',
      bubble_states: [
        {
          coords: [0, 0, 20, 20],
          polygon: [],
          originalText: '后端原文',
          translatedText: '后端译文',
          textboxText: '',
          fontSize: 18,
          fontFamily: 'fonts/STSONG.TTF',
          textDirection: 'vertical',
          autoTextDirection: 'vertical',
          textColor: '#222222',
          fillColor: '#eeeeee',
          rotationAngle: 0,
          position: { x: 1, y: 2 },
          strokeEnabled: false,
          strokeColor: '#000000',
          strokeWidth: 1,
          lineSpacing: 1.1,
          textAlign: 'center',
          inpaintMethod: 'solid',
          autoFgColor: null,
          autoBgColor: null,
          colorConfidence: 0.75,
          textlines: [{
            polygon: [[1, 1], [9, 1], [9, 9], [1, 9]],
            direction: 'v',
            confidence: 0.5
          }],
          ocrResult: {
            text: '后端原文',
            confidence: 0.66,
            confidenceSupported: true,
            engine: 'manga_ocr',
            primaryEngine: '48px_ocr',
            fallbackUsed: true
          }
        }
      ]
    })

    const { executeRender } = await import('@/composables/translation/core/steps/render')
    const result = await executeRender({
      imageIndex: 0,
      cleanImage: 'clean-image',
      bubbleCoords: [[0, 0, 20, 20]],
      bubbleAngles: [0],
      autoDirections: ['vertical'],
      textlinesPerBubble: [],
      existingBubbleStates: null,
      originalTexts: ['原文'],
      ocrResults: [],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      colors: [{
        textColor: '#111111',
        bgColor: '#ffffff'
      }],
      savedTextStyles: null,
      currentMode: 'standard'
    })

    expect(result.finalImage).toBe('rendered-complete')
    expect(result.bubbleStates[0]?.originalText).toBe('后端原文')
    expect(result.bubbleStates[0]?.textlines).toEqual([
      {
        polygon: [[1, 1], [9, 1], [9, 9], [1, 9]],
        direction: 'v',
        confidence: 0.5
      }
    ])
    expect(result.bubbleStates[0]?.ocrResult?.engine).toBe('manga_ocr')
    expect(result.bubbleStates[0]?.colorConfidence).toBe(0.75)
  })
})
