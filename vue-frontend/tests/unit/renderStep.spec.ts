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
      currentMode: 'standard',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
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
      currentMode: 'standard',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
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

  it('preserves existing per-bubble style fields instead of overwriting them with global saved styles', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered-preserve',
      bubble_states: []
    })

    const { executeRender } = await import('@/composables/translation/core/steps/render')
    const result = await executeRender({
      imageIndex: 0,
      cleanImage: 'clean-image',
      bubbleCoords: [[0, 0, 20, 20]],
      bubbleAngles: [15],
      autoDirections: ['horizontal'],
      textlinesPerBubble: [],
      existingBubbleStates: [{
        coords: [0, 0, 20, 20],
        polygon: [],
        originalText: '原文',
        translatedText: '旧译文',
        textboxText: '',
        fontSize: 22,
        fontFamily: 'fonts/CUSTOM.TTF',
        textDirection: 'horizontal',
        autoTextDirection: 'horizontal',
        textColor: '#123456',
        fillColor: '#abcdef',
        rotationAngle: 15,
        position: { x: 1, y: 2 },
        strokeEnabled: true,
        strokeColor: '#654321',
        strokeWidth: 3,
        lineSpacing: 1.4,
        textAlign: 'center',
        inpaintMethod: 'solid',
        useAutoTextColor: false,
        autoFgColor: null,
        autoBgColor: null,
        colorConfidence: 0.6,
        textlines: [],
        ocrResult: null
      }],
      originalTexts: ['原文'],
      ocrResults: [],
      translatedTexts: ['新译文'],
      textboxTexts: [''],
      colors: [{ textColor: '#111111', bgColor: '#ffffff' }],
      savedTextStyles: {
        fontFamily: 'fonts/GLOBAL.TTF',
        fontSize: 99,
        autoFontSize: true,
        textDirection: 'vertical',
        autoTextDirection: false,
        layoutDirection: 'vertical',
        fillColor: '#ffffff',
        textColor: '#000000',
        rotationAngle: 0,
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        useAutoTextColor: false,
        inpaintMethod: 'litelama',
        lineSpacing: 1,
        textAlign: 'start'
      },
      currentMode: 'standard',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    })

    expect(result.finalImage).toBe('rendered-preserve')
    expect(result.bubbleStates[0]?.fontSize).toBe(22)
    expect(result.bubbleStates[0]?.fontFamily).toBe('fonts/CUSTOM.TTF')
    expect(result.bubbleStates[0]?.textDirection).toBe('horizontal')
    expect(result.bubbleStates[0]?.textColor).toBe('#123456')
    expect(result.bubbleStates[0]?.strokeEnabled).toBe(true)
    expect(result.bubbleStates[0]?.textAlign).toBe('center')
  })

  it('keeps auto font size disabled during ordinary preserve renders even when page settings still remember auto mode', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered-manual-font',
      bubble_states: []
    })

    const { executeRender } = await import('@/composables/translation/core/steps/render')
    await executeRender({
      imageIndex: 0,
      cleanImage: 'clean-image',
      bubbleCoords: [[0, 0, 20, 20]],
      bubbleAngles: [0],
      autoDirections: ['vertical'],
      textlinesPerBubble: [[]],
      existingBubbleStates: [{
        coords: [0, 0, 20, 20],
        polygon: [],
        originalText: '原文',
        translatedText: '译文',
        textboxText: '',
        fontSize: 26,
        fontFamily: 'fonts/CUSTOM.TTF',
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        textColor: '#123456',
        fillColor: '#abcdef',
        rotationAngle: 0,
        position: { x: 0, y: 0 },
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1.1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
        textlines: [],
        ocrResult: null
      }],
      originalTexts: ['原文'],
      ocrResults: [],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      colors: [{ textColor: '#010203', bgColor: '#0a0b0c' }],
      savedTextStyles: {
        fontFamily: 'fonts/GLOBAL.TTF',
        fontSize: 99,
        autoFontSize: true,
        textDirection: 'vertical',
        autoTextDirection: false,
        layoutDirection: 'vertical',
        fillColor: '#ffffff',
        textColor: '#000000',
        rotationAngle: 0,
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        useAutoTextColor: true,
        inpaintMethod: 'solid',
        lineSpacing: 1.1,
        textAlign: 'start'
      },
      currentMode: 'standard',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    } as any)

    expect(parallelRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      autoFontSize: false,
      bubble_states: [
        expect.objectContaining({
          fontSize: 26,
          textColor: '#123456',
          fillColor: '#abcdef',
        })
      ],
    }))
  })

  it('re-applies auto font size and auto colors only when explicitly requested', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered-auto-init',
      bubble_states: [
        {
          coords: [0, 0, 20, 20],
          polygon: [],
          originalText: '原文',
          translatedText: '译文',
          textboxText: '',
          fontSize: 34,
          fontFamily: 'fonts/CUSTOM.TTF',
          textDirection: 'vertical',
          autoTextDirection: 'vertical',
          textColor: '#010203',
          fillColor: '#0a0b0c',
          rotationAngle: 0,
          position: { x: 0, y: 0 },
          strokeEnabled: false,
          strokeColor: '#000000',
          strokeWidth: 1,
          lineSpacing: 1.1,
          textAlign: 'start',
          inpaintMethod: 'solid',
          autoFgColor: [1, 2, 3],
          autoBgColor: [10, 11, 12],
          textlines: [],
          ocrResult: null
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
      textlinesPerBubble: [[]],
      existingBubbleStates: [{
        coords: [0, 0, 20, 20],
        polygon: [],
        originalText: '原文',
        translatedText: '译文',
        textboxText: '',
        fontSize: 26,
        fontFamily: 'fonts/CUSTOM.TTF',
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        textColor: '#123456',
        fillColor: '#abcdef',
        rotationAngle: 0,
        position: { x: 0, y: 0 },
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1.1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
        textlines: [],
        ocrResult: null
      }],
      originalTexts: ['原文'],
      ocrResults: [],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      colors: [{ textColor: '#010203', bgColor: '#0a0b0c' }],
      savedTextStyles: {
        fontFamily: 'fonts/GLOBAL.TTF',
        fontSize: 99,
        autoFontSize: true,
        textDirection: 'vertical',
        autoTextDirection: false,
        layoutDirection: 'vertical',
        fillColor: '#ffffff',
        textColor: '#000000',
        rotationAngle: 0,
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        useAutoTextColor: true,
        inpaintMethod: 'solid',
        lineSpacing: 1.1,
        textAlign: 'start'
      },
      currentMode: 'standard',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'initialize_auto',
        color: 'initialize_auto',
      },
    } as any)

    expect(parallelRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      autoFontSize: true,
      bubble_states: [
        expect.objectContaining({
          textColor: '#010203',
          fillColor: '#0a0b0c',
        })
      ],
    }))
    expect(result.bubbleStates[0]?.fontSize).toBe(34)
    expect(result.bubbleStates[0]?.textColor).toBe('#010203')
    expect(result.bubbleStates[0]?.fillColor).toBe('#0a0b0c')
  })

  it('preserves existing auto color backups when render runs without fresh color extraction data', async () => {
    parallelRenderMock.mockResolvedValue({
      success: true,
      final_image: 'rendered-proofread',
      bubble_states: []
    })

    const { executeRender } = await import('@/composables/translation/core/steps/render')
    const result = await executeRender({
      imageIndex: 0,
      cleanImage: 'clean-image',
      bubbleCoords: [[0, 0, 20, 20]],
      bubbleAngles: [0],
      autoDirections: ['vertical'],
      textlinesPerBubble: [[]],
      existingBubbleStates: [{
        coords: [0, 0, 20, 20],
        polygon: [],
        originalText: '原文',
        translatedText: '译文',
        textboxText: '',
        fontSize: 26,
        fontFamily: 'fonts/CUSTOM.TTF',
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        textColor: '#123456',
        fillColor: '#abcdef',
        rotationAngle: 0,
        position: { x: 0, y: 0 },
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        lineSpacing: 1.1,
        textAlign: 'start',
        inpaintMethod: 'solid',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
        textlines: [],
        ocrResult: null
      }],
      originalTexts: ['原文'],
      ocrResults: [],
      translatedTexts: ['译文'],
      textboxTexts: [''],
      colors: [],
      savedTextStyles: {
        fontFamily: 'fonts/GLOBAL.TTF',
        fontSize: 99,
        autoFontSize: false,
        textDirection: 'vertical',
        autoTextDirection: false,
        layoutDirection: 'vertical',
        fillColor: '#ffffff',
        textColor: '#000000',
        rotationAngle: 0,
        strokeEnabled: false,
        strokeColor: '#000000',
        strokeWidth: 1,
        useAutoTextColor: false,
        inpaintMethod: 'solid',
        lineSpacing: 1.1,
        textAlign: 'start'
      },
      currentMode: 'proofread',
      settingsSnapshot: settingsStoreMock.settings as any,
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    } as any)

    expect(result.bubbleStates[0]?.autoFgColor).toEqual([1, 2, 3])
    expect(result.bubbleStates[0]?.autoBgColor).toEqual([10, 11, 12])
  })
})
