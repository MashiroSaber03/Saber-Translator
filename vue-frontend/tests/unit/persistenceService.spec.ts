import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import type { TranslationSettings } from '@/types/settings'
import type { SavedTextStyles } from '@/composables/translation/core/types'
import type { TaskContext, PipelineRuntime } from '@/composables/translation/core/runtime'

const {
  savePageImageMock,
  savePageMetaMock,
  saveSessionMetaMock,
  apiPutMock,
  fetchMock,
} = vi.hoisted(() => ({
  savePageImageMock: vi.fn(),
  savePageMetaMock: vi.fn(),
  saveSessionMetaMock: vi.fn(),
  apiPutMock: vi.fn(),
  fetchMock: vi.fn(),
}))

vi.mock('@/api/pageStorage', () => ({
  savePageImage: savePageImageMock,
  savePageMeta: savePageMetaMock,
  saveSessionMeta: saveSessionMetaMock,
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    put: apiPutMock,
  },
}))

describe('persistenceService', () => {
  beforeEach(() => {
    savePageImageMock.mockReset()
    savePageMetaMock.mockReset()
    saveSessionMetaMock.mockReset()
    apiPutMock.mockReset()
    fetchMock.mockReset()

    savePageImageMock.mockResolvedValue({ success: true })
    savePageMetaMock.mockResolvedValue({ success: true })
    saveSessionMetaMock.mockResolvedValue({ success: true })
    apiPutMock.mockResolvedValue({ success: true })

    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('FileReader', class {
      result: string | ArrayBuffer | null = 'data:image/png;base64,from-url'
      onloadend: null | (() => void) = null
      onerror: null | (() => void) = null

      readAsDataURL() {
        this.onloadend?.()
      }
    })
  })

  function createSettings(): TranslationSettings {
    return {
      settingsSchemaVersion: 1,
      textStyle: {
        fontSize: 24,
        autoFontSize: true,
        fontFamily: 'fonts/SourceHanSans.ttf',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#ffffff',
        strokeEnabled: true,
        strokeColor: '#ffffff',
        strokeWidth: 3,
        inpaintMethod: 'litelama',
        useAutoTextColor: false,
        lineSpacing: 1,
        textAlign: 'start',
      },
      ocrEngine: 'manga_ocr',
      sourceLanguage: 'japanese',
      textDetector: 'default',
      enableAuxYoloDetection: false,
      auxYoloConfThreshold: 0.3,
      auxYoloOverlapThreshold: 0.3,
      enableSaberYoloRefine: true,
      saberYoloRefineOverlapThreshold: 0.5,
      baiduOcr: { apiKey: '', secretKey: '', version: 'accurate', sourceLanguage: 'JAP' },
      paddleOcrVl: { sourceLanguage: 'japanese' },
      aiVisionOcr: {
        provider: 'custom',
        apiKey: '',
        modelName: '',
        prompt: '',
        promptMode: 'normal',
        customBaseUrl: '',
        openaiOptions: {
          request: { forceJsonOutput: false },
          execution: { useStream: false, rpmLimit: 0, transportRetries: 0, businessRetries: 0 },
        },
        minImageSize: 28,
      },
      hybridOcr: { enabled: false, secondaryEngine: '48px_ocr', confidenceThreshold: 0.6 },
      translation: {
        provider: 'custom',
        apiKey: '',
        modelName: '',
        customBaseUrl: '',
        openaiOptions: {
          request: { forceJsonOutput: false },
          execution: { useStream: false, rpmLimit: 0, transportRetries: 0, businessRetries: 0 },
        },
        translationMode: 'batch',
        batchNormalPrompt: '',
        batchJsonPrompt: '',
        singleNormalPrompt: '',
        singleJsonPrompt: '',
      },
      targetLanguage: 'zh-CN',
      translatePrompt: '',
      useTextboxPrompt: false,
      textboxPrompt: '',
      hqTranslation: {
        provider: 'custom',
        apiKey: '',
        modelName: '',
        customBaseUrl: '',
        openaiOptions: {
          request: { forceJsonOutput: false },
          execution: { useStream: false, rpmLimit: 0, transportRetries: 0, businessRetries: 0 },
        },
        batchSize: 3,
        prompt: '',
      },
      pluginAgent: {
        provider: 'custom',
        apiKey: '',
        modelName: '',
        customBaseUrl: '',
        openaiOptions: {
          request: { forceJsonOutput: false },
          execution: { useStream: false, rpmLimit: 0, transportRetries: 0, businessRetries: 0 },
        },
      },
      proofreading: { enabled: false, rounds: [], maxRetries: 0 },
      boxExpand: { ratio: 0, top: 0, bottom: 0, left: 0, right: 0 },
      preciseMask: { dilateSize: 0, boxExpandRatio: 0 },
      pdfProcessingMethod: 'backend',
      showDetectionDebug: false,
      parallel: { enabled: true, deepLearningLockSize: 1 },
      autoSaveInBookshelfMode: true,
      removeTextWithOcr: false,
      enableVerboseLogs: false,
      lamaDisableResize: false,
    }
  }

  function createSavedTextStyles(): SavedTextStyles {
    return {
      fontFamily: 'fonts/SourceHanSans.ttf',
      fontSize: 24,
      autoFontSize: true,
      textDirection: 'vertical',
      autoTextDirection: true,
      layoutDirection: 'auto',
      fillColor: '#ffffff',
      textColor: '#000000',
      rotationAngle: 0,
      strokeEnabled: true,
      strokeColor: '#ffffff',
      strokeWidth: 3,
      useAutoTextColor: false,
      inpaintMethod: 'litelama',
      lineSpacing: 1,
      textAlign: 'start',
    }
  }

  function createRuntime(overrides: Partial<PipelineRuntime> = {}): PipelineRuntime {
    return {
      mode: 'standard',
      settingsSnapshot: createSettings(),
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      savedTextStyles: createSavedTextStyles(),
      autoSaveEnabled: true,
      isBookshelfMode: true,
      sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
      bookId: 'book-1',
      chapterId: 'chapter-1',
      ...overrides,
    }
  }

  function createContext(overrides: Partial<TaskContext> = {}): TaskContext {
    return {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      status: 'processing',
      sourceImage: {
        id: 'img-1',
        fileName: 'page-1.png',
        width: 100,
        height: 100,
        originalDataURL: 'data:image/png;base64,original-image',
        translatedDataURL: 'data:image/png;base64,stale-store-image',
        cleanImageData: 'stale-clean',
        bubbleStates: null,
        translationStatus: 'completed',
        translationFailed: false,
        hasUnsavedChanges: true,
      } as any,
      bubbleCoords: [],
      bubbleAngles: [],
      bubblePolygons: [],
      autoDirections: [],
      textMask: 'text-mask',
      textlinesPerBubble: [],
      originalTexts: [],
      ocrResults: [],
      colors: [],
      translatedTexts: [],
      textboxTexts: [],
      warnings: [],
      cleanImage: 'latest-clean-image',
      finalImage: 'latest-rendered-image',
      bubbleStates: [],
      persisted: false,
      ...overrides,
    }
  }

  it('persists the latest task context result instead of stale store image fields', async () => {
    const { persistPage } = await import('@/composables/translation/core/persistenceService')

    const context = createContext({
      sourceImage: {
        id: 'img-1',
        fileName: 'page-1.png',
        originalDataURL: 'data:image/png;base64,original-image',
        translatedDataURL: 'data:image/png;base64,stale-store-image',
        cleanImageData: 'stale-clean',
        bubbleStates: [{ translatedText: '旧译文' }],
        translationStatus: 'completed',
        translationFailed: false,
        hasUnsavedChanges: true,
      } as any,
      bubbleStates: [{ translatedText: '新译文' }] as any,
      translatedTexts: ['新译文'],
      finalImage: 'latest-rendered-image',
      cleanImage: 'latest-clean-image',
    })

    const runtime = createRuntime()
    const result = await persistPage(context, runtime)

    expect(result.persisted).toBe(true)
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'translated', 'latest-rendered-image')
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'clean', 'latest-clean-image')
    expect(savePageMetaMock).toHaveBeenCalledWith(
      runtime.sessionPath,
      0,
      expect.objectContaining({
        fileName: 'page-1.png',
        bubbleStates: [{ translatedText: '新译文' }],
        hasUnsavedChanges: false,
      }),
    )
  })

  it('persists original images for manual/session initialization flows and updates session meta', async () => {
    const { persistAllPages, persistSessionMeta } = await import('@/composables/translation/core/persistenceService')

    const runtime = createRuntime()
    const contexts = [
      createContext(),
      createContext({
        id: 'task-2',
        imageIndex: 1,
        sourceImage: {
          id: 'img-2',
          fileName: 'page-2.png',
          originalDataURL: 'data:image/png;base64,original-image-2',
          translatedDataURL: null,
          cleanImageData: null,
          bubbleStates: null,
          translationStatus: 'pending',
          translationFailed: false,
          hasUnsavedChanges: false,
        } as any,
        finalImage: undefined,
        cleanImage: undefined,
      }),
    ]

    const progress = vi.fn()
    await persistAllPages(contexts, runtime, { includeOriginal: true, onProgress: progress, currentImageIndex: 1 })
    await persistSessionMeta(runtime, { totalPages: 2, currentImageIndex: 1 })

    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'original', 'original-image')
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 1, 'original', 'original-image-2')
    expect(saveSessionMetaMock).toHaveBeenCalledWith(runtime.sessionPath, expect.objectContaining({
      total_pages: 2,
      currentImageIndex: 1,
    }))
    expect(apiPutMock).toHaveBeenCalledWith('/api/bookshelf/books/book-1/chapters/chapter-1/image-count', { count: 2 })
    expect(progress).toHaveBeenNthCalledWith(1, 1, 2)
    expect(progress).toHaveBeenNthCalledWith(2, 2, 2)
  })

  it('converts stored /api/ image urls before saving manual chapter snapshots', async () => {
    const { persistPage } = await import('@/composables/translation/core/persistenceService')

    fetchMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['binary'], { type: 'image/png' }),
    })

    const runtime = createRuntime()
    const context = createContext({
      sourceImage: {
        id: 'img-1',
        fileName: 'page-1.png',
        originalDataURL: '/api/sessions/page/bookshelf/book-1/chapters/chapter-1/session/0/original',
        translatedDataURL: '/api/sessions/page/bookshelf/book-1/chapters/chapter-1/session/0/translated',
        cleanImageData: '/api/sessions/page/bookshelf/book-1/chapters/chapter-1/session/0/clean',
        bubbleStates: [],
        translationStatus: 'completed',
        translationFailed: false,
        hasUnsavedChanges: false,
      } as any,
      finalImage: undefined,
      cleanImage: undefined,
    })

    await persistPage(context, runtime, { includeOriginal: true, includeDerivedImagesFromSource: true })

    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'original', 'from-url')
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'translated', 'from-url')
    expect(savePageImageMock).toHaveBeenCalledWith(runtime.sessionPath, 0, 'clean', 'from-url')
  })

  it('hydrates task context from cloned image state so task mutations do not leak into the store image', async () => {
    const { hydrateTaskContextFromImage } = await import('@/composables/translation/core/runtime')

    const sourceImage = {
      id: 'img-1',
      fileName: 'page-1.png',
      originalDataURL: 'data:image/png;base64,original-image',
      translatedDataURL: 'data:image/png;base64,translated-image',
      cleanImageData: 'clean-image',
      bubbleStates: [{ translatedText: '原始值', originalText: '原文', textboxText: '', coords: [0, 0, 10, 10] }],
      translationWarnings: [{ imageIndex: 0, bubbleIndex: 0, source: 'a', expectedTarget: 'b', actualTranslation: 'c' }],
    } as any

    const runtime = createRuntime()
    const context = hydrateTaskContextFromImage(0, sourceImage, 'standard', runtime)
    ;(context.bubbleStates as any[])[0].translatedText = '任务内修改'

    expect(sourceImage.bubbleStates[0].translatedText).toBe('原始值')
    expect((context.sourceImage.bubbleStates as any[])[0].translatedText).toBe('任务内修改')
  })
})
