import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createBubbleState } from '@/utils/bubbleFactory'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import type { PipelineRuntime, TaskContext } from '@/composables/translation/core/runtime'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { ImageData } from '@/types/image'
import type { OcrResult } from '@/types/ocr'

const { executeAutoGlossaryMock, executeOcrMock } = vi.hoisted(() => ({
  executeAutoGlossaryMock: vi.fn(),
  executeOcrMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: vi.fn(),
  executeOcr: executeOcrMock,
  executeColor: vi.fn(),
  executeAutoGlossary: executeAutoGlossaryMock,
  executeTranslate: vi.fn(),
  executeAiTranslate: vi.fn(),
  executeInpaint: vi.fn(),
  executeRender: vi.fn(),
}))

function createTestImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    id: 'image-1',
    fileName: 'page.png',
    originalDataURL: 'data:image/png;base64,original',
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    fontSize: 18,
    autoFontSize: false,
    fontFamily: 'fonts/STSONG.TTF',
    layoutDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#ffffff',
    inpaintMethod: 'solid',
    strokeEnabled: false,
    strokeColor: '#000000',
    strokeWidth: 1,
    lineSpacing: 1,
    textAlign: 'start',
    useAutoTextColor: false,
    hasUnsavedChanges: false,
    ...overrides,
  }
}

function createTestRuntime(overrides: Partial<PipelineRuntime> = {}): PipelineRuntime {
  return {
    mode: 'standard',
    settingsSnapshot: createDefaultSettings(),
    bookTranslationConstraints: createEmptyBookTranslationConstraints(),
    savedTextStyles: null,
    autoSaveEnabled: false,
    isBookshelfMode: false,
    sessionPath: null,
    bookId: null,
    chapterId: null,
    ...overrides,
  }
}

function createTestContext(overrides: Partial<TaskContext> = {}): TaskContext {
  return {
    id: 'task-1',
    imageIndex: 0,
    translationMode: 'standard',
    sourceImage: createTestImage(),
    status: 'processing',
    bubbleCoords: [],
    bubbleAngles: [],
    bubblePolygons: [],
    autoDirections: [],
    textlinesPerBubble: [],
    originalTexts: [],
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
    bubbleStates: [],
    persisted: false,
    ...overrides,
  }
}

describe('executeAtomicStep', () => {
  beforeEach(() => {
    executeAutoGlossaryMock.mockReset()
    executeOcrMock.mockReset()
  })

  it('keeps atomic-step runtime cloning on the shared helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/translation/core/atomicSteps.ts'), 'utf8')

    expect(source).toContain("from '@/utils/deepClone'")
    expect(source).not.toContain('JSON.parse(JSON.stringify(result.bookTranslationConstraints))')
  })

  it('keeps atomic-step fixtures typed to the current runtime contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/atomicSteps.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('merges OCR output into bubble states so remove-text mode keeps original text metadata', async () => {
    const ocrResult: OcrResult = {
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
    const result = await executeAtomicStep('ocr', createTestContext({
      translationMode: 'removeText',
      bubbleCoords: [[0, 0, 100, 80]],
      autoDirections: ['vertical'],
      textlinesPerBubble: [[]],
      originalTexts: [''],
      bubbleStates: [createBubbleState({
        originalText: '',
        coords: [0, 0, 100, 80],
        textDirection: 'vertical',
        autoTextDirection: 'vertical',
        ocrResult: null,
      })],
    }), createTestRuntime({
      mode: 'removeText',
    }))

    expect(result.originalTexts).toEqual(['縦書き原文'])
    expect(result.ocrResults).toEqual([ocrResult])
    expect(result.bubbleStates?.[0]?.originalText).toBe('縦書き原文')
    expect(result.bubbleStates?.[0]?.ocrResult).toEqual(ocrResult)
  })

  it('deep-clones auto glossary constraints before storing them on runtime', async () => {
    const nextConstraints: BookTranslationConstraints = {
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        autoExtractPrompt: 'prompt',
        entries: [{ source: '魔法', target: 'magic', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    }
    executeAutoGlossaryMock.mockResolvedValue({
      bookTranslationConstraints: nextConstraints,
      autoGlossaryStats: {
        added: 1,
        duplicates: 2,
        failedPages: 3,
      },
    })

    const runtime = createTestRuntime({
      isBookshelfMode: true,
      bookId: 'book-1',
      chapterId: 'chapter-1',
    })

    const context = createTestContext({
      originalTexts: ['魔法'],
      autoGlossaryStats: {
        added: 4,
        duplicates: 5,
        failedPages: 6,
      },
    })

    const { executeAtomicStep } = await import('@/composables/translation/core/atomicSteps')
    const result = await executeAtomicStep('autoGlossary', context, runtime)

    expect(runtime.bookTranslationConstraints).toEqual(nextConstraints)
    expect(runtime.bookTranslationConstraints).not.toBe(nextConstraints)
    expect(runtime.bookTranslationConstraints.glossary.entries).not.toBe(nextConstraints.glossary.entries)
    expect(result.autoGlossaryStats).toEqual({
      added: 5,
      duplicates: 7,
      failedPages: 9,
    })
  })
})
