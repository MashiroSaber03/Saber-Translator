import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { projectTaskContext } from '@/composables/translation/core/taskProjector'
import type { PipelineRuntime, TaskContext } from '@/composables/translation/core/runtime'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'
import { createDefaultSettings } from '@/stores/settings/defaults'
import type { ImageData } from '@/types/image'

describe('taskProjector', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  function createRuntime(): PipelineRuntime {
    return {
      mode: 'standard',
      settingsSnapshot: createDefaultSettings(),
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      savedTextStyles: null,
      autoSaveEnabled: true,
      isBookshelfMode: true,
      sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
      bookId: 'book-1',
      chapterId: 'chapter-1',
    }
  }

  function createSourceImage(): ImageData {
    return {
      id: 'img-1',
      fileName: 'page-1.png',
      originalDataURL: 'data:image/png;base64,orig',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: [],
      translationStatus: 'pending',
      translationFailed: false,
      hasUnsavedChanges: false,
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
      useAutoTextColor: false,
    }
  }

  function createContext(status: TaskContext['status'], persisted: boolean): TaskContext {
    return {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      sourceImage: createSourceImage(),
      status,
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
      finalImage: 'rendered',
      cleanImage: 'clean',
      bubbleStates: [],
      persisted,
    }
  }

  it('projects preview render state as processing until save completes', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

    projectTaskContext(createContext('processing', false), createRuntime(), { imageStore, bubbleStore })

    expect(imageStore.images[0]?.translationStatus).toBe('processing')
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(true)
  })

  it('projects completed save state as completed and clears unsaved flag', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

    projectTaskContext(createContext('completed', true), createRuntime(), { imageStore, bubbleStore })

    expect(imageStore.images[0]?.translationStatus).toBe('completed')
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })

  it('keeps projector fixtures typed to the current runtime contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/taskProjector.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('normalizes raw rendered payloads and preserves API image references', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,orig')

    const rawPayloadContext = createContext('completed', false)
    rawPayloadContext.finalImage = 'rendered-payload'
    projectTaskContext(rawPayloadContext, createRuntime(), { imageStore, bubbleStore })
    expect(imageStore.images[0]?.translatedDataURL).toBe('data:image/png;base64,rendered-payload')

    const apiUrlContext = createContext('completed', false)
    apiUrlContext.finalImage = '/api/images/rendered-page'
    projectTaskContext(apiUrlContext, createRuntime(), { imageStore, bubbleStore })
    expect(imageStore.images[0]?.translatedDataURL).toBe('/api/images/rendered-page')
  })

  it('shares task style projection between preview projection and persistence', () => {
    const projectorSource = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/taskProjector.ts'),
      'utf8',
    )
    const persistenceSource = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/persistenceService.ts'),
      'utf8',
    )

    expect(projectorSource).toContain('resolveTaskStyleFields')
    expect(persistenceSource).toContain('resolveTaskStyleFields')
    expect(projectorSource).not.toContain('function buildStyleProjection')
    expect(persistenceSource).not.toContain('function buildResolvedStyleFields')
  })
})
