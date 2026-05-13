import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { projectTaskContext } from '@/composables/translation/core/taskProjector'
import type { PipelineRuntime, TaskContext } from '@/composables/translation/core/runtime'
import { createEmptyBookTranslationConstraints } from '@/utils/bookTranslationConstraints'

describe('taskProjector', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  function createRuntime(): PipelineRuntime {
    return {
      mode: 'standard',
      settingsSnapshot: {
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
          inpaintMethod: 'solid',
          useAutoTextColor: false,
          lineSpacing: 1,
          textAlign: 'start',
        },
      } as any,
      bookTranslationConstraints: createEmptyBookTranslationConstraints(),
      savedTextStyles: null,
      autoSaveEnabled: true,
      isBookshelfMode: true,
      sessionPath: 'bookshelf/book-1/chapters/chapter-1/session',
      bookId: 'book-1',
      chapterId: 'chapter-1',
    }
  }

  function createContext(status: TaskContext['status'], persisted: boolean): TaskContext {
    return {
      id: 'task-1',
      imageIndex: 0,
      translationMode: 'standard',
      sourceImage: {
        id: 'img-1',
        fileName: 'page-1.png',
        originalDataURL: 'data:image/png;base64,orig',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: [],
      } as any,
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
})
