import { computed, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useTranslateViewActions } from '@/views/useTranslateViewActions'

function createOptions() {
  return {
    imageStore: {
      hasImages: true,
      currentImageIndex: 0,
      sortImagesByFileName: vi.fn(),
      deleteCurrentImage: vi.fn(),
      clearImages: vi.fn(),
    },
    settingsStore: {
      settings: {
        textStyle: {
          autoFontSize: false,
          fontSize: 18,
        },
      },
      updateTextStyle: vi.fn(),
    },
    sessionStore: {
      saveChapterSession: vi.fn(),
    },
    translation: {
      translateCurrentImage: vi.fn(),
      translateAllImages: vi.fn(),
      executeHqTranslation: vi.fn(),
      executeProofreading: vi.fn(),
      removeTextOnly: vi.fn(),
      removeAllTexts: vi.fn(),
      translateSelectedImages: vi.fn(),
      removeTextSelection: vi.fn(),
      retryFailedImages: vi.fn(),
    },
    translateInit: {
      initializeBookChapterContext: vi.fn(),
      switchImage: vi.fn(),
      goToPrevious: vi.fn(),
      goToNext: vi.fn(),
    },
    validateBeforeTranslation: vi.fn(() => true),
    currentImage: computed(() => ({ fileName: 'page.png' })),
    hasImages: computed(() => true),
    hasFailedImages: computed(() => false),
    currentBookId: computed(() => 'book-1'),
    currentChapterId: computed(() => 'chapter-1'),
    isEditMode: ref(false),
  }
}

describe('useTranslateViewActions', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('sorts uploaded images without routine console noise', () => {
    const options = createOptions()
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})

    const actions = useTranslateViewActions(options as unknown as Parameters<typeof useTranslateViewActions>[0])
    actions.handleUploadComplete(3)

    expect(options.imageStore.sortImagesByFileName).toHaveBeenCalled()
    expect(options.translateInit.switchImage).toHaveBeenCalledWith(0)
    expect(logSpy).not.toHaveBeenCalled()
  })
})
