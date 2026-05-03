import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useSettingsStore } from '@/stores/settingsStore'
import {
  forceInitializeBookshelfSession,
  isBookshelfSessionInitialized,
  resetSaveState,
  saveBookshelfPageProgress,
} from '@/composables/translation/core/saveStep'

const {
  loadSessionMetaMock,
  saveSessionMetaMock,
  saveTranslatedPageMock,
  saveAllPagesSequentiallyMock,
  apiPutMock,
} = vi.hoisted(() => ({
  loadSessionMetaMock: vi.fn(),
  saveSessionMetaMock: vi.fn(),
  saveTranslatedPageMock: vi.fn(),
  saveAllPagesSequentiallyMock: vi.fn(),
  apiPutMock: vi.fn(),
}))

vi.mock('@/api/pageStorage', () => ({
  loadSessionMeta: loadSessionMetaMock,
  saveSessionMeta: saveSessionMetaMock,
  saveTranslatedPage: saveTranslatedPageMock,
  saveAllPagesSequentially: saveAllPagesSequentiallyMock,
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    put: apiPutMock,
  },
}))

describe('saveStep bookshelf persistence helpers', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    resetSaveState()

    loadSessionMetaMock.mockReset()
    saveSessionMetaMock.mockReset()
    saveTranslatedPageMock.mockReset()
    saveAllPagesSequentiallyMock.mockReset()
    apiPutMock.mockReset()

    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')

    const settingsStore = useSettingsStore()
    settingsStore.setAutoSaveInBookshelfMode(false)

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,original-1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,original-2')
  })

  it('treats missing session meta as an uninitialized bookshelf session', async () => {
    loadSessionMetaMock.mockRejectedValue({ status: 404, message: 'not found' })

    await expect(isBookshelfSessionInitialized()).resolves.toBe(false)
  })

  it('forces chapter initialization even when bookshelf auto save is disabled', async () => {
    saveAllPagesSequentiallyMock.mockResolvedValue(2)
    saveSessionMetaMock.mockResolvedValue({ success: true })
    apiPutMock.mockResolvedValue({ success: true })

    const imageStore = useImageStore()
    imageStore.setCurrentImageIndex(1)

    await expect(forceInitializeBookshelfSession()).resolves.toBe(true)

    expect(saveAllPagesSequentiallyMock).toHaveBeenCalledTimes(1)
    expect(saveAllPagesSequentiallyMock.mock.calls[0]?.[0]).toBe('bookshelf/book-1/chapters/chapter-1/session')
    expect(saveSessionMetaMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', expect.objectContaining({
      total_pages: 2,
      currentImageIndex: 1,
    }))
    expect(apiPutMock).toHaveBeenCalledTimes(1)
  })

  it('persists the current page payload and updates session progress metadata', async () => {
    saveTranslatedPageMock.mockResolvedValue({ success: true })
    saveSessionMetaMock.mockResolvedValue({ success: true })

    const imageStore = useImageStore()
    imageStore.updateImageByIndex(0, {
      translatedDataURL: 'data:image/png;base64,translated-1',
      cleanImageData: 'clean-1',
      hasUnsavedChanges: true,
      bubbleStates: [],
    })

    await saveBookshelfPageProgress(0, 1)

    expect(saveTranslatedPageMock).toHaveBeenCalledWith(
      'bookshelf/book-1/chapters/chapter-1/session',
      0,
      expect.objectContaining({
        translated: 'data:image/png;base64,translated-1',
        clean: 'data:image/png;base64,clean-1',
        meta: expect.objectContaining({
          fileName: 'page-1.png',
          hasUnsavedChanges: true,
        }),
      })
    )
    expect(saveSessionMetaMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', expect.objectContaining({
      total_pages: 2,
      currentImageIndex: 1,
    }))
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })

  it('prefers the current chapter context over a stale cached session path', async () => {
    saveAllPagesSequentiallyMock.mockResolvedValue(2)
    saveSessionMetaMock.mockResolvedValue({ success: true })
    saveTranslatedPageMock.mockResolvedValue({ success: true })
    apiPutMock.mockResolvedValue({ success: true })

    await forceInitializeBookshelfSession()

    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-2', 'chapter-2', 'Book 2', 'Chapter 2')

    saveTranslatedPageMock.mockClear()
    saveSessionMetaMock.mockClear()

    await saveBookshelfPageProgress(0, 1)

    expect(saveTranslatedPageMock).toHaveBeenCalledWith(
      'bookshelf/book-2/chapters/chapter-2/session',
      0,
      expect.any(Object)
    )
    expect(saveSessionMetaMock).toHaveBeenCalledWith(
      'bookshelf/book-2/chapters/chapter-2/session',
      expect.objectContaining({ currentImageIndex: 1 })
    )
  })
})
