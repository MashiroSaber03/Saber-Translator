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
  savePageImageMock,
  savePageMetaMock,
  apiPutMock,
} = vi.hoisted(() => ({
  loadSessionMetaMock: vi.fn(),
  saveSessionMetaMock: vi.fn(),
  savePageImageMock: vi.fn(),
  savePageMetaMock: vi.fn(),
  apiPutMock: vi.fn(),
}))

vi.mock('@/api/pageStorage', () => ({
  loadSessionMeta: loadSessionMetaMock,
  saveSessionMeta: saveSessionMetaMock,
  savePageImage: savePageImageMock,
  savePageMeta: savePageMetaMock,
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
    savePageImageMock.mockReset()
    savePageMetaMock.mockReset()
    apiPutMock.mockReset()

    savePageImageMock.mockResolvedValue({ success: true })
    savePageMetaMock.mockResolvedValue({ success: true })
    saveSessionMetaMock.mockResolvedValue({ success: true })
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
    const imageStore = useImageStore()
    imageStore.setCurrentImageIndex(1)

    await expect(forceInitializeBookshelfSession()).resolves.toBe(true)

    expect(savePageImageMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', 0, 'original', 'original-1')
    expect(savePageImageMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', 1, 'original', 'original-2')
    expect(saveSessionMetaMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', expect.objectContaining({
      total_pages: 2,
      currentImageIndex: 1,
    }))
    expect(apiPutMock).toHaveBeenCalledTimes(1)
  })

  it('persists the current page payload and updates session progress metadata', async () => {
    const imageStore = useImageStore()
    imageStore.updateImageByIndex(0, {
      translatedDataURL: 'data:image/png;base64,translated-1',
      cleanImageData: 'clean-1',
      hasUnsavedChanges: true,
      bubbleStates: [],
    })

    await saveBookshelfPageProgress(0, 1)

    expect(savePageImageMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', 0, 'translated', 'translated-1')
    expect(savePageImageMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', 0, 'clean', 'clean-1')
    expect(savePageMetaMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', 0, expect.objectContaining({
      fileName: 'page-1.png',
      hasUnsavedChanges: false,
    }))
    expect(saveSessionMetaMock).toHaveBeenCalledWith('bookshelf/book-1/chapters/chapter-1/session', expect.objectContaining({
      total_pages: 2,
      currentImageIndex: 1,
    }))
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
  })

  it('prefers the current chapter context over a stale cached session path', async () => {
    await forceInitializeBookshelfSession()

    const sessionStore = useSessionStore()
    sessionStore.setBookChapterContext('book-2', 'chapter-2', 'Book 2', 'Chapter 2')

    savePageImageMock.mockClear()
    savePageMetaMock.mockClear()
    saveSessionMetaMock.mockClear()

    await saveBookshelfPageProgress(0, 1)

    expect(savePageMetaMock).toHaveBeenCalledWith('bookshelf/book-2/chapters/chapter-2/session', 0, expect.any(Object))
    expect(saveSessionMetaMock).toHaveBeenCalledWith(
      'bookshelf/book-2/chapters/chapter-2/session',
      expect.objectContaining({ currentImageIndex: 1 })
    )
  })
})
