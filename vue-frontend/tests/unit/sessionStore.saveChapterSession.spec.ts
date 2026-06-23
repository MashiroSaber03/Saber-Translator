import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises } from '@vue/test-utils'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'

const {
  persistAllPagesMock,
} = vi.hoisted(() => ({
  persistAllPagesMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/persistenceService', () => ({
  persistAllPages: persistAllPagesMock,
}))

describe('sessionStore.saveChapterSession', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    persistAllPagesMock.mockReset()
    persistAllPagesMock.mockResolvedValue([])
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('does not write routine console logs for normal session state transitions', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const sessionStore = useSessionStore()

    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    sessionStore.clearContext()
    sessionStore.setSessionName('draft-session')
    sessionStore.setSessionList([])
    sessionStore.reset()

    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('clears unsaved flags after a successful full chapter save', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const sessionStore = useSessionStore()

    sessionStore.setBookChapterContext('book-1', 'chapter-1', 'Book', 'Chapter')
    imageStore.addImage('page-1.png', 'data:image/png;base64,one')
    imageStore.addImage('page-2.png', 'data:image/png;base64,two')

    imageStore.updateImageByIndex(0, { hasUnsavedChanges: true })
    imageStore.updateImageByIndex(1, { hasUnsavedChanges: true })

    const success = await sessionStore.saveChapterSession('book-1', 'chapter-1')

    expect(success).toBe(true)
    expect(persistAllPagesMock).toHaveBeenCalledTimes(1)
    expect(imageStore.images[0]?.hasUnsavedChanges).toBe(false)
    expect(imageStore.images[1]?.hasUnsavedChanges).toBe(false)
    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('does not let a previous save completion timer clear active save progress', async () => {
    vi.useFakeTimers()
    const imageStore = useImageStore()
    const sessionStore = useSessionStore()

    imageStore.addImage('page-1.png', 'data:image/png;base64,one')

    await sessionStore.saveChapterSession('book-1', 'chapter-1')
    expect(sessionStore.loadingProgress.message).toBe('保存完成')

    let resolveSecondSave: (() => void) | undefined
    persistAllPagesMock.mockImplementationOnce((_contexts, _runtime, options) => {
      options.onProgress(1, 1)
      return new Promise<void>((resolve) => {
        resolveSecondSave = resolve
      })
    })

    const secondSave = sessionStore.saveChapterSession('book-1', 'chapter-1')
    await flushPromises()

    expect(sessionStore.loadingProgress.message).toBe('保存图片 1/1...')

    await vi.advanceTimersByTimeAsync(1000)
    expect(sessionStore.loadingProgress.message).toBe('保存图片 1/1...')

    resolveSecondSave?.()
    await secondSave
  })
})
