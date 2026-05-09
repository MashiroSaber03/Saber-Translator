import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
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

  it('clears unsaved flags after a successful full chapter save', async () => {
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
  })
})
