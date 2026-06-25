import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import * as sessionApi from '@/api/session'

function createSessionPayload(fileName: string) {
  return {
    success: true,
    session: {
      name: fileName,
      version: '2.0',
      savedAt: '2026-06-25T00:00:00.000Z',
      imageCount: 1,
      ui_settings: {},
      currentImageIndex: 0,
      images: [{
        originalDataURL: `data:image/png;base64,${fileName}`,
        fileName,
      }],
    },
  }
}

describe('sessionStore.loadSessionByPath', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  it('ignores stale session loads after a newer session path starts loading', async () => {
    let resolveFirst!: (value: ReturnType<typeof createSessionPayload>) => void
    vi.spyOn(sessionApi, 'loadSessionByPath')
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce(createSessionPayload('second.png'))

    const { useSessionStore } = await import('@/stores/sessionStore')
    const { useImageStore } = await import('@/stores/imageStore')
    const sessionStore = useSessionStore()
    const imageStore = useImageStore()

    const firstLoad = sessionStore.loadSessionByPath('first-session')
    const secondLoad = sessionStore.loadSessionByPath('second-session')
    await secondLoad

    resolveFirst(createSessionPayload('first.png'))
    await firstLoad

    expect(imageStore.images.map(image => image.fileName)).toEqual(['second.png'])
    expect(sessionStore.currentSessionName).toBe('second-session')
  })
})
