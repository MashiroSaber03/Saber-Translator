import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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

  it('hydrates session images through a focused helper', async () => {
    const { hydrateSessionImages } = await import('@/stores/sessionImageHydration')

    const images = hydrateSessionImages([{
      originalDataURL: 'data:image/png;base64,source',
      translatedDataURL: 'data:image/png;base64,translated',
      fileName: 'page.png',
      width: 320,
      height: 480,
      bubbleStates: [{
        coords: [1, 2, 3, 4],
        rotationAngle: 10,
        originalText: '原文',
        translatedText: '译文',
        textboxText: '框文',
        textlines: [],
        ocrResult: null,
      }],
      textlinesPerBubble: [[{ polygon: [[1, 2]], direction: 'h', confidence: 0.8 }]],
      ocrResults: [{
        text: 'OCR',
        confidence: 0.9,
        confidenceSupported: true,
        engine: 'engine',
        primaryEngine: 'engine',
        fallbackUsed: false,
      }],
    }])

    expect(images[0]).toMatchObject({
      originalDataURL: 'data:image/png;base64,source',
      translatedDataURL: 'data:image/png;base64,translated',
      cleanImageData: null,
      width: 320,
      height: 480,
      fileName: 'page.png',
      translationStatus: 'pending',
      translationFailed: false,
      hasUnsavedChanges: false,
    })
    expect(images[0]?.bubbleStates?.[0]?.textlines).toEqual([
      { polygon: [[1, 2]], direction: 'h', confidence: 0.8 },
    ])
    expect(images[0]?.textlinesPerBubble).toEqual([
      [{ polygon: [[1, 2]], direction: 'h', confidence: 0.8 }],
    ])
    expect(images[0]?.ocrResults?.[0]?.text).toBe('OCR')

    const source = readFileSync(resolve(process.cwd(), 'src/stores/sessionStore.ts'), 'utf8')
    expect(source).toContain("import { hydrateSessionImages } from '@/stores/sessionImageHydration'")
    expect(source).not.toContain('sessionData.images.map((img, index)')
    expect(source).not.toContain('function readTextlinesPerBubble')
  })
})
