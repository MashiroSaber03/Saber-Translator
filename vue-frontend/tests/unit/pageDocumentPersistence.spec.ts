import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { reactive } from 'vue'

import { createBubbleState } from '@/utils/bubbleFactory'

const { mutateMock } = vi.hoisted(() => ({
  mutateMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  mutatePageDocument: mutateMock,
}))

describe('page document persistence coordinator', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mutateMock.mockReset()
    vi.resetModules()
  })

  function serverDocument(
    bubbles: Array<Record<string, unknown>>,
    documentRevision: number,
  ) {
    return {
      bubbles,
      chapterId: 'chapter-1',
      defaultFontId: 'font-2',
      documentRevision,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
    }
  }

  it('flushes a bubble delta before the page style CAS without image payloads', async () => {
    mutateMock.mockResolvedValueOnce({
      bubbles: [{
        bubbleId: 'bubble-1',
        fontId: 'font-2',
        ordinal: 1,
        payload: {
          coords: [0, 0, 100, 80],
          fontSize: 28,
          translatedText: '译文',
        },
        updatedRevision: 2,
      }],
      chapterId: 'chapter-1',
      defaultFontId: 'font-2',
      documentRevision: 2,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
    }).mockResolvedValueOnce({
      bubbles: [{
        bubbleId: 'bubble-1',
        fontId: 'font-2',
        ordinal: 1,
        payload: {
          coords: [0, 0, 100, 80],
          fontSize: 28,
          translatedText: '译文',
        },
        updatedRevision: 3,
      }],
      chapterId: 'chapter-1',
      defaultFontId: 'font-2',
      documentRevision: 3,
      pageId: 'page-1',
      pageStyleDefaults: { fontSize: 28 },
      pageStyleSchemaVersion: 1,
    })
    const {
      queuePageDocumentMutation,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: 'font-1',
      documentRevision: 1,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
    })
    const bubble = createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 100, 80],
      fontFamily: 'font-2',
      fontSize: 28,
      polygon: [],
      translatedText: '译文',
    })

    await queuePageDocumentMutation('page-1', 1, [bubble], {
      defaultFontId: 'font-2',
      pageStyleDefaultsPatch: { fontSize: 28 },
      propagateStyleFields: ['fontFamily', 'fontSize'],
    })

    expect(mutateMock).toHaveBeenNthCalledWith(1, 'page-1', {
      baseRevision: 1,
      mutations: [{
        bubbleId: 'bubble-1',
        fields: expect.objectContaining({
          fontId: 'font-2',
          fontSize: 28,
          translatedText: '译文',
        }),
        op: 'create',
      }],
    })
    expect(mutateMock).toHaveBeenNthCalledWith(2, 'page-1', {
      baseRevision: 2,
      defaultFontId: 'font-2',
      mutations: [],
      pageStyleDefaultsPatch: {
        fontSize: 28,
      },
      propagateStyleFields: ['fontFamily', 'fontSize'],
    })
    const request = mutateMock.mock.calls[0]?.[1]
    expect(JSON.stringify(request)).not.toContain('sourceAssetUrl')
    expect(JSON.stringify(request)).not.toContain('base64')
  })

  it('coalesces rapid edits into one 150ms trailing CAS using the latest state', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockResolvedValue(serverDocument([{
        bubbleId: 'bubble-1',
        fontId: 'font-2',
        ordinal: 1,
        payload: {
          coords: [0, 0, 100, 80],
          translatedText: '第二次输入',
        },
        updatedRevision: 2,
      }], 2))
      const {
        queuePageDocumentSave,
        registerPageDocument,
      } = await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))
      const first = createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 100, 80],
        polygon: [],
        translatedText: '第一次输入',
      })
      const second = createBubbleState({
        ...first,
        translatedText: '第二次输入',
      })

      const pending = queuePageDocumentSave('page-1', 1, [first])
      await vi.advanceTimersByTimeAsync(100)
      void queuePageDocumentSave('page-1', 1, [second])
      await vi.advanceTimersByTimeAsync(149)
      expect(mutateMock).not.toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(1)
      await pending

      expect(mutateMock).toHaveBeenCalledTimes(1)
      expect(mutateMock).toHaveBeenCalledWith(
        'page-1',
        expect.objectContaining({
          baseRevision: 1,
          mutations: [
            expect.objectContaining({
              fields: expect.objectContaining({
                translatedText: '第二次输入',
              }),
            }),
          ],
        }),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it('flushes a pending trailing edit immediately', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockResolvedValue(serverDocument([], 2))
      const {
        flushPageDocument,
        queuePageDocumentSave,
        registerPageDocument,
      } = await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))
      const bubble = createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 100, 80],
        polygon: [],
        translatedText: '立即提交',
      })

      void queuePageDocumentSave('page-1', 1, [bubble])
      await flushPageDocument('page-1')

      expect(mutateMock).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('serializes reactive editor bubbles before queueing a backend CAS', async () => {
    mutateMock.mockResolvedValue(serverDocument([], 2))
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))
    const bubbles = reactive([
      createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 100, 80],
        polygon: [],
        translatedText: '来自响应式编辑器',
      }),
    ])

    await queuePageDocumentSave('page-1', 1, bubbles)

    expect(mutateMock).toHaveBeenCalledWith(
      'page-1',
      expect.objectContaining({
        mutations: [
          expect.objectContaining({
            bubbleId: 'bubble-1',
            op: 'create',
          }),
        ],
      }),
    )
  })

  it('keeps only the three most recently registered settled documents', async () => {
    const {
      isPageDocumentRegistered,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')

    for (let index = 1; index <= 4; index += 1) {
      registerPageDocument({
        ...serverDocument([], 1),
        pageId: `page-${index}`,
      })
    }

    expect(isPageDocumentRegistered('page-1')).toBe(false)
    expect(isPageDocumentRegistered('page-2')).toBe(true)
    expect(isPageDocumentRegistered('page-3')).toBe(true)
    expect(isPageDocumentRegistered('page-4')).toBe(true)
  })
})
