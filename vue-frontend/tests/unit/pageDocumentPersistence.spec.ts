import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { reactive } from 'vue'

import { ApiClientError } from '@/api/client'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const { mutateMock } = vi.hoisted(() => ({
  mutateMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  mutatePageDocument: mutateMock,
}))

function bubblePayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    originalText: '',
    translatedText: '',
    textboxText: '',
    coords: [0, 0, 100, 80],
    polygon: [],
    fontSize: 24,
    textDirection: 'vertical',
    autoTextDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#ffffff',
    rotationAngle: 0,
    position: { x: 0, y: 0 },
    strokeEnabled: false,
    strokeColor: '#ffffff',
    strokeWidth: 0,
    lineSpacing: 1.2,
    textAlign: 'center',
    inpaintMethod: 'solid',
    autoFgColor: null,
    autoBgColor: null,
    colorConfidence: 0,
    textlines: [],
    ocrResult: null,
    ...overrides,
  }
}

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
      renderStatus: 'not_rendered' as const,
    }
  }

  function mutationResponse(
    document: ReturnType<typeof serverDocument>,
    mutationResults: Array<{
      bubbleId: string
      clientMutationId: string
      op: 'create' | 'delete' | 'patch' | 'reset'
    }> = [],
  ) {
    return { document, mutationResults }
  }

  function createResponse(
    command: { mutations: Array<{ clientMutationId: string; op: string }> },
    translatedText: string,
    documentRevision: number,
    bubbleId = 'bubble-1',
  ) {
    const mutation = command.mutations.find(item => item.op === 'create')
    if (!mutation) throw new Error('test expected a create mutation')
    return mutationResponse(serverDocument([{
      bubbleId,
      fontId: 'font-2',
      ordinal: 1,
      payload: bubblePayload({
        coords: [0, 0, 100, 80],
        translatedText,
      }),
      updatedRevision: documentRevision,
    }], documentRevision), [{
      bubbleId,
      clientMutationId: mutation.clientMutationId,
      op: 'create',
    }])
  }

  it('flushes a bubble delta before the page style CAS without image payloads', async () => {
    mutateMock.mockImplementationOnce(async (
      _pageId: string,
      command: { mutations: Array<{ clientMutationId: string; op: string }> },
    ) => createResponse(command, '译文', 2)).mockResolvedValueOnce(mutationResponse({
      ...serverDocument([{
        bubbleId: 'bubble-1',
        fontId: 'font-2',
        ordinal: 1,
        payload: bubblePayload({
          coords: [0, 0, 100, 80],
          fontSize: 28,
          translatedText: '译文',
        }),
        updatedRevision: 3,
      }], 3),
      pageStyleDefaults: { fontSize: 28 },
    }))
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
      renderStatus: 'not_rendered',
    })
    const bubble = createBubbleState({
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
        clientMutationId: expect.any(String),
        fields: expect.objectContaining({
          fontId: 'font-2',
          fontSize: 28,
          translatedText: '译文',
        }),
        op: 'create',
      }],
    }, expect.any(String))
    expect(mutateMock).toHaveBeenNthCalledWith(2, 'page-1', {
      baseRevision: 2,
      defaultFontId: 'font-2',
      mutations: [],
      pageStyleDefaultsPatch: {
        fontSize: 28,
      },
      propagateStyleFields: ['fontFamily', 'fontSize'],
    }, expect.any(String))
    const request = mutateMock.mock.calls[0]?.[1]
    expect(request.mutations[0]).not.toHaveProperty('bubbleId')
    expect(JSON.stringify(request)).not.toContain('sourceAssetUrl')
    expect(JSON.stringify(request)).not.toContain('base64')
  })

  it('coalesces rapid edits into one 150ms trailing CAS using the latest state', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockImplementation(async (
        _pageId: string,
        command: { mutations: Array<{ clientMutationId: string; op: string }> },
      ) => createResponse(command, '第二次输入', 2))
      const {
        queuePageDocumentSave,
        registerPageDocument,
      } = await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))
      const first = createBubbleState({
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
        expect.any(String),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it('flushes a pending trailing edit immediately', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockImplementation(async (
        _pageId: string,
        command: { mutations: Array<{ clientMutationId: string; op: string }> },
      ) => createResponse(command, '立即提交', 2))
      const {
        flushPageDocument,
        queuePageDocumentSave,
        registerPageDocument,
      } = await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))
      const bubble = createBubbleState({
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

  it('restores trailing debounce after a flush requested during an active write', async () => {
    vi.useFakeTimers()
    try {
      let resolveFirst!: () => void
      mutateMock.mockImplementationOnce((
        _pageId: string,
        command: { mutations: Array<{ clientMutationId: string; op: string }> },
      ) => new Promise(resolve => {
        resolveFirst = () => resolve(createResponse(command, '第一次输入', 2))
      })).mockResolvedValueOnce(mutationResponse(serverDocument([], 3)))
      const {
        flushPageDocument,
        queuePageDocumentSave,
        registerPageDocument,
      } = await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))

      const first = queuePageDocumentSave('page-1', 1, [createBubbleState({
        translatedText: '第一次输入',
      })])
      await vi.advanceTimersByTimeAsync(150)
      expect(mutateMock).toHaveBeenCalledTimes(1)

      const flush = flushPageDocument('page-1')
      resolveFirst()
      await flush
      await first

      const second = queuePageDocumentSave('page-1', 2, [createBubbleState({
        translatedText: '第二次输入',
      })])
      await vi.advanceTimersByTimeAsync(149)
      expect(mutateMock).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(1)
      await second
      expect(mutateMock).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('serializes reactive editor bubbles before queueing a backend CAS', async () => {
    mutateMock.mockImplementation(async (
      _pageId: string,
      command: { mutations: Array<{ clientMutationId: string; op: string }> },
    ) => createResponse(command, '来自响应式编辑器', 2))
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))
    const bubbles = reactive([
      createBubbleState({
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
            clientMutationId: expect.any(String),
            op: 'create',
          }),
        ],
      }),
      expect.any(String),
    )
    expect(mutateMock.mock.calls[0]?.[1].mutations[0]).not.toHaveProperty('bubbleId')
  })

  it('applies the backend-created bubble ID without clearing the current selection', async () => {
    mutateMock.mockImplementation(async (
      _pageId: string,
      command: { mutations: Array<{ clientMutationId: string; op: string }> },
    ) => createResponse(command, '保持选中', 2, 'server-bubble-1'))
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    const imageStore = useImageStore()
    imageStore.setImages([{
      bubbleStates: null,
      cleanAssetUrl: null,
      fileName: 'page.png',
      hasUnsavedChanges: false,
      id: 'page-1',
      sourceAssetUrl: '/api/v2/assets/source',
      translatedAssetUrl: null,
      translationStatus: 'pending',
    }])
    const bubbleStore = useBubbleStore()
    bubbleStore.setBubbles([createBubbleState({
      coords: [0, 0, 100, 80],
      polygon: [],
      translatedText: '保持选中',
    })], true)
    bubbleStore.selectBubble(0)

    await queuePageDocumentSave('page-1', 1, bubbleStore.bubbles)

    const sentMutation = mutateMock.mock.calls[0]?.[1].mutations[0]
    expect(sentMutation).not.toHaveProperty('bubbleId')
    expect(sentMutation.clientMutationId).toEqual(expect.any(String))
    expect(bubbleStore.selectedIndex).toBe(0)
    expect(bubbleStore.selectedBubble?.backendBubbleId).toBe('server-bubble-1')
    expect(bubbleStore.selectedBubble?.clientMutationId).toBeUndefined()
  })

  it('replays an ambiguous transport failure with the same idempotency key', async () => {
    mutateMock.mockRejectedValueOnce(new ApiClientError({
      code: 'ERR_NETWORK',
      message: 'response lost',
      status: 500,
    })).mockImplementationOnce(async (
      _pageId: string,
      command: { mutations: Array<{ clientMutationId: string; op: string }> },
    ) => createResponse(command, '幂等重放', 2))
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    await queuePageDocumentSave('page-1', 1, [createBubbleState({
      translatedText: '幂等重放',
    })])

    expect(mutateMock).toHaveBeenCalledTimes(2)
    expect(mutateMock.mock.calls[1]?.[1]).toEqual(mutateMock.mock.calls[0]?.[1])
    expect(mutateMock.mock.calls[1]?.[2]).toBe(mutateMock.mock.calls[0]?.[2])
  })

  it('requires an authoritative backend document before queueing mutations', async () => {
    const { queuePageDocumentSave } = await import('@/services/pageDocumentPersistence')

    expect(() => queuePageDocumentSave('missing-page', 1, [createBubbleState()]))
      .toThrow('页面文档 missing-page 尚未从后端注册')
    expect(mutateMock).not.toHaveBeenCalled()
  })

  it('rejects a stale document revision before changing editor bubble identity', async () => {
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 2))
    const bubble = createBubbleState({ translatedText: '过期编辑' })

    expect(() => queuePageDocumentSave('page-1', 1, [bubble])).toThrow(
      '页面文档 page-1 版本已变化：当前为 2，提交版本为 1',
    )
    expect(bubble.clientMutationId).toBeUndefined()
    expect(mutateMock).not.toHaveBeenCalled()
  })

  it('accepts an authoritative reload after a conflict and resumes from its revision', async () => {
    mutateMock.mockRejectedValueOnce(new ApiClientError({
      code: 'page_revision_conflict',
      message: 'revision conflict',
      status: 409,
    })).mockResolvedValueOnce(mutationResponse(serverDocument([{
      bubbleId: 'bubble-1',
      fontId: 'font-2',
      ordinal: 1,
      payload: bubblePayload({ translatedText: '冲突后新编辑' }),
      updatedRevision: 3,
    }], 3)))
    const {
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    await expect(queuePageDocumentSave('page-1', 1, [createBubbleState({
      translatedText: '过期编辑',
    })])).rejects.toThrow('revision conflict')
    const authoritative = registerPageDocument(serverDocument([{
      bubbleId: 'bubble-1',
      fontId: 'font-2',
      ordinal: 1,
      payload: bubblePayload({ translatedText: '后端版本' }),
      updatedRevision: 2,
    }], 2))
    authoritative[0]!.translatedText = '冲突后新编辑'

    await queuePageDocumentSave('page-1', 2, authoritative)

    expect(mutateMock).toHaveBeenNthCalledWith(2, 'page-1', expect.objectContaining({
      baseRevision: 2,
    }), expect.any(String))
  })

  it('does not let failed settled documents grow beyond the three-page cache', async () => {
    mutateMock.mockRejectedValueOnce(new Error('write failed'))
    const {
      isPageDocumentRegistered,
      queuePageDocumentSave,
      registerPageDocument,
    } = await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))
    await expect(queuePageDocumentSave('page-1', 1, [createBubbleState()]))
      .rejects.toThrow('write failed')

    for (let index = 2; index <= 4; index += 1) {
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
