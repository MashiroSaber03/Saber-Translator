import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { reactive } from 'vue'

import { ApiClientError } from '@/api/client'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import { addTestImage } from '../helpers/imageFixtures'

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
    inlineAlign: 'center',
    blockAlign: 'end',
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
  it('marks only a committed manual edit and retains that marker across document reloads', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const images = useImageStore()
    addTestImage(images, 'page.png', '/source', { id: 'page-1', chapterId: 'chapter-1', isManuallyAnnotated: false })
    const row = { bubbleId: 'bubble-1', fontId: 'font-2', ordinal: 1, payload: bubblePayload({ translatedText: '原译文' }), updatedRevision: 1 }
    persistence.registerPageDocument({ ...serverDocument([row], 1), isManuallyAnnotated: false })
    const bubbles = useBubbleStore()
    bubbles.updateBubble(0, { translatedText: '手动校对' })
    expect(images.currentImage!.isManuallyAnnotated).toBe(false)
    mutateMock.mockRejectedValueOnce(new Error('save failed'))
    await expect(persistence.queuePageDocumentSave('page-1', 1, bubbles.bubbles)).rejects.toThrow('save failed')
    expect(images.currentImage!.isManuallyAnnotated).toBe(false)
    const committed = { ...serverDocument([{ ...row, payload: bubblePayload({ translatedText: '手动校对' }), updatedRevision: 2 }], 2), isManuallyAnnotated: true }
    mutateMock.mockResolvedValueOnce(mutationResponse(committed))
    await persistence.queuePageDocumentSave('page-1', 1, bubbles.bubbles)
    expect(images.currentImage!.isManuallyAnnotated).toBe(true)
    persistence.registerPageDocument({ ...serverDocument([row], 1), isManuallyAnnotated: false })
    expect(images.currentImage!.isManuallyAnnotated).toBe(true)
    persistence.registerPageDocument(committed)
    expect(images.currentImage!.isManuallyAnnotated).toBe(true)
  })
  it('publishes one page draft and ignores older reads before the next save', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const images = useImageStore()
    addTestImage(images, 'page.png', '/source', { id: 'page-1', chapterId: 'chapter-1' })
    const row = (text: string, revision: number) => ({ bubbleId: 'bubble-1', fontId: 'font-2', ordinal: 1,
      payload: bubblePayload({ translatedText: text }), updatedRevision: revision })
    persistence.registerPageDocument(serverDocument([row('新版本', 20)], 20))
    const bubbles = useBubbleStore()
    expect(bubbles.bubbles).toBe(images.currentImage!.bubbleStates)
    bubbles.selectBubble(0)
    const retained = persistence.registerPageDocument(serverDocument([row('较早返回的内容', 19)], 19))
    expect(retained[0]!.translatedText).toBe('新版本')
    expect(images.currentImage!.documentRevision).toBe(20)
    expect(bubbles.selectedIndex).toBe(0)
    bubbles.updateBubble(0, { translatedText: '继续编辑' })
    expect(images.currentImage!.bubbleStates![0]!.translatedText).toBe('继续编辑')
    mutateMock.mockResolvedValueOnce(mutationResponse(serverDocument([row('继续编辑', 21)], 21)))
    await persistence.queuePageDocumentSave('page-1', 20, bubbles.bubbles)
    expect(mutateMock.mock.calls[0]![1].baseRevision).toBe(20)
    expect(bubbles.bubbles).toBe(images.currentImage!.bubbleStates)
    expect(bubbles.selectedIndex).toBe(0)
  })
  it('settles the dirty indicator on an unchanged save without a backend mutation or undo entry', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const images = useImageStore()
    addTestImage(images, 'page.png', '/source', { id: 'page-1', chapterId: 'chapter-1', documentRevision: 1, hasUnsavedChanges: true })
    const current = persistence.registerPageDocument(serverDocument([], 1))
    await persistence.queuePageDocumentSave('page-1', 1, current)
    expect(images.currentImage?.hasUnsavedChanges).toBe(false)
    expect(mutateMock).not.toHaveBeenCalled()
  })
  it('keeps failed edits when a background refresh returns the same revision and retries them through the ordinary commit', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const row = { bubbleId: 'bubble-1', fontId: 'font-2', ordinal: 1, updatedRevision: 1,
      payload: bubblePayload({ translatedText: '校对前' }) }
    const document = serverDocument([row], 1)
    const initial = persistence.registerPageDocument(document)
    mutateMock.mockRejectedValueOnce(new Error('offline'))
    await expect(persistence.queuePageDocumentSave('page-1', 1, [{ ...initial[0]!, translatedText: '校对后😀' }])).rejects.toThrow('offline')
    const retained = persistence.registerPageDocument(document)
    expect(retained[0]!.translatedText).toBe('校对后😀')
    mutateMock.mockResolvedValueOnce(mutationResponse(serverDocument([{ ...row, updatedRevision: 2,
      payload: bubblePayload({ translatedText: '校对后😀' }) }], 2)))
    await persistence.queuePageDocumentSave('page-1', 1, retained)
    expect(mutateMock.mock.calls.at(-1)![1].mutations[0].fields.translatedText).toBe('校对后😀')
  })
  it('rebases a pending move onto native point-text bounds changed by a style save', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const saved = (payload: Record<string, unknown>, revision: number) => ({ bubbleId: 'bubble-1', ordinal: 1,
      fontId: 'font-2', updatedRevision: revision, payload: bubblePayload(payload) })
    const initial = persistence.registerPageDocument(serverDocument([saved({}, 1)], 1))
    let complete!: (response: ReturnType<typeof mutationResponse>) => void
    mutateMock.mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
      .mockImplementationOnce(async (_page, command) => {
        expect(command.mutations[0].fields).toEqual({ coords: [1, 0, 161, 80] })
        return mutationResponse(serverDocument([saved({ fontSize: 48, coords: [1, 0, 161, 80] }, 3)], 3))
      })
    const styled = initial.map(bubble => ({ ...bubble, fontSize: 48 }))
    const first = persistence.queuePageDocumentSave('page-1', 1, styled)
    const flush = persistence.flushPageDocument('page-1')
    await vi.waitFor(() => expect(mutateMock).toHaveBeenCalledOnce())
    const moved = persistence.queuePageDocumentSave('page-1', 1, styled.map(bubble => ({ ...bubble, coords: [1, 0, 101, 80] })))
    complete(mutationResponse(serverDocument([saved({ fontSize: 48, coords: [0, 0, 160, 80] }, 2)], 2)))
    await Promise.all([first, moved, flush])
  })

  it('uses the existing create mutation to copy a native text layer and drops the pending copy hint after saving', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const source = { bubbleId: 'source', fontId: 'font-2', ordinal: 1, payload: bubblePayload({ translatedText: '原文字形' }), updatedRevision: 1 }
    const [initial] = persistence.registerPageDocument(serverDocument([source], 1))
    const copy = createBubbleState({ ...initial, backendBubbleId: undefined, duplicateSourceId: 'source', coords: [20, 20, 120, 100] })
    mutateMock.mockImplementationOnce(async (_page, command) => mutationResponse(serverDocument([
      source, { ...source, bubbleId: 'copy', ordinal: 2, payload: bubblePayload({ translatedText: '原文字形', coords: copy.coords }), updatedRevision: 2 },
    ], 2), [{ op: 'create', bubbleId: 'copy', clientMutationId: command.mutations[0].clientMutationId }]))
    await persistence.queuePageDocumentSave('page-1', 1, [initial!, copy])
    expect(mutateMock.mock.calls[0]![1].mutations).toEqual([expect.objectContaining({ op: 'create', sourceBubbleId: 'source' })])
    expect(mutateMock.mock.calls[0]![1].mutations[0].fields).not.toHaveProperty('duplicateSourceId')
    const saved = persistence.registerPageDocument(serverDocument([
      source, { ...source, bubbleId: 'copy', ordinal: 2, payload: bubblePayload({ translatedText: '原文字形', coords: copy.coords }), updatedRevision: 2 },
    ], 2))
    expect(saved[1]!.duplicateSourceId).toBeUndefined()
    expect(saved[1]!.backendBubbleId).toBe('copy')
  })

  it('settles no-op projections so an authoritative undo document can replace the cache', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const first = serverDocument([{
      bubbleId: 'bubble-1', fontId: 'font-2', ordinal: 1, payload: bubblePayload(), updatedRevision: 1,
    }], 1)
    const initial = persistence.registerPageDocument(first)
    // The store's factory and the API adapter use different property order.
    // They are the same persisted bubble and must not block a restored document.
    const fromStore = initial.map(bubble => createBubbleState(bubble))
    expect(JSON.stringify(fromStore)).not.toBe(JSON.stringify(initial))
    await persistence.queuePageDocumentSave('page-1', 1, fromStore)
    expect(mutateMock).not.toHaveBeenCalled()
    const restored = persistence.registerPageDocument({ ...first, documentRevision: 2 })
    await expect(persistence.queuePageDocumentSave('page-1', 2, restored)).resolves.toBeUndefined()
    expect(mutateMock).not.toHaveBeenCalled()
  })

  beforeEach(() => {
    setActivePinia(createPinia())
    mutateMock.mockReset()
    vi.resetModules()
  })

  function serverDocument(bubbles: Array<Record<string, unknown>>, documentRevision: number) {
    return {
      bubbles,
      chapterId: 'chapter-1',
      defaultFontId: 'font-2',
      documentRevision,
      pageId: 'page-1',
      pageStyleDefaults: {},
      renderStatus: 'not_rendered' as const,
    }
  }

  function mutationResponse(
    document: ReturnType<typeof serverDocument>,
    mutationResults: Array<{
      bubbleId: string
      clientMutationId: string
      op: 'create' | 'delete' | 'patch' | 'reset'
    }> = []
  ) {
    return { document, mutationResults }
  }

  function createResponse(
    command: { mutations: Array<{ clientMutationId: string; op: string }> },
    translatedText: string,
    documentRevision: number,
    bubbleId = 'bubble-1'
  ) {
    const mutation = command.mutations.find(item => item.op === 'create')
    if (!mutation) throw new Error('test expected a create mutation')
    return mutationResponse(
      serverDocument(
        [
          {
            bubbleId,
            fontId: 'font-2',
            ordinal: 1,
            payload: bubblePayload({
              coords: [0, 0, 100, 80],
              translatedText,
            }),
            updatedRevision: documentRevision,
          },
        ],
        documentRevision
      ),
      [
        {
          bubbleId,
          clientMutationId: mutation.clientMutationId,
          op: 'create',
        },
      ]
    )
  }

  it('flushes a bubble delta before the page style CAS without image payloads', async () => {
    mutateMock
      .mockImplementationOnce(
        async (
          _pageId: string,
          command: { mutations: Array<{ clientMutationId: string; op: string }> }
        ) => createResponse(command, '译文', 2)
      )
      .mockResolvedValueOnce(
        mutationResponse({
          ...serverDocument(
            [
              {
                bubbleId: 'bubble-1',
                fontId: 'font-2',
                ordinal: 1,
                payload: bubblePayload({
                  coords: [0, 0, 100, 80],
                  fontSize: 28,
                  translatedText: '译文',
                }),
                updatedRevision: 3,
              },
            ],
            3
          ),
          pageStyleDefaults: { fontSize: 28 },
        })
      )
    const { queuePageDocumentMutation, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: 'font-1',
      documentRevision: 1,
      pageId: 'page-1',
      pageStyleDefaults: {},
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

    expect(mutateMock).toHaveBeenNthCalledWith(
      1,
      'page-1',
      {
        baseRevision: 1,
        mutations: [
          {
            clientMutationId: expect.any(String),
            fields: expect.objectContaining({
              fontId: 'font-2',
              fontSize: 28,
              translatedText: '译文',
            }),
            op: 'create',
          },
        ],
      },
      expect.any(String)
    )
    expect(mutateMock).toHaveBeenNthCalledWith(
      2,
      'page-1',
      {
        baseRevision: 2,
        defaultFontId: 'font-2',
        mutations: [],
        pageStyleDefaultsPatch: {
          fontSize: 28,
        },
        propagateStyleFields: ['fontFamily', 'fontSize'],
      },
      expect.any(String)
    )
    const request = mutateMock.mock.calls[0]?.[1]
    expect(request.mutations[0]).not.toHaveProperty('bubbleId')
    expect(JSON.stringify(request)).not.toContain('sourceAssetUrl')
    expect(JSON.stringify(request)).not.toContain('base64')
  })

  it('coalesces rapid edits into one 150ms trailing CAS using the latest state', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockImplementation(
        async (
          _pageId: string,
          command: { mutations: Array<{ clientMutationId: string; op: string }> }
        ) => createResponse(command, '第二次输入', 2)
      )
      const { queuePageDocumentSave, registerPageDocument } =
        await import('@/services/pageDocumentPersistence')
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
        expect.any(String)
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it('flushes a pending trailing edit immediately', async () => {
    vi.useFakeTimers()
    try {
      mutateMock.mockImplementation(
        async (
          _pageId: string,
          command: { mutations: Array<{ clientMutationId: string; op: string }> }
        ) => createResponse(command, '立即提交', 2)
      )
      const { flushPageDocument, queuePageDocumentSave, registerPageDocument } =
        await import('@/services/pageDocumentPersistence')
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

  it('carries projected character styles into a later ordinary edit without replacing new input', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const left = { runs: [{ len: 3, style: { size_pt: 24 } }], paragraphs: [{ len: 3, style: { align: 'Left' } }] }
    const centered = { runs: [{ len: 3, style: { size_pt: 28 } }], paragraphs: [{ len: 3, style: { align: 'Center' } }] }
    const savedBubble = (payload: Record<string, unknown>, revision: number) => ({
      bubbleId: 'bubble-1', fontId: 'font-2', ordinal: 1,
      payload: bubblePayload({ inlineAlign: 'start', richText: left, ...payload }), updatedRevision: revision,
    })
    const initial = persistence.registerPageDocument(serverDocument([savedBubble({}, 1)], 1))
    let complete!: (value: ReturnType<typeof mutationResponse>) => void
    mutateMock.mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
      .mockImplementationOnce(async (_page, command) => {
        const fields = command.mutations[0].fields
        expect(command.mutations[0].op).toBe('patch')
        expect(fields).toEqual({ blockAlign: 'center' })
        const payload = { fontSize: 28, inlineAlign: 'center', richText: centered, ...fields }
        return mutationResponse(serverDocument([savedBubble(payload, 3)], 3))
      })
    const first = initial.map(b => ({ ...b, fontSize: 28, inlineAlign: 'center' as const }))
    const pending = persistence.queuePageDocumentSave('page-1', 1, first)
    const flushed = persistence.flushPageDocument('page-1')
    await vi.waitFor(() => expect(mutateMock).toHaveBeenCalledTimes(1))
    const second = persistence.queuePageDocumentSave('page-1', 1, first.map(b => ({ ...b, blockAlign: 'center' })))
    complete(mutationResponse(serverDocument([savedBubble({ fontSize: 28, inlineAlign: 'center', richText: centered }, 2)], 2)))
    await Promise.all([pending, second, flushed])
    expect(mutateMock).toHaveBeenCalledTimes(2)
  })

  it('submits an explicit same-value box style without sending its cached character runs', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const initial = persistence.registerPageDocument(serverDocument([{
      bubbleId: 'bubble-1', ordinal: 1, fontId: 'font-2', updatedRevision: 1,
      payload: bubblePayload({ richText: { runs: [
        { len: 1, style: { size_pt: 24 } }, { len: 1, style: { size_pt: 48 } },
      ], paragraphs: [] } }),
    }], 1))
    mutateMock.mockImplementationOnce(async (_page, command) => {
      expect(command.mutations).toEqual([{
        bubbleId: 'bubble-1', clientMutationId: expect.any(String), op: 'patch', fields: { fontSize: 24 },
      }])
      return mutationResponse(serverDocument([{
        bubbleId: 'bubble-1', ordinal: 1, fontId: 'font-2', updatedRevision: 2,
        payload: bubblePayload({ richText: { runs: [{ len: 2, style: { size_pt: 24 } }], paragraphs: [] } }),
      }], 2))
    })
    await persistence.queuePageDocumentMutation('page-1', 1, initial, { bubbleFields: { 'bubble-1': ['fontSize'] } })
    expect(mutateMock).toHaveBeenCalledOnce()
  })

  it('consumes explicit fields once when rendering queues the identical pending document', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    const savedBubble = (payload: Record<string, unknown>, revision: number) => ({ bubbleId: 'bubble-1', ordinal: 1,
      fontId: 'font-2', updatedRevision: revision, payload: bubblePayload(payload) })
    const initial = persistence.registerPageDocument(serverDocument([savedBubble({}, 1)], 1))
    let complete!: (response: ReturnType<typeof mutationResponse>) => void
    mutateMock.mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
    const changed = initial.map(b => ({ ...b, fontSize: 36 }))
    const save = persistence.queuePageDocumentMutation('page-1', 1, changed, { bubbleFields: { 'bubble-1': ['fontSize'] } })
    const flush = persistence.flushPageDocument('page-1')
    await vi.waitFor(() => expect(mutateMock).toHaveBeenCalledOnce())
    const renderSave = persistence.queuePageDocumentSave('page-1', 1, changed)
    complete(mutationResponse(serverDocument([savedBubble({ fontSize: 36 }, 2)], 2)))
    await Promise.all([save, flush, renderSave])
    expect(mutateMock).toHaveBeenCalledOnce()
  })

  it('restores trailing debounce after a flush requested during an active write', async () => {
    vi.useFakeTimers()
    try {
      let resolveFirst!: () => void
      mutateMock
        .mockImplementationOnce(
          (
            _pageId: string,
            command: { mutations: Array<{ clientMutationId: string; op: string }> }
          ) =>
            new Promise(resolve => {
              resolveFirst = () => resolve(createResponse(command, '第一次输入', 2))
            })
        )
        .mockResolvedValueOnce(mutationResponse(serverDocument([], 3)))
      const { flushPageDocument, queuePageDocumentSave, registerPageDocument } =
        await import('@/services/pageDocumentPersistence')
      registerPageDocument(serverDocument([], 1))

      const first = queuePageDocumentSave('page-1', 1, [
        createBubbleState({
          translatedText: '第一次输入',
        }),
      ])
      await vi.advanceTimersByTimeAsync(150)
      expect(mutateMock).toHaveBeenCalledTimes(1)

      const flush = flushPageDocument('page-1')
      resolveFirst()
      await flush
      await first

      const second = queuePageDocumentSave('page-1', 2, [
        createBubbleState({
          translatedText: '第二次输入',
        }),
      ])
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
    mutateMock.mockImplementation(
      async (
        _pageId: string,
        command: { mutations: Array<{ clientMutationId: string; op: string }> }
      ) => createResponse(command, '来自响应式编辑器', 2)
    )
    const { queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
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
      expect.any(String)
    )
    expect(mutateMock.mock.calls[0]?.[1].mutations[0]).not.toHaveProperty('bubbleId')
  })

  it('applies the backend-created bubble ID without clearing the current selection', async () => {
    mutateMock.mockImplementation(
      async (
        _pageId: string,
        command: { mutations: Array<{ clientMutationId: string; op: string }> }
      ) => createResponse(command, '保持选中', 2, 'server-bubble-1')
    )
    const { queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    const imageStore = useImageStore()
    imageStore.setImages([
      {
        bubbleStates: null,
        cleanAssetUrl: null,
        fileName: 'page.png',
        hasUnsavedChanges: false,
        id: 'page-1',
        sourceAssetUrl: '/api/v2/assets/source',
        translatedAssetUrl: null,
        translationStatus: 'pending',
      },
    ])
    const bubbleStore = useBubbleStore()
    bubbleStore.setBubbles(
      [
        createBubbleState({
          coords: [0, 0, 100, 80],
          polygon: [],
          translatedText: '保持选中',
        }),
      ],
      true
    )
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
    mutateMock
      .mockRejectedValueOnce(
        new ApiClientError({
          code: 'network_error',
          message: 'response lost',
          status: 0,
        })
      )
      .mockImplementationOnce(
        async (
          _pageId: string,
          command: { mutations: Array<{ clientMutationId: string; op: string }> }
        ) => createResponse(command, '幂等重放', 2)
      )
    const { queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    await queuePageDocumentSave('page-1', 1, [
      createBubbleState({
        translatedText: '幂等重放',
      }),
    ])

    expect(mutateMock).toHaveBeenCalledTimes(2)
    expect(mutateMock.mock.calls[1]?.[1]).toEqual(mutateMock.mock.calls[0]?.[1])
    expect(mutateMock.mock.calls[1]?.[2]).toBe(mutateMock.mock.calls[0]?.[2])
  })

  it('requires an authoritative backend document before queueing mutations', async () => {
    const { queuePageDocumentSave } = await import('@/services/pageDocumentPersistence')

    expect(() => queuePageDocumentSave('missing-page', 1, [createBubbleState()])).toThrow(
      '页面文档 missing-page 尚未从后端注册'
    )
    expect(mutateMock).not.toHaveBeenCalled()
  })

  it('rejects a stale document revision before changing editor bubble identity', async () => {
    const { queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 2))
    const bubble = createBubbleState({ translatedText: '过期编辑' })

    expect(() => queuePageDocumentSave('page-1', 1, [bubble])).toThrow(
      '页面文档 page-1 版本已变化：当前为 2，提交版本为 1'
    )
    expect(bubble.clientMutationId).toBeUndefined()
    expect(mutateMock).not.toHaveBeenCalled()
  })

  it('accepts an authoritative reload after a conflict and resumes from its revision', async () => {
    mutateMock
      .mockRejectedValueOnce(
        new ApiClientError({
          code: 'page_revision_conflict',
          message: 'revision conflict',
          status: 409,
        })
      )
      .mockResolvedValueOnce(
        mutationResponse(
          serverDocument(
            [
              {
                bubbleId: 'bubble-1',
                fontId: 'font-2',
                ordinal: 1,
                payload: bubblePayload({ translatedText: '冲突后新编辑' }),
                updatedRevision: 3,
              },
            ],
            3
          )
        )
      )
    const { queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))

    await expect(
      queuePageDocumentSave('page-1', 1, [
        createBubbleState({
          translatedText: '过期编辑',
        }),
      ])
    ).rejects.toThrow('revision conflict')
    const authoritative = registerPageDocument(
      serverDocument(
        [
          {
            bubbleId: 'bubble-1',
            fontId: 'font-2',
            ordinal: 1,
            payload: bubblePayload({ translatedText: '后端版本' }),
            updatedRevision: 2,
          },
        ],
        2
      )
    )
    authoritative[0]!.translatedText = '冲突后新编辑'

    await queuePageDocumentSave('page-1', 2, authoritative)

    expect(mutateMock).toHaveBeenNthCalledWith(
      2,
      'page-1',
      expect.objectContaining({
        baseRevision: 2,
      }),
      expect.any(String)
    )
  })

  it('does not let failed settled documents grow beyond the three-page cache', async () => {
    mutateMock.mockRejectedValueOnce(new Error('write failed'))
    const { isPageDocumentRegistered, queuePageDocumentSave, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')
    registerPageDocument(serverDocument([], 1))
    await expect(queuePageDocumentSave('page-1', 1, [createBubbleState()])).rejects.toThrow(
      'write failed'
    )

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

  it('keeps a native edit arriving during an in-flight save and never resets it from stale bubbles', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    persistence.registerPageDocument(serverDocument([], 1))
    let complete!: (value: ReturnType<typeof mutationResponse>) => void
    const save = vi.fn().mockImplementationOnce(() => new Promise(resolve => { complete = resolve }))
      .mockResolvedValueOnce(mutationResponse(serverDocument([], 3)))
    const unregister = persistence.registerPageEditor('page-1', { save })
    const first = persistence.queueEditorDocumentSave('page-1', 1)
    const flushed = persistence.flushPageDocument('page-1')
    await vi.waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    const second = persistence.queueEditorDocumentSave('page-1', 2)
    complete(mutationResponse(serverDocument([], 2)))
    await Promise.all([first, second, flushed])
    expect(save.mock.calls).toEqual([[1], [2]])
    expect(mutateMock).not.toHaveBeenCalled()
    unregister()
  })

  it('does not upload the same native revision again when Save is pressed during autosave', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    persistence.registerPageDocument(serverDocument([], 1))
    let complete!: (value: ReturnType<typeof mutationResponse>) => void
    const save = vi.fn(() => new Promise<ReturnType<typeof mutationResponse>>(resolve => { complete = resolve }))
    const unregister = persistence.registerPageEditor('page-1', { save })
    const automatic = persistence.queueEditorDocumentSave('page-1', 1)
    const flushed = persistence.flushPageDocument('page-1')
    await vi.waitFor(() => expect(save).toHaveBeenCalledOnce())
    const manual = persistence.queueEditorDocumentSave('page-1', 1)
    complete(mutationResponse(serverDocument([], 2)))
    await Promise.all([automatic, manual, flushed])
    await persistence.queueEditorDocumentSave('page-1', 1)
    expect(save).toHaveBeenCalledOnce()
    unregister()
  })

  it('retains failed native changes for retry and uses the ordinary API after leaving fine editing', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    persistence.registerPageDocument(serverDocument([], 1))
    const save = vi.fn().mockRejectedValueOnce(new Error('native upload failed'))
      .mockResolvedValueOnce(mutationResponse(serverDocument([], 2)))
    const unregister = persistence.registerPageEditor('page-1', { save })
    await expect(persistence.queueEditorDocumentSave('page-1', 1)).rejects.toThrow('native upload failed')
    expect(persistence.hasPendingPageDocument('page-1')).toBe(true)
    await persistence.queueEditorDocumentSave('page-1', 1)
    expect(save).toHaveBeenCalledTimes(2)
    unregister()
    mutateMock.mockImplementationOnce(async (_page, command) => createResponse(command, '普通编辑', 3))
    await persistence.queuePageDocumentSave('page-1', 2, [createBubbleState({ translatedText: '普通编辑' })])
    expect(mutateMock).toHaveBeenCalledTimes(1)
  })

  it('replays ambiguous native transport with the same idempotency key', async () => {
    const { withDocumentTransportReplay } = await import('@/services/pageDocumentPersistence')
    const submit = vi.fn().mockRejectedValueOnce(new ApiClientError({code:'network_error', status:0, message:'offline'}))
      .mockResolvedValueOnce('saved')
    expect(await withDocumentTransportReplay(submit)).toBe('saved')
    expect(submit.mock.calls[0]).toEqual(submit.mock.calls[1])
  })

  it('saves native character changes before applying an overlapping page style command', async () => {
    const persistence = await import('@/services/pageDocumentPersistence')
    persistence.registerPageDocument(serverDocument([], 1))
    const sequence: string[] = []
    const save = vi.fn(async () => { sequence.push('native'); return mutationResponse(serverDocument([], 2)) })
    const unregister = persistence.registerPageEditor('page-1', { save })
    mutateMock.mockImplementationOnce(async (_page, command) => {
      sequence.push('style')
      expect(command.baseRevision).toBe(2)
      return mutationResponse(serverDocument([], 3))
    })
    const native = persistence.queueEditorDocumentSave('page-1', 1)
    const style = persistence.queuePageDocumentMutation('page-1', 1, [], { pageStyleDefaultsPatch: { fillColor: '#ffffff' } })
    await Promise.all([native, style, persistence.flushPageDocument('page-1')])
    expect(sequence).toEqual(['native', 'style'])
    unregister()
  })

  it('keeps only the three most recently registered settled documents', async () => {
    const { isPageDocumentRegistered, registerPageDocument } =
      await import('@/services/pageDocumentPersistence')

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
