import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const {
  deleteMock,
  getMock,
  patchMock,
  postMock,
  putMock,
  uploadMock,
} = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  patchMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  const apiClient = {
    delete: deleteMock,
    get: getMock,
    patch: patchMock,
    post: postMock,
    put: putMock,
    upload: uploadMock,
  }
  return { apiClient, default: apiClient }
})

function streamFromChunks(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

const document = {
  avatarAssetId: null,
  avatarUrl: null,
  bookId: 'book/id one',
  coreMessages: {},
  createdAt: '2026-07-01T00:00:00Z',
  exportArtifacts: {},
  id: 'doc/id one',
  identity: {
    name: 'Saber',
    aliases: [],
    description: '',
    personality: '',
    scenario: '',
  },
  lorebook: { name: '', entries: [] },
  meta: { title: 'Saber', tags: [] },
  origin: { type: 'manual', source_character: null, source_pages: [] },
  regexScripts: [],
  revision: 3,
  stateTasks: [],
  status: {
    is_favorite: false,
    frozen_sections: [],
    last_diagnostics: {
      valid: true,
      errors: [],
      warnings: ['测试警告'],
      checks: { document: true },
    },
    last_validated_at: '2026-07-01T00:00:00Z',
  },
  title: 'Saber',
  updatedAt: '2026-07-01T00:00:00Z',
}

const session = {
  archived: false,
  documentId: 'doc/id one',
  generation: 1,
  indexRevision: 2,
  messages: [
    {
      messageId: 'msg-user',
      role: 'user',
      content: 'hello',
      attachments: [],
      runtimeLog: [],
      variablesSnapshot: {},
      generationMeta: {},
    },
    {
      messageId: 'msg-assistant',
      role: 'assistant',
      content: 'hello back',
      attachments: [],
      runtimeLog: [],
      variablesSnapshot: {},
      generationMeta: {},
    },
  ],
  revision: 4,
  sessionId: 'session/id one',
  summaryBlocks: [],
  title: 'Chat',
  variables: {},
}

const chatState = {
  activeSession: session,
  availableGreetings: [],
  documentId: 'doc/id one',
  indexRevision: 2,
  sessions: [{
    sessionId: session.sessionId,
    title: session.title,
    revision: session.revision,
    archived: false,
    updatedAt: '2026-07-01T00:00:00Z',
  }],
}

describe('character studio v2 api facade', () => {
  beforeEach(() => {
    vi.resetModules()
    deleteMock.mockReset()
    getMock.mockReset()
    patchMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    uploadMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('uses v2 index, stable document IDs, CAS revisions, and durable generation', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/index')) {
        return Promise.resolve({
          bookId: 'book/id one',
          candidateStatus: { available: true, reason: null },
          documents: [{
            documentId: 'doc/id one',
            title: 'Saber',
            kind: 'manual',
            revision: 3,
            avatarAssetId: null,
            updatedAt: 'now',
          }],
        })
      }
      if (url.endsWith('/candidates')) {
        return Promise.resolve({ available: true, reason: null, items: [] })
      }
      if (url === '/api/v2/studio/documents/doc%2Fid%20one') {
        return Promise.resolve(document)
      }
      if (url === '/api/v2/operations/generate-op') {
        return Promise.resolve({ operationId: 'generate-op', status: 'completed' })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    putMock.mockResolvedValue({ ...document, revision: 4 })
    postMock.mockResolvedValue({ operationId: 'generate-op', status: 'pending' })
    const {
      generateCharacterStudioSection,
      getCharacterStudioDocument,
      getCharacterStudioIndex,
      saveCharacterStudioDocument,
    } = await import('@/api/characterStudio')

    const index = await getCharacterStudioIndex('book/id one')
    const loaded = await getCharacterStudioDocument('book/id one', 'doc/id one')
    await saveCharacterStudioDocument('book/id one', 'doc/id one', loaded.document!)
    await generateCharacterStudioSection('book/id one', 'doc/id one', 'identity')

    expect(index.documents?.[0]).toMatchObject({ id: 'doc/id one', title: 'Saber' })
    expect(loaded.document?.status.last_diagnostics).toEqual({
      valid: true,
      errors: [],
      warnings: ['测试警告'],
      checks: { document: true },
    })
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/studio/books/book%2Fid%20one/index',
    )
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/studio/documents/doc%2Fid%20one',
      expect.objectContaining({
        baseRevision: 3,
        title: 'Saber',
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/documents/doc%2Fid%20one/generate',
      { baseRevision: 4, section: 'identity' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })

  it('exports documents and chat sessions through v2 download routes', async () => {
    const blob = new Blob(['data'], { type: 'application/json' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: vi.fn().mockReturnValue('attachment; filename="studio.json"'),
      },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)
    const {
      downloadCharacterStudioExport,
      exportCharacterStudioChatSession,
    } = await import('@/api/characterStudio')

    await exportCharacterStudioChatSession('book-1', 'doc-a', 'session/id one')
    await downloadCharacterStudioExport('book-1', 'doc/id one', 'v3')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/studio/chat/sessions/session%2Fid%20one/export',
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/studio/documents/doc%2Fid%20one/export?format=v3',
    )
  })

  it('formats the structured backend prompt preview for display', async () => {
    getMock.mockResolvedValue({
      sessionId: 'session/id one',
      promptPreview: {
        system: '角色系统提示',
        messages: [
          { role: 'assistant', content: '你好', assetIds: [] },
          { role: 'user', content: '开始审阅', assetIds: ['asset-1'] },
        ],
        lorebookHits: [{ id: 'entry-1', content: '世界书内容' }],
      },
    })
    const { getCharacterStudioChatPromptPreview } = await import('@/api/characterStudio')

    const result = await getCharacterStudioChatPromptPreview(
      'book-1',
      'doc-a',
      'session/id one',
    )

    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one/prompt-preview',
    )
    expect(result.prompt_preview).toContain('[system]\n角色系统提示')
    expect(result.prompt_preview).toContain('[assistant]\n你好')
    expect(result.prompt_preview).toContain(
      '[user]\n开始审阅\n[assets] asset-1',
    )
    expect(result.prompt_preview).toContain('世界书内容')
  })

  it('creates a durable chat operation before subscribing to its event stream', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/documents/doc%2Fid%20one/chat')) return Promise.resolve(chatState)
      if (url === '/api/v2/operations/chat-op') {
        return Promise.resolve({ operationId: 'chat-op', status: 'completed' })
      }
      if (url === '/api/v2/studio/chat/sessions/session%2Fid%20one') {
        return Promise.resolve(session)
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    postMock.mockResolvedValue({ operationId: 'chat-op', status: 'pending' })
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamFromChunks([
      'event: chunk\n',
      'data: {"eventId":1,"type":"chunk","payload":{"text":"hello"}}\n\n',
    ]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const onEvent = vi.fn()
    const {
      getCharacterStudioChatState,
      streamCharacterStudioChatMessage,
    } = await import('@/api/characterStudio')

    await getCharacterStudioChatState('book/id one', 'doc/id one')
    await streamCharacterStudioChatMessage('book/id one', 'doc/id one', {
      sessionId: 'session/id one',
      content: 'hello',
      onEvent,
    })

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one/messages',
      {
        baseSessionRevision: 4,
        content: 'hello',
        assetIds: [],
      },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v2/operations/chat-op/events?stream=1',
      {
        headers: { Accept: 'text/event-stream' },
        signal: undefined,
      },
    )
    expect(onEvent).toHaveBeenCalledWith({
      type: 'assistant_delta',
      delta: 'hello',
      content: 'hello',
    })
    expect(onEvent).toHaveBeenCalledWith({
      type: 'state',
      session: expect.objectContaining({ session_id: 'session/id one' }),
    })
  })

  it('rehydrates the session after deleting a message chain mutation', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/documents/doc%2Fid%20one/chat')) return Promise.resolve(chatState)
      if (url === '/api/v2/studio/chat/sessions/session%2Fid%20one') {
        return Promise.resolve({
          ...session,
          generation: 2,
          revision: 5,
          messages: session.messages.slice(0, 1),
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    deleteMock.mockResolvedValue({
      sessionId: 'session/id one',
      sessionRevision: 5,
      sessionGeneration: 2,
    })
    const {
      deleteCharacterStudioChatMessage,
      getCharacterStudioChatState,
    } = await import('@/api/characterStudio')

    await getCharacterStudioChatState('book/id one', 'doc/id one')
    const result = await deleteCharacterStudioChatMessage(
      'book/id one',
      'doc/id one',
      'session/id one',
      'msg-assistant',
    )

    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/messages/msg-assistant',
      {
        data: { baseSessionRevision: 4 },
        headers: { 'Idempotency-Key': expect.any(String) },
      },
    )
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one',
    )
    expect(result.session).toMatchObject({
      generation: 2,
      revision: 5,
      session_id: 'session/id one',
    })
  })

  it('surfaces durable operation failures after the event subscription ends', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/documents/doc%2Fid%20one/chat')) return Promise.resolve(chatState)
      if (url === '/api/v2/operations/regenerate-op') {
        return Promise.resolve({
          operationId: 'regenerate-op',
          status: 'failed',
          error: { message: 'stream failed' },
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    postMock.mockResolvedValue({ operationId: 'regenerate-op', status: 'pending' })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(streamFromChunks([]), { status: 200 }),
    ))
    const {
      getCharacterStudioChatState,
      regenerateCharacterStudioChatMessage,
    } = await import('@/api/characterStudio')

    await getCharacterStudioChatState('book/id one', 'doc/id one')
    await expect(regenerateCharacterStudioChatMessage(
      'book/id one',
      'doc/id one',
      'session/id one',
      'msg-assistant',
      vi.fn(),
    )).rejects.toThrow('stream failed')
  })

  it('uses fenced v2 commands for server abort and archived-session deletion', async () => {
    postMock.mockResolvedValue({
      operationId: 'chat-op',
      sessionGeneration: 2,
      sessionRevision: 5,
      status: 'cancelled',
    })
    deleteMock.mockResolvedValue({ deleted: true, sessionId: 'session/id one' })
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/studio/chat/sessions/session%2Fid%20one') {
        return Promise.resolve({ ...session, revision: 5, generation: 2 })
      }
      if (url === '/api/v2/studio/documents/doc%2Fid%20one/chat') {
        return Promise.resolve({
          ...chatState,
          activeSession: null,
          sessions: [],
          indexRevision: 3,
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    const {
      abortCharacterStudioChatOperation,
      deleteCharacterStudioChatSession,
    } = await import('@/api/characterStudio')

    const aborted = await abortCharacterStudioChatOperation(
      'session/id one',
      'chat-op',
    )
    await deleteCharacterStudioChatSession(
      'book/id one',
      'doc/id one',
      'session/id one',
      5,
    )

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one/abort',
      { operationId: 'chat-op' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(aborted).toMatchObject({
      session_id: 'session/id one',
      revision: 5,
      generation: 2,
    })
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one',
      {
        headers: {
          'Idempotency-Key': expect.any(String),
          'If-Match': '5',
        },
      },
    )
  })
})
