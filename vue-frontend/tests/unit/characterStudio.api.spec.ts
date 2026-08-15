import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { deleteMock, getMock, patchMock, postMock, putMock, uploadMock } = vi.hoisted(() => ({
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

function durableStudioOperation(
  operationId: string,
  status: 'completed' | 'failed',
  errorMessage?: string,
) {
  return {
    operationId,
    kind: operationId === 'generate-op' ? 'studio_generate' : 'studio_chat',
    executorRole: 'api',
    status,
    pageId: null,
    bubbleId: null,
    studioDocumentId: 'doc/id one',
    studioSessionId: operationId === 'generate-op' ? null : 'session/id one',
    baseRevision: 4,
    baseGeneration: null,
    request: {},
    result: status === 'completed'
      ? operationId === 'generate-op'
        ? { documentId: 'doc/id one', documentRevision: 5 }
        : {
            sessionId: 'session/id one',
            sessionRevision: 5,
            sessionGeneration: 1,
            assistantMessageId: 'msg-assistant',
          }
      : null,
    error: status === 'failed'
      ? { code: 'OPERATION_FAILED', message: errorMessage ?? 'operation failed' }
      : null,
    createdAt: '2026-07-01T00:00:00Z',
    startedAt: '2026-07-01T00:00:01Z',
    finishedAt: '2026-07-01T00:00:02Z',
  }
}

function acceptedStudioOperation(
  operationId: string,
  kind: 'studio_chat' | 'studio_generate' | 'studio_summary',
) {
  return {
    operationId,
    kind,
    status: 'pending',
    executorRole: 'api',
  }
}

const document = {
  avatarAssetId: null,
  avatarUrl: null,
  bookId: 'book/id one',
  coreMessages: {
    first_message: '',
    message_example: '',
    alternate_greetings: [],
    system_prompt: '',
    post_history_instructions: '',
    creator_notes: '',
    character_version: '2.0.0',
  },
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
  origin: { type: 'manual', source_character: null },
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
      checks: { document: true, v3_export: true, v2_export: true },
    },
    last_validated_at: '2026-07-01T00:00:00Z',
  },
  title: 'Saber',
  updatedAt: '2026-07-01T00:00:00Z',
}

const session = {
  archived: false,
  archivedAt: null,
  createdAt: '2026-07-01T00:00:00Z',
  documentId: 'doc/id one',
  generation: 1,
  greetingSource: {},
  indexRevision: 2,
  messages: [
    {
      messageId: 'msg-user',
      ordinal: 1,
      role: 'user',
      content: 'hello',
      attachments: [],
      runtimeLog: [],
      variablesSnapshot: {},
      generationMeta: {},
      createdAt: '2026-07-01T00:00:00Z',
      updatedAt: '2026-07-01T00:00:00Z',
    },
    {
      messageId: 'msg-assistant',
      ordinal: 2,
      role: 'assistant',
      content: 'hello back',
      attachments: [],
      runtimeLog: [],
      variablesSnapshot: {},
      generationMeta: {},
      createdAt: '2026-07-01T00:00:01Z',
      updatedAt: '2026-07-01T00:00:01Z',
    },
  ],
  revision: 4,
  runtimeState: {},
  sessionId: 'session/id one',
  summaryBlocks: [],
  summaryGeneration: 0,
  summaryThroughMessageId: null,
  title: 'Chat',
  updatedAt: '2026-07-01T00:00:01Z',
  variables: {},
}

const chatState = {
  activeSession: session,
  availableGreetings: [],
  documentId: 'doc/id one',
  indexRevision: 2,
  sessions: [
    {
      sessionId: session.sessionId,
      title: session.title,
      revision: session.revision,
      generation: session.generation,
      archived: false,
      archivedAt: null,
      messageCount: session.messages.length,
      lastMessageExcerpt: 'hello back',
      updatedAt: '2026-07-01T00:00:00Z',
    },
  ],
}

describe('character studio v2 api', () => {
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
          documents: [
            {
              documentId: 'doc/id one',
              title: 'Saber',
              kind: 'manual',
              revision: 3,
              avatarAssetId: null,
              hasAvatar: false,
              sourceCharacter: null,
              tags: [],
              isFavorite: false,
              updatedAt: '2026-07-01T00:00:00Z',
            },
          ],
        })
      }
      if (url.endsWith('/candidates')) {
        return Promise.resolve({ available: true, reason: null, items: [] })
      }
      if (url === '/api/v2/studio/documents/doc%2Fid%20one') {
        return Promise.resolve(document)
      }
      if (url === '/api/v2/operations/generate-op') {
        return Promise.resolve(durableStudioOperation('generate-op', 'completed'))
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    putMock.mockResolvedValue({ ...document, revision: 4 })
    postMock.mockResolvedValue(acceptedStudioOperation('generate-op', 'studio_generate'))
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamFromChunks([]), {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }))
    vi.stubGlobal('fetch', fetchMock)
    const {
      generateCharacterStudioSection,
      getCharacterStudioDocument,
      getCharacterStudioIndex,
      saveCharacterStudioDocument,
    } = await import('@/api/characterStudio')

    const index = await getCharacterStudioIndex('book/id one')
    const loaded = await getCharacterStudioDocument('doc/id one')
    const saved = await saveCharacterStudioDocument('doc/id one', loaded)
    await generateCharacterStudioSection('doc/id one', saved.revision, 'identity')

    expect(index.documents?.[0]).toMatchObject({ id: 'doc/id one', title: 'Saber' })
    expect(loaded.status.last_diagnostics).toEqual({
      valid: true,
      errors: [],
      warnings: ['测试警告'],
      checks: { document: true, v3_export: true, v2_export: true },
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/studio/books/book%2Fid%20one/index')
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/studio/documents/doc%2Fid%20one',
      expect.objectContaining({
        baseRevision: 3,
        title: 'Saber',
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/documents/doc%2Fid%20one/generate',
      { baseRevision: 4, section: 'identity' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/v2/operations/generate-op/events?stream=1', {
      headers: { Accept: 'text/event-stream' },
      signal: undefined,
    })
  })

  it('keeps avatar URLs on the document returned by the backend', async () => {
    getMock.mockImplementation((url: string) => {
      const match = url.match(/\/documents\/doc-(\d+)$/)
      if (!match) throw new Error(`Unexpected GET ${url}`)
      const index = Number(match[1])
      return Promise.resolve({
        ...document,
        avatarUrl: `/api/v2/assets/avatar-${index}`,
        id: `doc-${index}`,
      })
    })
    const { getCharacterStudioDocument } = await import('@/api/characterStudio')

    const loadedDocuments = []
    for (let index = 0; index < 25; index += 1) {
      loadedDocuments.push(await getCharacterStudioDocument(`doc-${index}`))
    }

    expect(loadedDocuments[0]?.avatarUrl).toBe('/api/v2/assets/avatar-0')
    expect(loadedDocuments[24]?.avatarUrl).toBe('/api/v2/assets/avatar-24')
  })

  it('creates analysis documents by stable candidate ID without echoing candidate data', async () => {
    postMock.mockResolvedValue({
      ...document,
      origin: { type: 'analysis', source_character: 'Saber' },
    })
    const { createCharacterStudioDocument } = await import('@/api/characterStudio')

    const created = await createCharacterStudioDocument('book/id one', {
      candidate_id: 'candidate/id one',
    })

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/books/book%2Fid%20one/documents',
      { candidateId: 'candidate/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(created.origin).toEqual({
      type: 'analysis',
      source_character: 'Saber',
    })
  })

  it('preserves an empty chat state and imported system messages', async () => {
    getMock
      .mockResolvedValueOnce({
        activeSession: null,
        availableGreetings: [],
        documentId: 'doc/id one',
        indexRevision: 7,
        sessions: [],
      })
      .mockResolvedValueOnce({
        ...chatState,
        activeSession: {
          ...session,
          messages: [
            {
              messageId: 'msg-system',
              ordinal: 1,
              role: 'system',
              content: '导入上下文',
              attachments: [],
              runtimeLog: [],
              variablesSnapshot: {},
              generationMeta: {},
              createdAt: '2026-07-01T00:00:00Z',
              updatedAt: '2026-07-01T00:00:00Z',
            },
          ],
        },
      })
    const { getCharacterStudioChatState } = await import('@/api/characterStudio')

    const emptyState = await getCharacterStudioChatState('doc/id one')
    const importedState = await getCharacterStudioChatState('doc/id one')

    expect(emptyState).toMatchObject({
      index_revision: 7,
      active_session: null,
      archived_sessions: [],
    })
    expect(importedState.active_session?.messages[0]?.role).toBe('system')
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
    const { downloadCharacterStudioExport, exportCharacterStudioChatSession } =
      await import('@/api/characterStudio')

    await exportCharacterStudioChatSession('session/id one')
    await downloadCharacterStudioExport('doc/id one', 'v3')

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/studio/chat/sessions/session%2Fid%20one/export'
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/studio/documents/doc%2Fid%20one/export?format=v3'
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
        lorebookHits: [{ id: 'entry-1', comment: '世界书内容' }],
      },
    })
    const { getCharacterStudioChatPromptPreview } = await import('@/api/characterStudio')

    const result = await getCharacterStudioChatPromptPreview('session/id one')

    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one/prompt-preview'
    )
    expect(result).toContain('[system]\n角色系统提示')
    expect(result).toContain('[assistant]\n你好')
    expect(result).toContain('[user]\n开始审阅\n[assets] asset-1')
    expect(result).toContain('世界书内容')
  })

  it('rejects malformed current document and chat DTOs instead of filling defaults', async () => {
    const invalidSession = { ...session } as Record<string, unknown>
    delete invalidSession.archivedAt
    getMock
      .mockResolvedValueOnce({ ...document, coreMessages: {} })
      .mockResolvedValueOnce({
        ...chatState,
        activeSession: invalidSession,
      })
    const { getCharacterStudioChatState, getCharacterStudioDocument } =
      await import('@/api/characterStudio')

    await expect(getCharacterStudioDocument('doc/id one')).rejects.toThrow(
      '角色文档.coreMessages字段无效'
    )
    await expect(getCharacterStudioChatState('doc/id one')).rejects.toThrow(
      '角色聊天会话字段无效'
    )
  })

  it('keeps structurally valid draft values available for diagnostics and editing', async () => {
    getMock.mockResolvedValue({
      ...document,
      stateTasks: [{
        id: 'draft-task',
        name: '',
        triggerTiming: 'unsupported-draft-value',
        interval: -1,
        commands: '',
        disabled: false,
      }],
    })
    const { getCharacterStudioDocument } = await import('@/api/characterStudio')

    const loaded = await getCharacterStudioDocument('doc/id one')

    expect(loaded.stateTasks).toEqual([{
      id: 'draft-task',
      name: '',
      triggerTiming: 'unsupported-draft-value',
      interval: -1,
      commands: '',
      disabled: false,
    }])
  })

  it('requires the exact Agent SSE lifecycle and non-empty content', async () => {
    const success = [
      'event: ready\ndata: {}\n\n',
      'event: chunk\ndata: {"text":"审阅完成"}\n\n',
      'event: done\ndata: {}\n\n',
    ]
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(streamFromChunks(success), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }))
      .mockResolvedValueOnce(new Response(streamFromChunks([
        'event: ready\ndata: {}\n\n',
        'event: done\ndata: {}\n\n',
      ]), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }))
    vi.stubGlobal('fetch', fetchMock)
    const { runCharacterStudioAgent } = await import('@/api/characterStudio')

    await expect(runCharacterStudioAgent('doc/id one', '审阅')).resolves.toBe('审阅完成')
    await expect(runCharacterStudioAgent('doc/id one', '审阅')).rejects.toThrow(
      'Agent 未返回有效内容'
    )
  })

  it('creates a durable chat operation before subscribing to its event stream', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/documents/doc%2Fid%20one/chat')) return Promise.resolve(chatState)
      if (url === '/api/v2/operations/chat-op') {
        return Promise.resolve(durableStudioOperation('chat-op', 'completed'))
      }
      if (url === '/api/v2/studio/chat/sessions/session%2Fid%20one') {
        return Promise.resolve(session)
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    postMock.mockResolvedValue(acceptedStudioOperation('chat-op', 'studio_chat'))
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          streamFromChunks([
            'id: 1\n',
            'event: chunk\n',
            'data: {"eventId":1,"operationId":"chat-op","type":"chunk","payload":{"text":"hello","totalCharacters":5},"createdAt":"2026-07-01T00:00:01Z"}\n\n',
          ]),
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
        )
      )
    vi.stubGlobal('fetch', fetchMock)
    const onEvent = vi.fn()
    const { getCharacterStudioChatState, streamCharacterStudioChatMessage } =
      await import('@/api/characterStudio')

    await getCharacterStudioChatState('doc/id one')
    await streamCharacterStudioChatMessage({
      sessionId: 'session/id one',
      baseSessionRevision: 4,
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
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/v2/operations/chat-op/events?stream=1', {
      headers: { Accept: 'text/event-stream' },
      signal: undefined,
    })
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
    const { deleteCharacterStudioChatMessage, getCharacterStudioChatState } =
      await import('@/api/characterStudio')

    await getCharacterStudioChatState('doc/id one')
    const result = await deleteCharacterStudioChatMessage('session/id one', 4, 'msg-assistant')

    expect(deleteMock).toHaveBeenCalledWith('/api/v2/studio/chat/messages/msg-assistant', {
      data: { baseSessionRevision: 4 },
      headers: { 'Idempotency-Key': expect.any(String) },
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/studio/chat/sessions/session%2Fid%20one')
    expect(result).toMatchObject({
      generation: 2,
      revision: 5,
      session_id: 'session/id one',
    })
  })

  it('surfaces durable operation failures after the event subscription ends', async () => {
    getMock.mockImplementation((url: string) => {
      if (url.endsWith('/documents/doc%2Fid%20one/chat')) return Promise.resolve(chatState)
      if (url === '/api/v2/operations/regenerate-op') {
        return Promise.resolve(durableStudioOperation('regenerate-op', 'failed', 'stream failed'))
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    postMock.mockResolvedValue(acceptedStudioOperation('regenerate-op', 'studio_chat'))
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(streamFromChunks([]), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      }))
    )
    const { getCharacterStudioChatState, regenerateCharacterStudioChatMessage } =
      await import('@/api/characterStudio')

    await getCharacterStudioChatState('doc/id one')
    await expect(
      regenerateCharacterStudioChatMessage('session/id one', 4, 'msg-assistant', vi.fn())
    ).rejects.toThrow('stream failed')
  })

  it('rejects a completed chat operation without the exact assistant result', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/operations/chat-op') {
        return Promise.resolve({
          ...durableStudioOperation('chat-op', 'completed'),
          result: { sessionId: 'session/id one' },
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    postMock.mockResolvedValue(acceptedStudioOperation('chat-op', 'studio_chat'))
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(streamFromChunks([]), {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    })))
    const { streamCharacterStudioChatMessage } = await import('@/api/characterStudio')

    await expect(streamCharacterStudioChatMessage({
      sessionId: 'session/id one',
      baseSessionRevision: 4,
      content: 'hello',
      onEvent: vi.fn(),
    })).rejects.toThrow('角色聊天操作结果字段无效')
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
    const { abortCharacterStudioChatOperation, deleteCharacterStudioChatSession } =
      await import('@/api/characterStudio')

    const aborted = await abortCharacterStudioChatOperation('session/id one', 'chat-op')
    await deleteCharacterStudioChatSession('doc/id one', 'session/id one', 5)

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/studio/chat/sessions/session%2Fid%20one/abort',
      { operationId: 'chat-op' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(aborted).toMatchObject({
      session_id: 'session/id one',
      revision: 5,
      generation: 2,
    })
    expect(deleteMock).toHaveBeenCalledWith('/api/v2/studio/chat/sessions/session%2Fid%20one', {
      headers: {
        'Idempotency-Key': expect.any(String),
        'If-Match': '5',
      },
    })
  })
})
