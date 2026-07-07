import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, putMock, deleteMock, uploadMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  deleteMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    put: putMock,
    delete: deleteMock,
    upload: uploadMock,
  },
}))

function streamFromChunks(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}

describe('character studio api downloads', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    deleteMock.mockReset()
    uploadMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('routes document and chat endpoints through encoded path helpers', async () => {
    getMock.mockResolvedValue({ success: true })
    postMock.mockResolvedValue({ success: true })
    putMock.mockResolvedValue({ success: true })
    deleteMock.mockResolvedValue({ success: true })

    const {
      createCharacterStudioDocument,
      deleteCharacterStudioChatMessage,
      deleteCharacterStudioDocument,
      editCharacterStudioChatMessage,
      generateCharacterStudioSection,
      getCharacterStudioChatPromptPreview,
      getCharacterStudioDocument,
      getCharacterStudioIndex,
      saveCharacterStudioDocument,
      switchCharacterStudioChatSession,
    } = await import('@/api/characterStudio')

    const bookId = 'book/id one'
    const docId = 'doc/id one'
    const encodedBase = '/api/manga-insight/book%2Fid%20one/character-studio'
    const encodedDocument = `${encodedBase}/documents/doc%2Fid%20one`

    await getCharacterStudioIndex(bookId)
    await createCharacterStudioDocument(bookId, { title: 'New Card' })
    await getCharacterStudioDocument(bookId, docId)
    await saveCharacterStudioDocument(bookId, docId, { title: 'Saved' })
    await deleteCharacterStudioDocument(bookId, docId)
    await generateCharacterStudioSection(bookId, docId, 'profile')
    await switchCharacterStudioChatSession(bookId, docId, 'session/id one')
    await editCharacterStudioChatMessage(bookId, docId, 'session/id one', 'message/id one', 'updated')
    await deleteCharacterStudioChatMessage(bookId, docId, 'session/id one', 'message/id one')
    await getCharacterStudioChatPromptPreview(bookId, docId, 'session/id one')

    expect(getMock).toHaveBeenNthCalledWith(1, `${encodedBase}/index`)
    expect(postMock).toHaveBeenNthCalledWith(1, `${encodedBase}/documents`, { title: 'New Card' })
    expect(getMock).toHaveBeenNthCalledWith(2, encodedDocument)
    expect(putMock).toHaveBeenNthCalledWith(1, encodedDocument, { title: 'Saved' })
    expect(deleteMock).toHaveBeenNthCalledWith(1, encodedDocument)
    expect(postMock).toHaveBeenNthCalledWith(2, `${encodedDocument}/generate/profile`, {})
    expect(postMock).toHaveBeenNthCalledWith(3, `${encodedDocument}/chat/sessions/session%2Fid%20one/activate`, {})
    expect(putMock).toHaveBeenNthCalledWith(2, `${encodedDocument}/chat/messages/message%2Fid%20one`, {
      session_id: 'session/id one',
      content: 'updated',
    })
    expect(deleteMock).toHaveBeenNthCalledWith(
      2,
      `${encodedDocument}/chat/messages/message%2Fid%20one?session_id=session%2Fid+one`,
    )
    expect(getMock).toHaveBeenNthCalledWith(
      3,
      `${encodedDocument}/chat/prompt-preview?session_id=session%2Fid+one`,
    )
  })

  it('exports chat sessions with shared filename parsing', async () => {
    const blob = new Blob(['chat'], { type: 'application/json' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: vi.fn().mockImplementation((name: string) => (
          name.toLowerCase() === 'content-disposition'
            ? 'attachment; filename="chat-session.json"'
            : null
        )),
      },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { exportCharacterStudioChatSession } = await import('@/api/characterStudio')
    const result = await exportCharacterStudioChatSession('book-1', 'doc-a', 'session/one')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/manga-insight/book-1/character-studio/documents/doc-a/chat/export?session_id=session%2Fone'
    )
    expect(result).toEqual({ blob, filename: 'chat-session.json' })
  })

  it('uses download fallback filenames and shared json errors', async () => {
    const successBlob = new Blob(['card'], { type: 'application/json' })
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        headers: { get: vi.fn().mockReturnValue(null) },
        blob: vi.fn().mockResolvedValue(successBlob),
      })
      .mockResolvedValueOnce(new Response(JSON.stringify({ message: 'studio export failed' }), { status: 500 }))
    vi.stubGlobal('fetch', fetchMock)

    const { downloadCharacterStudioExport, downloadCharacterStudioWorldbook } = await import('@/api/characterStudio')

    await expect(downloadCharacterStudioExport('book-1', 'doc-a', 'json')).resolves.toEqual({
      blob: successBlob,
      filename: 'doc-a.json',
    })
    await expect(downloadCharacterStudioWorldbook('book-1', 'doc-a')).rejects.toThrow('studio export failed')
  })

  it('streams chat events through the shared SSE reader', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamFromChunks([
      'event: assistant_delta\n',
      'data: {"delta":"hel',
      'lo","content":"hello"}\n\n',
      'event: assistant_done\ndata: {"message_id":"msg-1","content":"hello"}\n\n',
    ]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const onEvent = vi.fn()

    const { streamCharacterStudioChatMessage } = await import('@/api/characterStudio')
    await streamCharacterStudioChatMessage('book-1', 'doc-a', {
      sessionId: 'session-1',
      content: 'hello',
      onEvent,
    })

    expect(onEvent).toHaveBeenCalledWith({
      type: 'assistant_delta',
      delta: 'hello',
      content: 'hello',
    })
    expect(onEvent).toHaveBeenCalledWith({
      type: 'assistant_done',
      message_id: 'msg-1',
      content: 'hello',
    })
  })

  it('throws when a streamed chat error event arrives', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(streamFromChunks([
      'event: error\ndata: {"message":"stream failed"}\n\n',
    ]), { status: 200 })))

    const { regenerateCharacterStudioChatMessage } = await import('@/api/characterStudio')

    await expect(regenerateCharacterStudioChatMessage(
      'book-1',
      'doc-a',
      'session-1',
      'msg-1',
      vi.fn(),
    )).rejects.toThrow('stream failed')
  })
})
