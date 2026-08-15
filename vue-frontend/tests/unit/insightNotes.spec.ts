import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { ref } from 'vue'

const {
  createNoteMock,
  deleteNoteMock,
  getNoteDetailMock,
  getNotesMock,
  updateNoteMock,
} = vi.hoisted(() => ({
  createNoteMock: vi.fn(),
  deleteNoteMock: vi.fn(),
  getNoteDetailMock: vi.fn(),
  getNotesMock: vi.fn(),
  updateNoteMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getNotes: getNotesMock,
  getNoteDetail: getNoteDetailMock,
  createNote: createNoteMock,
  updateNote: updateNoteMock,
  deleteNote: deleteNoteMock,
}))

describe('useInsightNotes', () => {
  beforeEach(() => {
    createNoteMock.mockReset()
    deleteNoteMock.mockReset()
    getNoteDetailMock.mockReset()
    getNotesMock.mockReset()
    updateNoteMock.mockReset()
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps the list unchanged and preserves the original error when create fails', async () => {
    createNoteMock.mockRejectedValueOnce(new Error('create failed'))
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    const result = notesState.addNote({ type: 'text', content: 'draft note', pageNum: 3 })

    await expect(result).rejects.toThrow('create failed')
    expect(notesState.notes.value).toEqual([])
    expect(notesState.error.value).toBe('create failed')
  })

  it('always persists new notes even if a caller supplies a client id', async () => {
    createNoteMock.mockResolvedValueOnce({
      id: 'backend-note',
      type: 'text',
      content: 'persist me',
      createdAt: '2026-07-30T00:00:00.000Z',
      updatedAt: '2026-07-30T00:00:00.000Z',
    })
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    const result = await notesState.addNote({
      id: 'client-only-id',
      type: 'text',
      content: 'persist me',
    } as Parameters<typeof notesState.addNote>[0])

    expect(createNoteMock).toHaveBeenCalledOnce()
    expect(result?.id).toBe('backend-note')
    expect(notesState.notes.value[0]?.id).toBe('backend-note')
  })

  it('ignores stale note loads after the selected book changes', async () => {
    let resolveBookOne!: (value: {
      items: Array<Record<string, unknown>>
      nextCursor: string | null
    }) => void
    getNotesMock.mockImplementationOnce(() => new Promise((resolve) => {
      resolveBookOne = resolve
    }))
    const currentBookId = ref('book-1')
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId })

    const pendingLoad = notesState.loadNotes()
    currentBookId.value = 'book-2'
    notesState.clearNotes()

    resolveBookOne({ items: [{
        id: 'note-stale',
        type: 'text',
        content: 'stale note',
        createdAt: '2026-06-25T00:00:00.000Z',
        updatedAt: '2026-06-25T00:00:00.000Z',
      }], nextCursor: null })
    await pendingLoad

    expect(notesState.notes.value).toEqual([])
  })

  it('keeps notes empty when the backend request fails', async () => {
    getNotesMock.mockRejectedValueOnce(new Error('notes service unavailable'))

    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    await notesState.loadNotes()

    expect(notesState.error.value).toBe('notes service unavailable')
    expect(notesState.notes.value).toEqual([])
  })

  it('stores the current note DTO returned by the API facade', async () => {
    getNotesMock.mockResolvedValueOnce({ items: [{
        id: 'note-1',
        type: 'qa',
        content: 'answer',
        pageNum: 12,
        title: '问题记录',
        tags: ['角色'],
        question: '为什么？',
        answer: '因为剧情需要。',
        citations: [{ page: 12, content: '证据' }],
        comment: '保留当前字段',
        createdAt: '2026-06-25T00:00:00.000Z',
        updatedAt: '2026-06-25T00:00:01.000Z',
      }], nextCursor: null })

    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    await notesState.loadNotes()

    expect(notesState.notes.value).toEqual([{
      id: 'note-1',
      type: 'qa',
      content: 'answer',
      pageNum: 12,
      title: '问题记录',
      tags: ['角色'],
      question: '为什么？',
      answer: '因为剧情需要。',
      citations: [{ page: 12, content: '证据' }],
      comment: '保留当前字段',
      createdAt: '2026-06-25T00:00:00.000Z',
      updatedAt: '2026-06-25T00:00:01.000Z',
    }])

    const source = readFileSync(resolve(process.cwd(), 'src/stores/insight/useInsightNotes.ts'), 'utf8')
    expect(source).not.toContain('insightNotesModels')
    expect(source).not.toContain('mapInsightApiNote')
    expect(source).not.toContain('as unknown as NoteData')
    expect(source).not.toContain('使用转换器自动')
    expect(source).not.toContain('/**')
  })

  it('does not project a completed create into a newly selected book', async () => {
    let resolveCreate!: (value: Record<string, unknown>) => void
    createNoteMock.mockImplementationOnce(() => new Promise(resolve => {
      resolveCreate = resolve
    }))
    const currentBookId = ref('book-1')
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId })

    const pendingCreate = notesState.addNote({ type: 'text', content: 'old book' })
    currentBookId.value = 'book-2'
    notesState.clearNotes()
    resolveCreate({
      id: 'old-note',
      type: 'text',
      content: 'old book',
      createdAt: '2026-07-30T00:00:00.000Z',
      updatedAt: '2026-07-30T00:00:00.000Z',
    })

    await expect(pendingCreate).resolves.toMatchObject({ id: 'old-note' })
    expect(notesState.notes.value).toEqual([])
  })

  it('passes only requested update fields so QA metadata is preserved', async () => {
    updateNoteMock.mockResolvedValueOnce({
      id: 'qa-note',
      type: 'qa',
      title: 'new title',
      content: 'answer',
      question: 'question',
      answer: 'answer',
      citations: [{ page: 2, content: 'evidence' }],
      createdAt: '2026-07-30T00:00:00.000Z',
      updatedAt: '2026-07-30T00:00:01.000Z',
    })
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    await expect(notesState.updateNote('qa-note', { title: 'new title' })).resolves.toBeUndefined()

    expect(updateNoteMock).toHaveBeenCalledWith(
      'book-1',
      'qa-note',
      { title: 'new title' },
    )
  })

  it('preserves update and delete errors for their UI owners', async () => {
    updateNoteMock.mockRejectedValueOnce(new Error('update conflict'))
    deleteNoteMock.mockRejectedValueOnce(new Error('delete conflict'))
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    await expect(notesState.updateNote('note-1', { title: 'new title' }))
      .rejects.toThrow('update conflict')
    expect(notesState.error.value).toBe('update conflict')

    await expect(notesState.deleteNote('note-1')).rejects.toThrow('delete conflict')
    expect(notesState.error.value).toBe('delete conflict')
  })

  it('clears pending and error state when notes are reset', async () => {
    getNotesMock.mockImplementationOnce(() => new Promise(() => {}))
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    void notesState.loadNotes()
    expect(notesState.isLoading.value).toBe(true)
    notesState.error.value = 'old error'

    notesState.clearNotes()

    expect(notesState.isLoading.value).toBe(false)
    expect(notesState.isLoadingMore.value).toBe(false)
    expect(notesState.error.value).toBeNull()
  })

  it('ignores stale note detail after the selected book changes', async () => {
    let resolveDetail!: (value: Record<string, unknown>) => void
    getNotesMock.mockResolvedValueOnce({
      items: [{
        id: 'note-1',
        type: 'text',
        content: '',
        createdAt: '2026-06-25T00:00:00.000Z',
        updatedAt: '2026-06-25T00:00:00.000Z',
      }],
      nextCursor: null,
    })
    getNoteDetailMock.mockImplementationOnce(() => new Promise(resolve => {
      resolveDetail = resolve
    }))
    const currentBookId = ref('book-1')
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId })
    await notesState.loadNotes()

    const pendingDetail = notesState.loadNoteDetail('note-1')
    currentBookId.value = 'book-2'
    notesState.clearNotes()
    resolveDetail({
      id: 'note-1',
      type: 'text',
      content: 'stale detail',
      createdAt: '2026-06-25T00:00:00.000Z',
      updatedAt: '2026-06-25T00:00:01.000Z',
    })

    await expect(pendingDetail).resolves.toBeNull()
    expect(notesState.notes.value).toEqual([])
  })

  it('appends subsequent note pages without duplicating existing notes', async () => {
    getNotesMock
      .mockResolvedValueOnce({
        items: [{
          id: 'note-1',
          type: 'text',
          content: 'first',
          createdAt: '2026-06-25T00:00:00.000Z',
          updatedAt: '2026-06-25T00:00:00.000Z',
        }],
        nextCursor: 'cursor-1',
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'note-1',
            type: 'text',
            content: 'first',
            createdAt: '2026-06-25T00:00:00.000Z',
            updatedAt: '2026-06-25T00:00:00.000Z',
          },
          {
            id: 'note-2',
            type: 'text',
            content: 'second',
            createdAt: '2026-06-25T00:00:01.000Z',
            updatedAt: '2026-06-25T00:00:01.000Z',
          },
        ],
        nextCursor: null,
      })

    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    await notesState.loadNotes()
    await notesState.loadMoreNotes()

    expect(getNotesMock).toHaveBeenNthCalledWith(2, 'book-1', undefined, 'cursor-1')
    expect(notesState.notes.value.map(note => note.id)).toEqual(['note-1', 'note-2'])
    expect(notesState.nextCursor.value).toBeNull()
  })

  it('queries the selected note type on the backend and ignores the previous filter response', async () => {
    let resolveAll!: (value: { items: never[]; nextCursor: string | null }) => void
    getNotesMock
      .mockImplementationOnce(() => new Promise(resolve => {
        resolveAll = resolve
      }))
      .mockResolvedValueOnce({
        items: [{
          id: 'qa-note',
          type: 'qa',
          title: '当前问答',
          content: '回答',
          question: '问题',
          citations: [],
          tags: [],
          revision: 1,
          createdAt: '2026-06-25T00:00:00.000Z',
          updatedAt: '2026-06-25T00:00:00.000Z',
        }],
        nextCursor: null,
      })
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    const oldRequest = notesState.loadNotes()
    await notesState.setNoteTypeFilter('qa')
    resolveAll({ items: [], nextCursor: null })
    await oldRequest

    expect(getNotesMock).toHaveBeenNthCalledWith(1, 'book-1', undefined)
    expect(getNotesMock).toHaveBeenNthCalledWith(2, 'book-1', 'qa')
    expect(notesState.notes.value.map(note => note.id)).toEqual(['qa-note'])
  })

  it('keeps note persistence backend-owned', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/useInsightNotes.ts'),
      'utf8',
    )

    expect(source).toContain('getNotes(')
    expect(source).toContain('createNote(')
    expect(source).not.toContain('localStorage')
  })
})
