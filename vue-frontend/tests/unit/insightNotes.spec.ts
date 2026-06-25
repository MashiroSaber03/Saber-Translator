import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

const {
  createNoteMock,
  getNotesMock,
} = vi.hoisted(() => ({
  createNoteMock: vi.fn(),
  getNotesMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getNotes: getNotesMock,
  createNote: createNoteMock,
}))

describe('useInsightNotes', () => {
  beforeEach(() => {
    localStorage.clear()
    createNoteMock.mockReset()
    getNotesMock.mockReset()
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('rolls back the optimistic note when create fails', async () => {
    createNoteMock.mockRejectedValueOnce(new Error('create failed'))
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    const result = await notesState.addNote({
      type: 'text',
      content: 'draft note',
      pageNum: 3,
    })

    expect(result).toBeNull()
    expect(notesState.notes.value).toEqual([])
    expect(notesState.error.value).toBe('create failed')
    expect(localStorage.getItem('manga_notes_book-1')).toBe('[]')
  })

  it('ignores malformed local notes cache records', async () => {
    localStorage.setItem('manga_notes_book-1', JSON.stringify([
      {
        id: 'note-1',
        type: 'text',
        content: 'valid note',
        pageNum: 3,
        createdAt: '2026-06-23T00:00:00.000Z',
        updatedAt: '2026-06-23T00:00:00.000Z',
      },
      {
        id: 42,
        type: 'unknown',
        content: null,
      },
    ]))

    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId: ref('book-1') })

    notesState.loadNotesFromStorage()

    expect(notesState.notes.value).toEqual([
      {
        id: 'note-1',
        type: 'text',
        content: 'valid note',
        pageNum: 3,
        createdAt: '2026-06-23T00:00:00.000Z',
        updatedAt: '2026-06-23T00:00:00.000Z',
      },
    ])
  })

  it('ignores stale note loads after the selected book changes', async () => {
    let resolveBookOne!: (value: { success: boolean; notes: Array<Record<string, unknown>> }) => void
    getNotesMock.mockImplementationOnce(() => new Promise((resolve) => {
      resolveBookOne = resolve
    }))
    const currentBookId = ref('book-1')
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId })

    const pendingLoad = notesState.loadNotes()
    currentBookId.value = 'book-2'
    notesState.clearNotes()

    resolveBookOne({
      success: true,
      notes: [{
        id: 'note-stale',
        type: 'text',
        content: 'stale note',
        created_at: '2026-06-25T00:00:00.000Z',
        updated_at: '2026-06-25T00:00:00.000Z',
      }],
    })
    await pendingLoad

    expect(notesState.notes.value).toEqual([])
    expect(localStorage.getItem('manga_notes_book-2')).toBeNull()
  })
})
