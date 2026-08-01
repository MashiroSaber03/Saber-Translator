import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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
    let resolveBookOne!: (value: Array<Record<string, unknown>>) => void
    getNotesMock.mockImplementationOnce(() => new Promise((resolve) => {
      resolveBookOne = resolve
    }))
    const currentBookId = ref('book-1')
    const { useInsightNotes } = await import('@/stores/insight/useInsightNotes')
    const notesState = useInsightNotes({ currentBookId })

    const pendingLoad = notesState.loadNotes()
    currentBookId.value = 'book-2'
    notesState.clearNotes()

    resolveBookOne([{
        id: 'note-stale',
        type: 'text',
        content: 'stale note',
        createdAt: '2026-06-25T00:00:00.000Z',
        updatedAt: '2026-06-25T00:00:00.000Z',
      }])
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

  it('maps API note payloads through a typed current-schema mapper', async () => {
    getNotesMock.mockResolvedValueOnce([{
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
    expect(source).toContain(
      "import { mapInsightApiNote } from '@/stores/insight/insightNotesModels'"
    )
    expect(source).not.toContain('function isNoteData')
    expect(source).not.toContain('function mapApiNote')
    expect(source).not.toContain('as unknown as NoteData')
    expect(source).not.toContain('使用转换器自动')
    expect(source).not.toContain('/**')
  })

  it('keeps note payload validation in a focused model helper', async () => {
    const { mapInsightApiNote } = await import('@/stores/insight/insightNotesModels')

    expect(mapInsightApiNote({
      id: 'api-note',
      type: 'qa',
      content: 'mapped note',
      pageNum: 7,
      createdAt: '2026-06-25T00:00:00.000Z',
      updatedAt: '2026-06-25T00:00:01.000Z',
    })).toEqual({
      id: 'api-note',
      type: 'qa',
      content: 'mapped note',
      pageNum: 7,
      createdAt: '2026-06-25T00:00:00.000Z',
      updatedAt: '2026-06-25T00:00:01.000Z',
    })

    expect(() => mapInsightApiNote({
      id: 'api-note',
      type: 'qa',
      content: null,
    })).toThrow('笔记响应格式无效')
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
