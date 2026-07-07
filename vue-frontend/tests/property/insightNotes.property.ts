import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type NoteData, type NoteType } from '@/stores/insightStore'

const insightApiMocks = vi.hoisted(() => ({
  getNotes: vi.fn(),
  createNote: vi.fn(),
  updateNote: vi.fn(),
  deleteNote: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getNotes: insightApiMocks.getNotes,
  createNote: insightApiMocks.createNote,
  updateNote: insightApiMocks.updateNote,
  deleteNote: insightApiMocks.deleteNote,
}))

function configureApiMocks(): void {
  insightApiMocks.getNotes.mockResolvedValue({ success: false })
  insightApiMocks.createNote.mockResolvedValue({ success: true })
  insightApiMocks.updateNote.mockResolvedValue({ success: true })
  insightApiMocks.deleteNote.mockResolvedValue({ success: true })
}

function createStore(): ReturnType<typeof useInsightStore> {
  setActivePinia(createPinia())
  return useInsightStore()
}

async function addNotes(store: ReturnType<typeof useInsightStore>, notes: NoteData[]): Promise<void> {
  for (const note of notes) {
    await store.addNote(note)
  }
}

function expectNewestFirst(actual: NoteData[], source: NoteData[]): void {
  expect(actual).toHaveLength(source.length)
  const expected = [...source].reverse()

  for (const [index, expectedNote] of expected.entries()) {
    const expectedShape: Partial<NoteData> = {
      id: expectedNote.id,
      type: expectedNote.type,
      content: expectedNote.content,
      createdAt: expectedNote.createdAt,
      updatedAt: expectedNote.updatedAt,
    }
    if (expectedNote.pageNum !== undefined) {
      expectedShape.pageNum = expectedNote.pageNum
    }
    expect(actual[index]).toEqual(expect.objectContaining(expectedShape))
  }
}

const noteTypeArbitrary = fc.constantFrom<NoteType>('text', 'qa')
const noteIdArbitrary = fc.stringOf(fc.constantFrom(...'0123456789'.split('')), {
  minLength: 1,
  maxLength: 20,
})
const noteContentArbitrary = fc.string({ minLength: 1, maxLength: 500 })
const pageNumArbitrary = fc.option(fc.integer({ min: 1, max: 1000 }), { nil: undefined })
const isoDateArbitrary = fc.date().map(date => date.toISOString())

const noteDataArbitrary: fc.Arbitrary<NoteData> = fc.record({
  id: noteIdArbitrary,
  type: noteTypeArbitrary,
  content: noteContentArbitrary,
  pageNum: pageNumArbitrary,
  createdAt: isoDateArbitrary,
  updatedAt: isoDateArbitrary,
})

const bookIdArbitrary = fc.stringOf(
  fc.constantFrom(...'abcdef0123456789'.split('')),
  { minLength: 8, maxLength: 8 },
)

describe('insight notes properties', () => {
  beforeEach(() => {
    localStorage.clear()
    configureApiMocks()
  })

  afterEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('persists notes for the selected book in newest-first order', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(noteDataArbitrary, { minLength: 0, maxLength: 10 }),
        bookIdArbitrary,
        async (notes, bookId) => {
          localStorage.clear()
          const store = createStore()
          store.setCurrentBook(bookId)

          await addNotes(store, notes)

          expectNewestFirst(store.notes, notes)
          const stored = localStorage.getItem(`manga_notes_${bookId}`)
          if (notes.length === 0) {
            expect(stored).toBeNull()
          } else {
            expect(stored).toBe(JSON.stringify(store.notes))
          }
        },
      ),
    )
  })

  it('keeps notes isolated by selected book id', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(noteDataArbitrary, { minLength: 1, maxLength: 5 }),
        fc.array(noteDataArbitrary, { minLength: 1, maxLength: 5 }),
        bookIdArbitrary,
        bookIdArbitrary,
        async (notesForFirstBook, notesForSecondBook, firstBookId, secondBookBaseId) => {
          localStorage.clear()
          const secondBookId = secondBookBaseId === firstBookId ? `${secondBookBaseId}_2` : secondBookBaseId

          const firstStore = createStore()
          firstStore.setCurrentBook(firstBookId)
          await addNotes(firstStore, notesForFirstBook)

          const secondStore = createStore()
          secondStore.setCurrentBook(secondBookId)
          await addNotes(secondStore, notesForSecondBook)

          const reloadedFirstStore = createStore()
          reloadedFirstStore.setCurrentBook(firstBookId)
          reloadedFirstStore.loadNotesFromStorage()

          const reloadedSecondStore = createStore()
          reloadedSecondStore.setCurrentBook(secondBookId)
          reloadedSecondStore.loadNotesFromStorage()

          expectNewestFirst(reloadedFirstStore.notes, notesForFirstBook)
          expectNewestFirst(reloadedSecondStore.notes, notesForSecondBook)
        },
      ),
    )
  })

  it('reloads the same note payloads from local storage', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.array(noteDataArbitrary, { minLength: 1, maxLength: 10 }),
        bookIdArbitrary,
        async (notes, bookId) => {
          localStorage.clear()
          const store = createStore()
          store.setCurrentBook(bookId)
          await addNotes(store, notes)

          const reloadedStore = createStore()
          reloadedStore.setCurrentBook(bookId)
          reloadedStore.loadNotesFromStorage()

          expectNewestFirst(reloadedStore.notes, notes)
          expect(reloadedStore.notes).toEqual(store.notes)
        },
      ),
    )
  })
})
