import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'

describe('bookshelf book CRUD properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  const tagNameArbitrary = fc.string({ minLength: 1, maxLength: 40 })

  const bookArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(tagNameArbitrary, { maxLength: 5 }),
    chapters: fc.constant([]),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  const uniqueBooksArbitrary = (
    constraints: { minLength?: number; maxLength?: number } = {},
  ): fc.Arbitrary<BookData[]> => fc.uniqueArray(bookArbitrary, {
    ...constraints,
    selector: book => book.id,
  })

  it('adds a new book to the front of the list', () => {
    fc.assert(
      fc.property(
        uniqueBooksArbitrary({ maxLength: 10 }),
        bookArbitrary,
        (existingBooks, newBook) => {
          const store = useBookshelfStore()
          store.setBooks(existingBooks)
          const initialCount = store.bookCount

          store.addBook(newBook)

          expect(store.bookCount).toBe(initialCount + 1)
          expect(store.books[0].id).toBe(newBook.id)
          expect(store.getBookById(newBook.id)?.title).toBe(newBook.title)
        }
      ),
      { numRuns: 50 }
    )
  })

  it('removes the selected book id from the list', () => {
    fc.assert(
      fc.property(uniqueBooksArbitrary({ minLength: 1, maxLength: 10 }), (books) => {
        const store = useBookshelfStore()
        store.setBooks(books)
        const initialCount = store.bookCount

        const [bookToDelete] = books
        store.deleteBook(bookToDelete.id)

        expect(store.bookCount).toBe(initialCount - 1)
        expect(store.getBookById(bookToDelete.id)).toBeNull()
      }),
      { numRuns: 50 }
    )
  })

  it('removes every id passed to batch deletion', () => {
    fc.assert(
      fc.property(
        uniqueBooksArbitrary({ minLength: 3, maxLength: 10 }),
        fc.nat({ max: 2 }),
        (books, deleteCount) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          const initialCount = store.bookCount

          const idsToDelete = books.slice(0, deleteCount + 1).map(book => book.id)
          store.deleteBooks(idsToDelete)

          expect(store.bookCount).toBe(initialCount - idsToDelete.length)
          for (const id of idsToDelete) {
            expect(store.getBookById(id)).toBeNull()
          }
        }
      ),
      { numRuns: 50 }
    )
  })

  it('updates only the requested book fields', () => {
    fc.assert(
      fc.property(bookArbitrary, fc.string({ minLength: 1, maxLength: 100 }), (book, newTitle) => {
        const store = useBookshelfStore()
        store.setBooks([book])

        store.updateBook(book.id, { title: newTitle })

        const updated = store.getBookById(book.id)
        expect(updated?.title).toBe(newTitle)
        expect(updated?.id).toBe(book.id)
      }),
      { numRuns: 50 }
    )
  })

  it('ignores deletion requests for unknown book ids', () => {
    fc.assert(
      fc.property(
        uniqueBooksArbitrary({ minLength: 1, maxLength: 10 }),
        fc.uuid(),
        (books, nonExistentId) => {
          if (books.some(book => book.id === nonExistentId)) {
            return
          }

          const store = useBookshelfStore()
          store.setBooks(books)
          const initialCount = store.bookCount

          store.deleteBook(nonExistentId)

          expect(store.bookCount).toBe(initialCount)
        }
      ),
      { numRuns: 50 }
    )
  })
})
