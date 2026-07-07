import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, TagData } from '@/types/api'

describe('bookshelf tag batch properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  const tagArbitrary = fc.record({
    name: fc.string({ minLength: 1, maxLength: 20 }),
    color: fc.option(fc.hexaString({ minLength: 6, maxLength: 6 }).map(s => `#${s}`), { nil: undefined }),
  }) as fc.Arbitrary<TagData>

  const tagListArbitrary = (
    constraints: { minLength?: number; maxLength?: number } = {},
  ): fc.Arbitrary<TagData[]> => fc.uniqueArray(tagArbitrary, {
    ...constraints,
    selector: tag => tag.name,
  })

  const bookArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(fc.string({ minLength: 1, maxLength: 20 }), { maxLength: 3 }),
    chapters: fc.constant([]),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  const bookListArbitrary = (
    constraints: { minLength?: number; maxLength?: number } = {},
  ): fc.Arbitrary<BookData[]> => fc.uniqueArray(bookArbitrary, {
    ...constraints,
    selector: book => book.id,
  })

  it('adds every selected tag to every selected book', () => {
    fc.assert(
      fc.property(
        bookListArbitrary({ minLength: 2, maxLength: 5 }),
        tagListArbitrary({ minLength: 1, maxLength: 3 }),
        (books, tags) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          store.setTags(tags)

          const bookIds = books.map(book => book.id)
          const tagNames = tags.map(tag => tag.name)
          store.batchAddTags(bookIds, tagNames)

          for (const bookId of bookIds) {
            const book = store.getBookById(bookId)
            for (const tagName of tagNames) {
              expect(book?.tags).toContain(tagName)
            }
          }
        }
      ),
      { numRuns: 50 }
    )
  })

  it('removes every selected tag from every selected book', () => {
    fc.assert(
      fc.property(tagListArbitrary({ minLength: 2, maxLength: 4 }), (tags) => {
        const store = useBookshelfStore()
        const tagNames = tags.map(tag => tag.name)
        const books: BookData[] = [
          {
            id: 'book-1',
            title: 'Book 1',
            tags: [...tagNames],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
          {
            id: 'book-2',
            title: 'Book 2',
            tags: [...tagNames],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
        ]

        store.setBooks(books)
        store.setTags(tags)

        const tagsToRemove = [tagNames[0]]
        const bookIds = books.map(book => book.id)
        store.batchRemoveTags(bookIds, tagsToRemove)

        for (const bookId of bookIds) {
          expect(store.getBookById(bookId)?.tags).not.toContain(tagsToRemove[0])
        }
      }),
      { numRuns: 50 }
    )
  })

  it('does not duplicate an existing tag on a book', () => {
    fc.assert(
      fc.property(tagArbitrary, (tag) => {
        const store = useBookshelfStore()
        const book: BookData = {
          id: 'book-1',
          title: 'Book',
          tags: [tag.name],
          chapters: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }

        store.setBooks([book])
        store.setTags([tag])
        store.addTagToBook(book.id, tag.name)

        const tagCount = store.getBookById(book.id)?.tags?.filter(name => name === tag.name).length ?? 0
        expect(tagCount).toBe(1)
      }),
      { numRuns: 50 }
    )
  })

  it('ignores removal for a tag that the book does not have', () => {
    fc.assert(
      fc.property(
        tagListArbitrary({ minLength: 1, maxLength: 3 }),
        fc.string({ minLength: 1, maxLength: 20 }),
        (tags, nonExistentTagName) => {
          if (tags.some(tag => tag.name === nonExistentTagName)) {
            return
          }

          const store = useBookshelfStore()
          const tagNames = tags.map(tag => tag.name)
          const book: BookData = {
            id: 'book-1',
            title: 'Book',
            tags: [...tagNames],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          }

          store.setBooks([book])
          store.setTags(tags)

          const originalTags = [...(store.getBookById(book.id)?.tags ?? [])]
          store.removeTagFromBook(book.id, nonExistentTagName)

          expect(store.getBookById(book.id)?.tags).toEqual(originalTags)
        }
      ),
      { numRuns: 50 }
    )
  })

  it('limits batch tag updates to the requested books', () => {
    fc.assert(
      fc.property(tagListArbitrary({ minLength: 1, maxLength: 2 }), (tags) => {
        const store = useBookshelfStore()
        const tagNames = tags.map(tag => tag.name)
        const books: BookData[] = [
          {
            id: 'book-1',
            title: 'Book 1',
            tags: [],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
          {
            id: 'book-2',
            title: 'Book 2',
            tags: [],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
          {
            id: 'book-3',
            title: 'Book 3',
            tags: [],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
        ]

        store.setBooks(books)
        store.setTags(tags)
        store.batchAddTags(['book-1', 'book-2'], tagNames)

        expect(store.getBookById('book-3')?.tags?.length ?? 0).toBe(0)
        expect(store.getBookById('book-1')?.tags?.length).toBeGreaterThan(0)
        expect(store.getBookById('book-2')?.tags?.length).toBeGreaterThan(0)
      }),
      { numRuns: 50 }
    )
  })
})
