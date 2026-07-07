import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import * as bookshelfApi from '@/api/bookshelf'
import type { BookData, TagData } from '@/types/api'

const bookIdArbitrary = fc.uuid()
const bookTitleArbitrary = fc.string({ minLength: 1, maxLength: 100 })
const bookDescriptionArbitrary = fc.option(fc.string({ maxLength: 500 }), { nil: undefined })
const dateStringArbitrary = fc.date({ min: new Date('2020-01-01'), max: new Date('2025-12-31') })
  .map(d => d.toISOString())

const tagDataArbitrary = fc.record({
  name: fc.string({ minLength: 1, maxLength: 50 }),
  color: fc.option(fc.hexaString({ minLength: 6, maxLength: 6 }).map(h => `#${h}`), { nil: undefined }),
}) as fc.Arbitrary<TagData>

const uniqueTagsArbitrary = (
  constraints: { minLength?: number; maxLength?: number } = {},
): fc.Arbitrary<TagData[]> => fc.uniqueArray(tagDataArbitrary, {
  ...constraints,
  selector: tag => tag.name,
})

const bookDataArbitrary = (tagNames: string[]): fc.Arbitrary<BookData> => fc.record({
  id: bookIdArbitrary,
  title: bookTitleArbitrary,
  description: bookDescriptionArbitrary,
  cover: fc.option(fc.string(), { nil: undefined }),
  tags: fc.option(
    fc.subarray(tagNames, { minLength: 0, maxLength: Math.min(tagNames.length, 5) }),
    { nil: undefined }
  ),
  chapters: fc.constant(undefined),
  createdAt: dateStringArbitrary,
  updatedAt: dateStringArbitrary,
})

const minimalBookArbitrary = fc.record({
  id: bookIdArbitrary,
  title: bookTitleArbitrary,
  description: bookDescriptionArbitrary,
  createdAt: dateStringArbitrary,
  updatedAt: dateStringArbitrary,
}) as fc.Arbitrary<BookData>

const selectableBookArbitrary = fc.record({
  id: bookIdArbitrary,
  title: bookTitleArbitrary,
  createdAt: dateStringArbitrary,
  updatedAt: dateStringArbitrary,
}) as fc.Arbitrary<BookData>

const uniqueBooksArbitrary = (
  arbitrary: fc.Arbitrary<BookData>,
  constraints: { minLength?: number; maxLength?: number } = {},
): fc.Arbitrary<BookData[]> => fc.uniqueArray(arbitrary, {
  ...constraints,
  selector: book => book.id,
})

const bookshelfDataArbitrary = uniqueTagsArbitrary({ maxLength: 10 })
  .chain(tags => {
    const tagNames = tags.map(tag => tag.name)
    return fc.tuple(
      fc.constant(tags),
      uniqueBooksArbitrary(bookDataArbitrary(tagNames), { maxLength: 20 })
    )
  })

describe('bookshelf store properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue({
      success: false,
      error: 'Property test keeps backend refresh outside this invariant',
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('filters books by search keyword across title and description', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          uniqueBooksArbitrary(minimalBookArbitrary, { maxLength: 20 }),
          fc.string({ maxLength: 20 })
        ),
        ([books, keyword]) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          store.setSearchKeyword(keyword)

          const filtered = store.filteredBooks
          const normalizedKeyword = keyword.toLowerCase().trim()

          if (normalizedKeyword === '') {
            expect(filtered).toHaveLength(books.length)
            return
          }

          for (const book of filtered) {
            const titleMatch = book.title.toLowerCase().includes(normalizedKeyword)
            const descriptionMatch = book.description?.toLowerCase().includes(normalizedKeyword) ?? false
            expect(titleMatch || descriptionMatch).toBe(true)
          }

          for (const book of books) {
            const titleMatch = book.title.toLowerCase().includes(normalizedKeyword)
            const descriptionMatch = book.description?.toLowerCase().includes(normalizedKeyword) ?? false
            const isIncluded = filtered.some(filteredBook => filteredBook.id === book.id)
            expect(isIncluded).toBe(titleMatch || descriptionMatch)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  it('filters books by all selected tags', () => {
    fc.assert(
      fc.property(bookshelfDataArbitrary, ([tags, books]) => {
        const store = useBookshelfStore()
        store.setTags(tags)
        store.setBooks(books)

        const tagNames = tags.map(tag => tag.name)
        const selectedTags = tagNames.slice(0, Math.min(tagNames.length, 3))
        store.setTagFilter(selectedTags)

        const filtered = store.filteredBooks
        if (selectedTags.length === 0) {
          expect(filtered).toHaveLength(books.length)
          return
        }

        for (const book of filtered) {
          for (const tagName of selectedTags) {
            expect(book.tags?.includes(tagName)).toBe(true)
          }
        }

        for (const book of books) {
          const hasAllTags = selectedTags.every(tagName => book.tags?.includes(tagName))
          const isIncluded = filtered.some(filteredBook => filteredBook.id === book.id)
          expect(isIncluded).toBe(hasAllTags)
        }
      }),
      { numRuns: 100 }
    )
  })

  it('keeps batch selection as a toggle set', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          uniqueBooksArbitrary(selectableBookArbitrary, { minLength: 1, maxLength: 10 }),
          fc.array(fc.integer({ min: 0, max: 9 }), { minLength: 1, maxLength: 20 })
        ),
        ([books, toggleIndices]) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          store.enterBatchMode()
          expect(store.batchMode).toBe(true)

          const expectedSelected = new Set<string>()
          for (const index of toggleIndices) {
            if (index >= books.length) {
              continue
            }

            const bookId = books[index]?.id
            if (!bookId) {
              continue
            }

            store.toggleBookSelection(bookId)
            if (expectedSelected.has(bookId)) {
              expectedSelected.delete(bookId)
            } else {
              expectedSelected.add(bookId)
            }
          }

          expect(store.selectedBookIds.size).toBe(expectedSelected.size)
          for (const bookId of expectedSelected) {
            expect(store.selectedBookIds.has(bookId)).toBe(true)
          }

          store.exitBatchMode()
          expect(store.batchMode).toBe(false)
          expect(store.selectedBookIds.size).toBe(0)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('selects and clears all visible books', () => {
    fc.assert(
      fc.property(uniqueBooksArbitrary(selectableBookArbitrary, { minLength: 1, maxLength: 10 }), (books) => {
        const store = useBookshelfStore()
        store.setBooks(books)
        store.enterBatchMode()

        store.toggleSelectAll()
        expect(store.isAllSelected).toBe(true)
        expect(store.selectedBookIds.size).toBe(books.length)

        store.toggleSelectAll()
        expect(store.isAllSelected).toBe(false)
        expect(store.selectedBookIds.size).toBe(0)
      }),
      { numRuns: 100 }
    )
  })

  it('toggles tag filter selections', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          uniqueTagsArbitrary({ minLength: 1, maxLength: 5 }),
          fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 10 })
        ),
        ([tags, toggleIndices]) => {
          const store = useBookshelfStore()
          store.setTags(tags)
          store.clearTagFilter()

          const expectedSelected = new Set<string>()
          for (const index of toggleIndices) {
            if (index >= tags.length) {
              continue
            }

            const tagName = tags[index]?.name
            if (!tagName) {
              continue
            }

            store.toggleTagFilter(tagName)
            if (expectedSelected.has(tagName)) {
              expectedSelected.delete(tagName)
            } else {
              expectedSelected.add(tagName)
            }
          }

          const expectedArray = Array.from(expectedSelected)
          expect(store.selectedTagNames).toHaveLength(expectedArray.length)
          for (const tagName of expectedArray) {
            expect(store.selectedTagNames).toContain(tagName)
          }
        }
      ),
      { numRuns: 100 }
    )
  })

  it('removes a deleted book from batch selection', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          uniqueBooksArbitrary(selectableBookArbitrary, { minLength: 2, maxLength: 10 }),
          fc.integer({ min: 0, max: 9 })
        ),
        ([books, deleteIndex]) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          store.enterBatchMode()

          const bookToDelete = books[Math.min(deleteIndex, books.length - 1)]
          if (!bookToDelete) {
            return
          }

          store.toggleBookSelection(bookToDelete.id)
          expect(store.selectedBookIds.has(bookToDelete.id)).toBe(true)

          store.deleteBook(bookToDelete.id)

          expect(store.books.find(book => book.id === bookToDelete.id)).toBeUndefined()
          expect(store.selectedBookIds.has(bookToDelete.id)).toBe(false)
        }
      ),
      { numRuns: 100 }
    )
  })
})
