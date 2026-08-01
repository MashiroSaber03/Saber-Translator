import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import * as bookshelfApi from '@/api/bookshelf'
import type { BookData, TagData } from '@/types/api'
import { setTestBooks, setTestTags } from '../helpers/bookshelfFixtures'

const bookIdArbitrary = fc.uuid()
const bookTitleArbitrary = fc.string({ minLength: 1, maxLength: 100 })
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

describe('bookshelf store properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([])
  })

  afterEach(() => {
    vi.restoreAllMocks()
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
          setTestBooks(store, books)
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
        setTestBooks(store, books)
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
          setTestTags(store, tags)
          store.selectedTagNames = []

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

})
