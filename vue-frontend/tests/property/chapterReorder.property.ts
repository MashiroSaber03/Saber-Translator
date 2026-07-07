import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, ChapterData } from '@/types/api'

describe('bookshelf chapter reorder properties', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  const tagNameArbitrary = fc.stringOf(
    fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789'.split('')),
    { minLength: 1, maxLength: 20 }
  ).map((value) => `tag-${value}`)

  const chapterArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 50 }),
    order: fc.nat({ max: 100 }),
    imageCount: fc.nat({ max: 100 }),
    hasSession: fc.boolean(),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<ChapterData>

  const chapterListArbitrary = fc.uniqueArray(chapterArbitrary, {
    minLength: 1,
    maxLength: 20,
    selector: chapter => chapter.id,
  })

  const bookWithChaptersArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(tagNameArbitrary, { maxLength: 5 }),
    chapters: chapterListArbitrary,
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  it('reorders chapters to match the requested id order', () => {
    fc.assert(
      fc.property(bookWithChaptersArbitrary, fc.nat({ max: 100 }), (book, seed) => {
        const store = useBookshelfStore()
        store.setBooks([book])

        const originalIds = book.chapters!.map(chapter => chapter.id)
        if (originalIds.length < 2) {
          return
        }

        const shuffledIds = [...originalIds]
        for (let index = shuffledIds.length - 1; index > 0; index -= 1) {
          const swapIndex = (seed + index) % (index + 1)
          ;[shuffledIds[index], shuffledIds[swapIndex]] = [shuffledIds[swapIndex], shuffledIds[index]]
        }

        store.reorderChapters(book.id, shuffledIds)

        const reorderedIds = store.getBookById(book.id)?.chapters?.map(chapter => chapter.id) ?? []
        expect(reorderedIds).toEqual(shuffledIds)
      }),
      { numRuns: 50 }
    )
  })

  it('normalizes chapter order fields after reorder', () => {
    fc.assert(
      fc.property(bookWithChaptersArbitrary, (book) => {
        const store = useBookshelfStore()
        store.setBooks([book])

        const originalIds = book.chapters!.map(chapter => chapter.id)
        if (originalIds.length < 2) {
          return
        }

        store.reorderChapters(book.id, [...originalIds].reverse())

        const orders = store.getBookById(book.id)?.chapters?.map(chapter => chapter.order) ?? []
        for (let index = 0; index < orders.length; index += 1) {
          expect(orders[index]).toBe(index)
        }
      }),
      { numRuns: 50 }
    )
  })

  it('keeps the same chapter count after reorder', () => {
    fc.assert(
      fc.property(bookWithChaptersArbitrary, (book) => {
        const store = useBookshelfStore()
        store.setBooks([book])

        const originalIds = book.chapters!.map(chapter => chapter.id)
        store.reorderChapters(book.id, [...originalIds].reverse())

        expect(store.getBookById(book.id)?.chapters?.length).toBe(book.chapters!.length)
      }),
      { numRuns: 50 }
    )
  })

  it('keeps chapter content attached to every chapter id', () => {
    fc.assert(
      fc.property(bookWithChaptersArbitrary, (book) => {
        const store = useBookshelfStore()
        store.setBooks([book])

        const originalTitles = new Set(book.chapters!.map(chapter => chapter.title))
        const originalIds = book.chapters!.map(chapter => chapter.id)
        store.reorderChapters(book.id, [...originalIds].reverse())

        const reorderedTitles = new Set(
          store.getBookById(book.id)?.chapters?.map(chapter => chapter.title) ?? []
        )
        expect(reorderedTitles).toEqual(originalTitles)
      }),
      { numRuns: 50 }
    )
  })
})
