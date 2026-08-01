import { beforeEach, describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'

import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'
import { setTestBooks } from '../helpers/bookshelfFixtures'

const bookArbitrary = fc.record({
  id: fc.uuid(),
  title: fc.string({ minLength: 1, maxLength: 100 }),
  cover: fc.option(fc.string(), { nil: undefined }),
  tags: fc.array(fc.string({ minLength: 1, maxLength: 40 }), { maxLength: 5 }),
  chapters: fc.constant([]),
  createdAt: fc.date().map(date => date.toISOString()),
  updatedAt: fc.date().map(date => date.toISOString()),
}) as fc.Arbitrary<BookData>

describe('bookshelf book updates', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('updates only the requested book fields', () => {
    fc.assert(fc.property(
      bookArbitrary,
      fc.string({ minLength: 1, maxLength: 100 }),
      (book, newTitle) => {
        const store = useBookshelfStore()
        setTestBooks(store, [book])

        store.updateBook(book.id, { title: newTitle })

        expect(store.getBookById(book.id)).toMatchObject({
          id: book.id,
          title: newTitle,
        })
      },
    ))
  })
})
