import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'
import { setTestBooks, setTestTags } from '../helpers/bookshelfFixtures'

describe('bookshelfStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('resets current book selection with the rest of the bookshelf state', () => {
    const store = useBookshelfStore()
    const book: BookData = {
      id: 'book-1',
      title: 'Book One',
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-02T00:00:00.000Z',
    }

    setTestBooks(store, [book])
    setTestTags(store, [{ name: 'tag-a', color: '#ffffff' }])
    store.selectedTagNames = ['tag-a']
    store.setSort('title', 'asc')
    store.setCurrentBook(book.id)
    store.enterBatchMode()
    store.toggleBookSelection(book.id)
    store.isLoading = true
    store.error = 'boom'

    store.reset()

    expect(store.books).toEqual([])
    expect(store.tags).toEqual([])
    expect(store.selectedTagNames).toEqual([])
    expect(store.batchMode).toBe(false)
    expect(store.selectedBookIds.size).toBe(0)
    expect(store.sortBy).toBe('updatedAt')
    expect(store.sortOrder).toBe('desc')
    expect(store.currentBookId).toBeNull()
    expect(store.currentBook).toBeNull()
    expect(store.isLoading).toBe(false)
    expect(store.error).toBeNull()
  })

  it('keeps current bookshelf DTO fields intact', () => {
    const store = useBookshelfStore()

    setTestBooks(store, [{
      id: 'book-wire',
      title: 'Wire Book',
      createdAt: '2026-02-01T00:00:00.000Z',
      updatedAt: '2026-02-02T00:00:00.000Z',
      chapters: [{
        id: 'chapter-wire',
        title: 'Wire Chapter',
        order: 0,
        imageCount: 3,
      }],
    } as BookData])

    expect(store.books[0]).toMatchObject({
      id: 'book-wire',
      createdAt: '2026-02-01T00:00:00.000Z',
      updatedAt: '2026-02-02T00:00:00.000Z',
    })
    expect(store.books[0]?.chapters?.[0]).toMatchObject({
      id: 'chapter-wire',
      imageCount: 3,
    })
  })

  it('treats the backend response as the list projection', () => {
    const books: BookData[] = [
      {
        id: 'book-b',
        title: 'Beta',
        tags: ['fantasy'],
        createdAt: '2026-01-02T00:00:00.000Z',
        updatedAt: '2026-01-03T00:00:00.000Z',
      },
      {
        id: 'book-a',
        title: 'Alpha',
        tags: ['fantasy', 'favorite'],
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-04T00:00:00.000Z',
      },
      {
        id: 'book-c',
        title: 'Gamma',
        tags: ['science'],
        createdAt: '2026-01-03T00:00:00.000Z',
        updatedAt: '2026-01-02T00:00:00.000Z',
      },
    ]

    const store = useBookshelfStore()
    setTestBooks(store, books)

    expect(store.books.map(book => book.id)).toEqual([
      'book-b',
      'book-a',
      'book-c',
    ])

    const source = readFileSync(resolve(process.cwd(), 'src/stores/bookshelfStore.ts'), 'utf8')
    expect(source).not.toContain('projectBookshelfBooks')
    expect(source).not.toContain("'@/stores/bookshelfListProjection'")
    expect(source).not.toContain('return books.value.filter(book =>')
  })
})
