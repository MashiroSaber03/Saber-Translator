import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'

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

    store.setBooks([book])
    store.setTags([{ name: 'tag-a', color: '#ffffff' }])
    store.setSearchKeyword('book')
    store.setTagFilter(['tag-a'])
    store.setSort('title', 'asc')
    store.expandBook(book.id)
    store.setCurrentBook(book.id)
    store.enterBatchMode()
    store.toggleBookSelection(book.id)
    store.setLoading(true)
    store.setError('boom')

    store.reset()

    expect(store.books).toEqual([])
    expect(store.tags).toEqual([])
    expect(store.searchKeyword).toBe('')
    expect(store.selectedTagNames).toEqual([])
    expect(store.batchMode).toBe(false)
    expect(store.selectedBookIds.size).toBe(0)
    expect(store.sortBy).toBe('updatedAt')
    expect(store.sortOrder).toBe('desc')
    expect(store.expandedBookId).toBeNull()
    expect(store.currentBookId).toBeNull()
    expect(store.currentBook).toBeNull()
    expect(store.isLoading).toBe(false)
    expect(store.error).toBeNull()
  })

  it('normalizes bookshelf wire fields before exposing books to UI owners', () => {
    const store = useBookshelfStore()

    store.setBooks([{
      id: 'book-wire',
      title: 'Wire Book',
      created_at: '2026-02-01T00:00:00.000Z',
      updated_at: '2026-02-02T00:00:00.000Z',
      chapters: [{
        id: 'chapter-wire',
        title: 'Wire Chapter',
        order: 0,
        image_count: 3,
        has_session: true,
        session_path: 'sessions/chapter-wire',
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
      hasSession: true,
      session_path: 'sessions/chapter-wire',
    })
  })

  it('normalizes malformed bookshelf text fields before list sorting reaches the UI', () => {
    const store = useBookshelfStore()

    store.setBooks([{
      id: 'book-malformed-title',
      title: { text: 'Broken Title Shape' },
      description: ['unexpected', 'description'],
      created_at: 1700000000000,
      updated_at: { value: '2026-02-02T00:00:00.000Z' },
    } as unknown as BookData, {
      id: 'book-valid-title',
      title: 'Valid Title',
      createdAt: '2026-02-01T00:00:00.000Z',
      updatedAt: '2026-02-01T00:00:00.000Z',
    }])

    expect(() => store.filteredBooks).not.toThrow()
    expect(store.books[0]).toMatchObject({
      id: 'book-malformed-title',
      title: '',
      description: '',
      createdAt: '1700000000000',
      updatedAt: '',
    })
  })

  it('preserves omitted chapters when applying a partial local reorder', () => {
    const store = useBookshelfStore()

    store.setBooks([{
      id: 'book-partial-reorder',
      title: 'Partial Reorder',
      createdAt: '2026-02-01T00:00:00.000Z',
      updatedAt: '2026-02-02T00:00:00.000Z',
      chapters: [
        { id: 'chapter-a', title: 'A', order: 0, imageCount: 1, hasSession: false },
        { id: 'chapter-b', title: 'B', order: 1, imageCount: 1, hasSession: false },
        { id: 'chapter-c', title: 'C', order: 2, imageCount: 1, hasSession: false },
      ],
    }])

    store.reorderChapters('book-partial-reorder', ['chapter-c', 'chapter-a'])

    expect(store.getBookById('book-partial-reorder')?.chapters?.map(chapter => chapter.id)).toEqual([
      'chapter-c',
      'chapter-a',
      'chapter-b',
    ])
    expect(store.getBookById('book-partial-reorder')?.chapters?.map(chapter => chapter.order)).toEqual([0, 1, 2])
  })

  it('projects filtered and sorted books through a focused list helper', async () => {
    const { projectBookshelfBooks } = await import('@/stores/bookshelfListProjection')
    const books: BookData[] = [
      {
        id: 'book-b',
        title: 'Beta',
        description: 'fantasy archive',
        tags: ['fantasy'],
        createdAt: '2026-01-02T00:00:00.000Z',
        updatedAt: '2026-01-03T00:00:00.000Z',
      },
      {
        id: 'book-a',
        title: 'Alpha',
        description: 'fantasy primer',
        tags: ['fantasy', 'favorite'],
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-04T00:00:00.000Z',
      },
      {
        id: 'book-c',
        title: 'Gamma',
        description: 'science notes',
        tags: ['science'],
        createdAt: '2026-01-03T00:00:00.000Z',
        updatedAt: '2026-01-02T00:00:00.000Z',
      },
    ]

    expect(projectBookshelfBooks(books, {
      searchKeyword: 'fantasy',
      selectedTagNames: ['fantasy'],
      sortBy: 'title',
      sortOrder: 'asc',
    }).map(book => book.id)).toEqual(['book-a', 'book-b'])

    const store = useBookshelfStore()
    store.setBooks(books)
    store.setSearchKeyword('fantasy')
    store.setTagFilter(['fantasy'])
    store.setSort('updatedAt', 'desc')

    expect(store.filteredBooks.map(book => book.id)).toEqual(['book-a', 'book-b'])

    const source = readFileSync(resolve(process.cwd(), 'src/stores/bookshelfStore.ts'), 'utf8')
    expect(source).toContain('projectBookshelfBooks')
    expect(source).toContain("'@/stores/bookshelfListProjection'")
    expect(source).not.toContain('return books.value.filter(book =>')
  })
})
