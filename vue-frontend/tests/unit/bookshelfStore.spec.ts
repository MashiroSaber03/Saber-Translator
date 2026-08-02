import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as bookshelfApi from '@/api/bookshelf'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'
import { setTestBooks, setTestTags } from '../helpers/bookshelfFixtures'

describe('bookshelfStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
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

  it('keeps the selected detail projection when the list refreshes', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{
      id: 'book-detail',
      title: 'Detailed Book',
      tags: [],
      chapters: [{
        id: 'chapter-1',
        title: 'Chapter One',
        order: 0,
        imageCount: 1,
      }],
      translationConstraints: {
        glossary: {
          enabled: true,
          autoExtractEnabled: false,
          autoExtractPrompt: '',
          entries: [],
        },
        non_translate: {
          enabled: false,
          entries: [],
        },
      },
    }])
    store.setCurrentBook('book-detail')
    vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([{
      id: 'book-detail',
      title: 'Detailed Book',
      tags: ['updated'],
      chapterCount: 1,
      chapters: undefined,
      translationConstraints: undefined,
    }])

    await store.loadBooks()

    expect(store.books[0]?.chapters).toBeUndefined()
    expect(store.currentBook?.tags).toEqual(['updated'])
    expect(store.currentBook?.chapters?.[0]?.id).toBe('chapter-1')
    expect(store.currentBook?.translationConstraints?.glossary.enabled).toBe(true)
  })

  it('refreshes the backend list projection after creating a book', async () => {
    const store = useBookshelfStore()
    store.sortBy = 'title'
    store.sortOrder = 'asc'
    const created: BookData = { id: 'book-a', title: 'Alpha' }
    const sorted: BookData[] = [
      created,
      { id: 'book-b', title: 'Beta' },
    ]
    vi.spyOn(bookshelfApi, 'createBook').mockResolvedValue(created)
    const getBooks = vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue(sorted)

    await expect(store.createBook('Alpha')).resolves.toEqual(created)

    expect(getBooks).toHaveBeenCalledWith({
      sortBy: 'title',
      sortOrder: 'asc',
    })
    expect(store.books.map(book => book.id)).toEqual(['book-a', 'book-b'])
  })

  it('reloads books after deleting the active tag filter', async () => {
    const store = useBookshelfStore()
    store.selectedTagNames = ['obsolete']
    setTestTags(store, [{ name: 'obsolete', color: '#ffffff' }])
    vi.spyOn(bookshelfApi, 'deleteTag').mockResolvedValue(undefined)
    const getBooks = vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([
      { id: 'book-all', title: 'All Books' },
    ])

    await expect(store.deleteTagApi('obsolete')).resolves.toBe(true)

    expect(getBooks).toHaveBeenCalledWith({
      sortBy: 'updatedAt',
      sortOrder: 'desc',
    })
    expect(store.selectedTagNames).toEqual([])
    expect(store.books.map(book => book.id)).toEqual(['book-all'])
  })
})
