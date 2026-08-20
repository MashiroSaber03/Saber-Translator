import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as bookshelfApi from '@/api/bookshelf'
import { ApiClientError } from '@/api/client'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, TagData } from '@/types/api'
import { setTestBooks, setTestTags } from '../helpers/bookshelfFixtures'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function apiError(status: number, message = 'request failed') {
  return new ApiClientError({ code: 'test_error', message, status })
}

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
    setTestTags(store, [{ id: 'tag-a', name: 'tag-a', color: '#ffffff' }])
    store.selectedTagNames = ['tag-a']
    store.setSort('title', 'asc')
    store.setCurrentBook(book.id)
    store.enterBatchMode()
    store.toggleBookSelection(book.id)
    store.isLoading = true
    store.error = 'boom'
    store.tagsError = 'tag boom'

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
    expect(store.tagsError).toBeNull()
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
        nonTranslate: {
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

  it('only commits the newest bookshelf request', async () => {
    const store = useBookshelfStore()
    const first = deferred<BookData[]>()
    const second = deferred<BookData[]>()
    const getBooks = vi.spyOn(bookshelfApi, 'getBooks')
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)

    const firstRequest = store.loadBooks()
    store.setSearchQuery('latest')

    expect(getBooks).toHaveBeenNthCalledWith(2, {
      search: 'latest',
      sortBy: 'updatedAt',
      sortOrder: 'desc',
    })

    second.resolve([{ id: 'latest', title: 'Latest' }])
    await vi.waitFor(() => expect(store.books[0]?.id).toBe('latest'))
    expect(store.isLoading).toBe(false)

    first.resolve([{ id: 'stale', title: 'Stale' }])
    await firstRequest

    expect(store.books[0]?.id).toBe('latest')
    expect(store.error).toBeNull()
  })

  it('does not let a stale request clear the active loading state', async () => {
    const store = useBookshelfStore()
    const first = deferred<BookData[]>()
    const second = deferred<BookData[]>()
    vi.spyOn(bookshelfApi, 'getBooks')
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)

    const firstRequest = store.loadBooks()
    store.setSearchQuery('latest')
    first.resolve([])
    await firstRequest

    expect(store.isLoading).toBe(true)

    second.resolve([])
    await vi.waitFor(() => expect(store.isLoading).toBe(false))
  })

  it('does not restore a book from a list request started before deletion', async () => {
    const store = useBookshelfStore()
    const book: BookData = { id: 'book-delete', title: 'Delete Me' }
    setTestBooks(store, [book])
    store.setCurrentBook(book.id)
    const staleList = deferred<BookData[]>()
    vi.spyOn(bookshelfApi, 'getBooks').mockReturnValue(staleList.promise)
    vi.spyOn(bookshelfApi, 'deleteBook').mockResolvedValue(undefined)

    const loading = store.loadBooks()
    await store.deleteBookApi(book.id)
    staleList.resolve([book])
    await loading

    expect(store.books).toEqual([])
    expect(store.currentBookId).toBeNull()
    expect(store.currentBook).toBeNull()
    expect(store.isLoading).toBe(false)
  })

  it('accepts an ambiguous delete response when the backend confirms the book is gone', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{ id: 'book-deleted', title: 'Deleted' }])
    vi.spyOn(bookshelfApi, 'deleteBook').mockRejectedValue(apiError(0, 'connection reset'))
    vi.spyOn(bookshelfApi, 'getBookDetail').mockRejectedValue(apiError(404, 'not found'))

    await expect(store.deleteBookApi('book-deleted')).resolves.toBeUndefined()

    expect(store.books).toEqual([])
  })

  it('does not hide a book when the backend confirms deletion was rejected', async () => {
    const store = useBookshelfStore()
    const book: BookData = { id: 'book-locked', title: 'Locked' }
    setTestBooks(store, [book])
    vi.spyOn(bookshelfApi, 'deleteBook').mockRejectedValue(apiError(423, 'locked'))
    const getBookDetail = vi.spyOn(bookshelfApi, 'getBookDetail')

    await expect(store.deleteBookApi(book.id)).rejects.toThrow('locked')

    expect(getBookDetail).not.toHaveBeenCalled()
    expect(store.books).toEqual([book])
  })

  it('reconciles an ambiguous chapter delete without a hard refresh', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{
      id: 'book-1',
      title: 'Book',
      chapters: [{ id: 'chapter-1', title: 'Chapter', order: 0, imageCount: 1 }],
      chapterCount: 1,
    }])
    vi.spyOn(bookshelfApi, 'deleteChapter').mockRejectedValue(apiError(0, 'connection reset'))
    vi.spyOn(bookshelfApi, 'getBookDetail').mockResolvedValue({
      id: 'book-1',
      title: 'Book',
      chapters: [],
      chapterCount: 0,
    })

    await expect(store.deleteChapterApi('book-1', 'chapter-1')).resolves.toBeUndefined()

    expect(store.books[0]?.chapters).toEqual([])
    expect(store.books[0]?.chapterCount).toBe(0)
  })

  it('reports tag load failures without letting an older request overwrite newer tags', async () => {
    const store = useBookshelfStore()
    const first = deferred<TagData[]>()
    const second = deferred<TagData[]>()
    vi.spyOn(bookshelfApi, 'getTags')
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)

    const firstRequest = store.loadTags()
    const secondRequest = store.loadTags()
    second.resolve([{ id: 'latest', name: 'Latest', color: '#ffffff' }])
    await secondRequest
    first.reject(new Error('stale failure'))
    await firstRequest

    expect(store.tags.map(tag => tag.id)).toEqual(['latest'])
    expect(store.tagsError).toBeNull()

    vi.spyOn(bookshelfApi, 'getTags').mockRejectedValueOnce(new Error('tag service unavailable'))
    await store.loadTags()
    expect(store.tagsError).toBe('tag service unavailable')
  })

  it('drops hidden batch selections when the backend list projection changes', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [
      { id: 'visible', title: 'Visible' },
      { id: 'hidden', title: 'Hidden' },
    ])
    store.enterBatchMode()
    store.toggleBookSelection('visible')
    store.toggleBookSelection('hidden')
    vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([
      { id: 'visible', title: 'Visible' },
    ])

    await store.loadBooks()

    expect([...store.selectedBookIds]).toEqual(['visible'])
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
    setTestTags(store, [{ id: 'obsolete', name: 'obsolete', color: '#ffffff' }])
    vi.spyOn(bookshelfApi, 'deleteTag').mockResolvedValue(undefined)
    const getBooks = vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([
      { id: 'book-all', title: 'All Books' },
    ])

    await expect(store.deleteTagApi('obsolete')).resolves.toBeUndefined()

    expect(getBooks).toHaveBeenCalledWith({
      sortBy: 'updatedAt',
      sortOrder: 'desc',
    })
    expect(store.selectedTagNames).toEqual([])
    expect(store.books.map(book => book.id)).toEqual(['book-all'])
  })

  it('removes a deleted tag from local book projections even when the list refresh fails', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{ id: 'book-1', title: 'Book', tags: ['obsolete', 'keep'] }])
    store.setCurrentBook('book-1')
    setTestTags(store, [{ id: 'obsolete', name: 'obsolete', color: '#ffffff' }])
    vi.spyOn(bookshelfApi, 'deleteTag').mockResolvedValue(undefined)
    vi.spyOn(bookshelfApi, 'getBooks').mockRejectedValue(new Error('refresh failed'))

    await store.deleteTagApi('obsolete')

    expect(store.tags).toEqual([])
    expect(store.currentBook?.tags).toEqual(['keep'])
    expect(store.error).toBe('refresh failed')
  })

  it('keeps an active tag filter valid when the tag is renamed', async () => {
    const store = useBookshelfStore()
    store.selectedTagNames = ['old-name']
    setTestTags(store, [{ id: 'tag-1', name: 'old-name', color: '#ffffff' }])
    vi.spyOn(bookshelfApi, 'updateTag').mockResolvedValue({
      id: 'tag-1',
      name: 'new-name',
      color: '#ffffff',
    })
    vi.spyOn(bookshelfApi, 'getTags').mockResolvedValue([
      { id: 'tag-1', name: 'new-name', color: '#ffffff' },
    ])
    const getBooks = vi.spyOn(bookshelfApi, 'getBooks').mockResolvedValue([])

    await expect(
      store.updateTagApi('old-name', 'new-name', '#ffffff'),
    ).resolves.toMatchObject({ id: 'tag-1', name: 'new-name' })

    expect(store.selectedTagNames).toEqual(['new-name'])
    expect(getBooks).toHaveBeenCalledWith({
      tags: ['new-name'],
      sortBy: 'updatedAt',
      sortOrder: 'desc',
    })
  })

  it('renames tags in local book projections even when refreshes fail', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{ id: 'book-1', title: 'Book', tags: ['old-name'] }])
    store.setCurrentBook('book-1')
    store.selectedTagNames = ['old-name']
    setTestTags(store, [{ id: 'tag-1', name: 'old-name', color: '#000000', bookCount: 1 }])
    vi.spyOn(bookshelfApi, 'updateTag').mockResolvedValue({
      id: 'tag-1',
      name: 'new-name',
      color: '#ffffff',
    })
    vi.spyOn(bookshelfApi, 'getTags').mockRejectedValue(new Error('tag refresh failed'))
    vi.spyOn(bookshelfApi, 'getBooks').mockRejectedValue(new Error('book refresh failed'))

    await store.updateTagApi('old-name', 'new-name', '#ffffff')

    expect(store.tags[0]).toEqual({
      id: 'tag-1',
      name: 'new-name',
      color: '#ffffff',
      bookCount: 1,
    })
    expect(store.currentBook?.tags).toEqual(['new-name'])
    expect(store.selectedTagNames).toEqual(['new-name'])
    expect(store.tagsError).toBe('tag refresh failed')
    expect(store.error).toBe('book refresh failed')
  })

  it('preserves loaded chapters when a book update returns only the summary projection', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{
      id: 'book-1',
      title: 'Old title',
      chapters: [{ id: 'chapter-1', title: 'Chapter', order: 0, imageCount: 3 }],
    }])
    store.setCurrentBook('book-1')
    vi.spyOn(bookshelfApi, 'updateBook').mockResolvedValue({
      id: 'book-1',
      title: 'New title',
      chapterCount: 1,
    })
    vi.spyOn(bookshelfApi, 'getBooks').mockRejectedValue(new Error('refresh failed'))

    await store.updateBookApi('book-1', { title: 'New title' })

    expect(store.currentBook?.title).toBe('New title')
    expect(store.currentBook?.chapters).toEqual([
      { id: 'chapter-1', title: 'Chapter', order: 0, imageCount: 3 },
    ])
  })

  it('updates a chapter title without replacing its backend-owned projection fields', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{
      id: 'book-1',
      title: 'Book',
      chapters: [{
        id: 'chapter-1',
        title: 'Old title',
        order: 3,
        imageCount: 27,
      }],
    }])
    vi.spyOn(bookshelfApi, 'updateChapter').mockResolvedValue({
      id: 'chapter-1',
      title: 'New title',
    })

    await store.updateChapterApi('book-1', 'chapter-1', 'New title')

    expect(store.books[0]?.chapters?.[0]).toEqual({
      id: 'chapter-1',
      title: 'New title',
      order: 3,
      imageCount: 27,
    })
  })
})
