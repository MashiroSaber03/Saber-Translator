import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, putMock, deleteMock, uploadMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  deleteMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    put: putMock,
    delete: deleteMock,
    upload: uploadMock,
  },
}))

vi.mock('@/api/v2/content', () => ({
  newIdempotencyKey: () => 'bookshelf-idempotency-key',
}))

const book = {
  id: 'book/id one',
  title: 'Book',
  chapterOrderRevision: 3,
  tags: [],
  chapters: [],
}

const constraints = {
  bookId: book.id,
  revision: 2,
  payload: {
    glossary: {},
    nonTranslate: {},
  },
}

const commandConfig = {
  headers: {
    'Idempotency-Key': 'bookshelf-idempotency-key',
  },
}

describe('bookshelf v2 api contracts', () => {
  beforeEach(() => {
    vi.resetModules()
    getMock.mockReset().mockImplementation((url: string) => {
      if (url.endsWith('/translation-constraints')) {
        return Promise.resolve(constraints)
      }
      if (url === '/api/v2/tags') return Promise.resolve({ items: [] })
      return Promise.resolve(book)
    })
    postMock.mockReset()
    putMock.mockReset()
    deleteMock.mockReset().mockResolvedValue({ deleted: true })
    uploadMock.mockReset()
  })

  it('loads book details and constraints from encoded v2 resources', async () => {
    const { getBookDetail } = await import('@/api/bookshelf')

    const result = await getBookDetail(book.id)

    expect(getMock).toHaveBeenCalledWith('/api/v2/books/book%2Fid%20one')
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/books/book%2Fid%20one/translation-constraints',
    )
    expect(result.book?.translation_constraints).toEqual({
      glossary: {},
      non_translate: {},
    })
  })

  it('uses direct chapter resources and idempotency headers', async () => {
    putMock.mockResolvedValue({
      id: 'chapter/id one',
      title: 'Updated Chapter',
    })
    const { deleteChapter, updateChapter } = await import('@/api/bookshelf')

    await updateChapter(book.id, 'chapter/id one', 'Updated Chapter')
    await deleteChapter(book.id, 'chapter/id one')

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/chapters/chapter%2Fid%20one',
      { title: 'Updated Chapter' },
      commandConfig,
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/chapters/chapter%2Fid%20one',
      commandConfig,
    )
  })

  it('reorders chapters with the authoritative book revision', async () => {
    putMock.mockResolvedValue({ chapterOrderRevision: 4 })
    const { getBookDetail, reorderChapters } = await import('@/api/bookshelf')
    await getBookDetail(book.id)

    await reorderChapters(book.id, ['chapter/id one', 'chapter two'])

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/books/book%2Fid%20one/chapters/order',
      {
        baseRevision: 3,
        orderedIds: ['chapter/id one', 'chapter two'],
      },
      commandConfig,
    )
  })
})
