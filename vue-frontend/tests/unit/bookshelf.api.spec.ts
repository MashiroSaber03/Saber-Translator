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
  schemaVersion: 2,
  payload: {
    glossary: {
      enabled: true,
      autoExtractEnabled: false,
      autoExtractPrompt: '提取 {ocr_text}',
      entries: [
        {
          source: 'Saber',
          target: '阿尔托莉雅',
          note: '',
          matchMode: 'text',
        },
      ],
    },
    nonTranslate: {
      enabled: true,
      entries: [
        {
          pattern: 'Excalibur',
          note: '',
          matchMode: 'text',
        },
      ],
    },
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
    expect(result.book?.translationConstraints).toEqual({
      glossary: constraints.payload.glossary,
      non_translate: constraints.payload.nonTranslate,
    })
  })

  it('saves the structured constraint document through its dedicated CAS resource', async () => {
    putMock.mockResolvedValue({
      ...constraints,
      revision: 3,
    })
    const { getBookDetail, updateBook } = await import('@/api/bookshelf')
    await getBookDetail(book.id)
    await updateBook(book.id, {
      translationConstraints: {
        glossary: constraints.payload.glossary,
        non_translate: constraints.payload.nonTranslate,
      },
    })

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/books/book%2Fid%20one/translation-constraints',
      {
        baseRevision: 2,
        payload: {
          glossary: constraints.payload.glossary,
          nonTranslate: constraints.payload.nonTranslate,
        },
      },
      commandConfig,
    )
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
