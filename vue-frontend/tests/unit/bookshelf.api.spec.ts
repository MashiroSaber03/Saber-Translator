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
    expect(result.translationConstraints).toEqual({
      glossary: constraints.payload.glossary,
      nonTranslate: constraints.payload.nonTranslate,
    })
  })

  it('returns the create response without a redundant detail read', async () => {
    postMock.mockResolvedValue(book)
    const { createBook } = await import('@/api/bookshelf')

    const created = await createBook('Book')

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/books',
      { title: 'Book', tagIds: [] },
      commandConfig,
    )
    expect(getMock).not.toHaveBeenCalled()
    expect(created).toMatchObject({ id: book.id, title: 'Book' })
  })

  it('saves the structured constraint document through its dedicated CAS resource', async () => {
    putMock.mockResolvedValue({
      ...constraints,
      revision: 3,
    })
    const { updateBookTranslationConstraints } = await import('@/api/bookshelf')
    const result = await updateBookTranslationConstraints(book.id, {
      glossary: constraints.payload.glossary,
      nonTranslate: constraints.payload.nonTranslate,
    }, 2)

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
    expect(getMock).not.toHaveBeenCalled()
    expect(result).toEqual({
      constraints: constraints.payload,
      revision: 3,
    })
  })

  it('updates a book cover through the multipart PUT contract', async () => {
    uploadMock.mockResolvedValue(book)
    const cover = new File(['cover'], 'cover.png', { type: 'image/png' })
    const { updateBook } = await import('@/api/bookshelf')

    await updateBook(book.id, { title: 'Updated Book', cover })

    expect(uploadMock).toHaveBeenCalledTimes(1)
    const [url, body, config, method] = uploadMock.mock.calls[0]
    expect(url).toBe('/api/v2/books/book%2Fid%20one')
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('title')).toBe('Updated Book')
    expect((body as FormData).has('tagIds')).toBe(false)
    const uploadedCover = (body as FormData).get('cover')
    expect(uploadedCover).toBeInstanceOf(File)
    expect((uploadedCover as File).name).toBe('cover.png')
    expect((uploadedCover as File).type).toBe('image/png')
    expect((uploadedCover as File).size).toBe(cover.size)
    expect(config).toEqual(commandConfig)
    expect(method).toBe('put')
    expect(putMock).not.toHaveBeenCalled()
    expect(getMock).not.toHaveBeenCalled()
  })

  it('updates only the requested book fields without reading or overwriting the rest', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/tags') {
        return Promise.resolve({
          items: [{ id: 'tag-id', name: 'Fantasy', color: '#4466aa' }],
        })
      }
      return Promise.reject(new Error(`unexpected GET ${url}`))
    })
    putMock.mockResolvedValue({
      id: book.id,
      title: book.title,
      chapterOrderRevision: book.chapterOrderRevision,
      tags: [{ id: 'tag-id', name: 'Fantasy', color: '#4466aa' }],
    })
    const { updateBook } = await import('@/api/bookshelf')

    const updated = await updateBook(book.id, { tags: ['Fantasy'] })

    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/books/book%2Fid%20one',
      { tagIds: ['tag-id'] },
      commandConfig,
    )
    expect(getMock).toHaveBeenCalledTimes(1)
    expect(getMock).toHaveBeenCalledWith('/api/v2/tags')
    expect(updated.tags).toEqual(['Fantasy'])
    expect(updated).not.toHaveProperty('chapters')
  })

  it('uses direct chapter resources and idempotency headers', async () => {
    putMock.mockResolvedValue({
      id: 'chapter/id one',
      title: 'Updated Chapter',
    })
    const { deleteChapter, updateChapter } = await import('@/api/bookshelf')

    await updateChapter('chapter/id one', 'Updated Chapter')
    await deleteChapter('chapter/id one')

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
