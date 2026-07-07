import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, putMock, deleteMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  deleteMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    put: putMock,
    delete: deleteMock,
  },
}))

describe('bookshelf api path contracts', () => {
  beforeEach(() => {
    getMock.mockReset().mockResolvedValue({ success: true })
    postMock.mockReset().mockResolvedValue({ success: true })
    putMock.mockReset().mockResolvedValue({ success: true })
    deleteMock.mockReset().mockResolvedValue({ success: true })
  })

  it('routes book and chapter endpoints through encoded path helpers', async () => {
    const {
      deleteBook,
      deleteChapter,
      getBookDetail,
      getChapterImages,
      getChapters,
      reorderChapters,
      updateBook,
      updateChapter,
    } = await import('@/api/bookshelf')

    const bookId = 'book/id one'
    const chapterId = 'chapter/id one'
    const encodedBook = '/api/bookshelf/books/book%2Fid%20one'
    const encodedChapter = `${encodedBook}/chapters/chapter%2Fid%20one`

    await getBookDetail(bookId)
    await updateBook(bookId, { title: 'Updated' })
    await deleteBook(bookId)
    await getChapters(bookId)
    await updateChapter(bookId, chapterId, 'Updated Chapter')
    await deleteChapter(bookId, chapterId)
    await reorderChapters(bookId, ['chapter/id one', 'chapter two'])
    await getChapterImages(bookId, chapterId)

    expect(getMock).toHaveBeenNthCalledWith(1, encodedBook)
    expect(putMock).toHaveBeenNthCalledWith(1, encodedBook, { title: 'Updated' })
    expect(deleteMock).toHaveBeenNthCalledWith(1, encodedBook)
    expect(getMock).toHaveBeenNthCalledWith(2, `${encodedBook}/chapters`)
    expect(putMock).toHaveBeenNthCalledWith(2, encodedChapter, { title: 'Updated Chapter' })
    expect(deleteMock).toHaveBeenNthCalledWith(2, encodedChapter)
    expect(postMock).toHaveBeenNthCalledWith(1, `${encodedBook}/chapters/reorder`, {
      chapter_ids: ['chapter/id one', 'chapter two'],
    })
    expect(getMock).toHaveBeenNthCalledWith(3, `${encodedChapter}/images`)
  })
})
