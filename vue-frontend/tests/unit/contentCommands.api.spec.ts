import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  ApiClientError: class ApiClientError extends Error {},
  apiClient: {
    get: mocks.get,
    post: mocks.post,
  },
}))

describe('content command API', () => {
  beforeEach(() => {
    mocks.get.mockReset()
    mocks.post.mockReset()
  })

  it('forwards every supplied page-list query value to the backend contract', async () => {
    mocks.get.mockResolvedValue({ items: [], nextCursor: null, pageOrderRevision: 1 })
    const { listChapterPages } = await import('@/api/v2/content')

    await listChapterPages('chapter/1', { cursor: 0, limit: 0 })

    expect(mocks.get).toHaveBeenCalledWith(
      '/api/v2/chapters/chapter%2F1/pages?cursor=0&limit=0',
      { signal: undefined },
    )
  })

  it('does not turn an incomplete translation identity into the quick workspace', async () => {
    mocks.get.mockResolvedValue({})
    const { getTranslationBootstrap } = await import('@/api/v2/content')

    await getTranslationBootstrap({ bookId: 'book-1' })
    await getTranslationBootstrap({ chapterId: '' })

    expect(mocks.get).toHaveBeenNthCalledWith(
      1,
      '/api/v2/translation/bootstrap?bookId=book-1',
      { signal: undefined },
    )
    expect(mocks.get).toHaveBeenNthCalledWith(
      2,
      '/api/v2/translation/bootstrap?chapterId=',
      { signal: undefined },
    )
  })

  it('returns the reset command response without issuing a second bootstrap request', async () => {
    const context = {
      bookId: 'book-1',
      chapterId: 'chapter-1',
      title: '快速翻译',
    }
    mocks.post.mockResolvedValue(context)
    const { resetQuickWorkspace } = await import('@/api/v2/content')

    await expect(resetQuickWorkspace()).resolves.toBe(context)

    expect(mocks.post).toHaveBeenCalledOnce()
    expect(mocks.post).toHaveBeenCalledWith('/api/v2/quick-workspace/reset')
  })
})
