import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
  },
}))

function streamFromChunks(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}

describe('web import api streams', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('routes non-stream JSON endpoints through apiClient', async () => {
    getMock.mockResolvedValue({ success: true })
    postMock.mockResolvedValue({ success: true })

    const {
      checkGalleryDLSupport,
      downloadImages,
      getGalleryDLImages,
      testAgentConnection,
      testFirecrawlConnection,
    } = await import('@/api/webImport')

    await checkGalleryDLSupport('https://example.test/book?a=1')
    await getGalleryDLImages()
    await downloadImages(
      [{ pageNumber: 1, imageUrl: 'https://example.test/001.jpg' }],
      'https://example.test/book',
      { agent: { provider: 'deepseek' } } as Parameters<typeof downloadImages>[2],
      'ai-agent',
    )
    await testFirecrawlConnection('firecrawl-key')
    await testAgentConnection('deepseek', 'agent-key', '', 'deepseek-chat')

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/web-import/check-support', {
      params: { url: 'https://example.test/book?a=1' },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/web-import/gallery-dl-images')
    expect(postMock).toHaveBeenNthCalledWith(1, '/api/web-import/download', {
      pages: [{ pageNumber: 1, imageUrl: 'https://example.test/001.jpg' }],
      sourceUrl: 'https://example.test/book',
      config: { agent: { provider: 'deepseek' } },
      engine: 'ai-agent',
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/web-import/test-firecrawl', {
      apiKey: 'firecrawl-key',
    })
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/web-import/test-agent', {
      provider: 'deepseek',
      apiKey: 'agent-key',
      customBaseUrl: '',
      modelName: 'deepseek-chat',
    })
  })

  it('extracts log, page, and result events through the shared SSE reader', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamFromChunks([
      'event: log\ndata: {"level":"info","message":"started"}\n\n',
      'event: page\ndata: {"index":0,"filename":"001.jpg","dataUrl":"data:image/jpeg;base64,abc","size":3}\n\n',
      'event: result\ndata: {"success":true,"images":[],"logs":[]}\n\n',
    ]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const onLog = vi.fn()
    const onPage = vi.fn()
    const onResult = vi.fn()
    const onError = vi.fn()

    const { extractImages } = await import('@/api/webImport')
    await extractImages(
      'https://example.test/book',
      {} as Parameters<typeof extractImages>[1],
      onLog,
      onResult,
      onError,
      'auto',
      onPage,
    )

    expect(onLog).toHaveBeenCalledWith({ level: 'info', message: 'started' })
    expect(onPage).toHaveBeenCalledWith({
      index: 0,
      filename: '001.jpg',
      dataUrl: 'data:image/jpeg;base64,abc',
      size: 3,
    })
    expect(onResult).toHaveBeenCalledWith({ success: true, images: [], logs: [] })
    expect(onError).not.toHaveBeenCalled()
  })

  it('lets apiClient normalize failed JSON endpoint errors', async () => {
    const { downloadImages } = await import('@/api/webImport')
    postMock.mockRejectedValueOnce(new Error('download failed'))

    await expect(downloadImages([], 'https://example.test/book', {} as Parameters<typeof downloadImages>[2]))
      .rejects.toThrow('download failed')
  })
})
