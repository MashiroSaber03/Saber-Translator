import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  deleteMock,
  getMock,
  patchMock,
  postMock,
} = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  patchMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  const apiClient = {
    delete: deleteMock,
    get: getMock,
    patch: patchMock,
    post: postMock,
    put: vi.fn(),
    upload: vi.fn(),
  }
  return { apiClient, default: apiClient }
})

const pages = [
  {
    activeAnalysisId: null,
    analysisState: 'not_analyzed',
    chapterId: 'chapter-1',
    displayPageNumber: 1,
    pageId: 'page-1',
    thumbnailUrl: '/api/v2/assets/thumb-1',
  },
  {
    activeAnalysisId: 'analysis-2',
    analysisState: 'ready',
    chapterId: 'chapter-1',
    displayPageNumber: 2,
    pageId: 'page-2',
    thumbnailUrl: '/api/v2/assets/thumb-2',
  },
]

const note = {
  bookId: 'book/id one',
  citations: [{
    excerpt: 'evidence',
    pageId: 'page-2',
    pageIdSnapshot: 'page-2',
    pageNumberSnapshot: 2,
    score: null,
    sourceAnalysisId: null,
  }],
  commentCount: 1,
  comments: [{ question: '', answer: '', comment: '' }],
  content: 'note body',
  createdAt: '2026-07-01T00:00:00Z',
  excerpt: 'note body',
  kind: 'text',
  noteId: 'note/id one',
  revision: 7,
  tags: [],
  title: 'Note',
  updatedAt: '2026-07-01T00:00:00Z',
}

function accepted(jobId: string) {
  return { batchId: 'batch-1', jobIds: [jobId], runId: 'run-1', status: 'queued' }
}

describe('insight v2 api facade', () => {
  beforeEach(() => {
    vi.resetModules()
    deleteMock.mockReset()
    getMock.mockReset()
    patchMock.mockReset()
    postMock.mockReset()
  })

  it('resolves display page numbers to stable IDs and controls durable jobs', async () => {
    getMock.mockResolvedValueOnce({ items: pages, nextCursor: null })
    postMock
      .mockResolvedValueOnce(accepted('job-1'))
      .mockResolvedValue({})
    const {
      cancelAnalysis,
      pauseAnalysis,
      resumeAnalysis,
      startAnalysis,
    } = await import('@/api/insight')

    const result = await startAnalysis('book/id one', {
      mode: 'pages',
      pages: [1, 2],
      force: true,
    })
    await pauseAnalysis('book/id one', 'job-1')
    await resumeAnalysis('book/id one', 'job-1')
    await cancelAnalysis('book/id one', 'job-1')

    expect(result).toMatchObject({ success: true, task_id: 'job-1', run_id: 'run-1' })
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/pages',
      { params: { cursor: 0, limit: 100 } },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/analysis-jobs',
      {
        bookId: 'book/id one',
        scope: 'page',
        pageIds: ['page-1', 'page-2'],
        force: true,
      },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    for (const [index, command] of ['pause', 'resume', 'cancel'].entries()) {
      expect(postMock).toHaveBeenNthCalledWith(
        index + 2,
        `/api/v2/jobs/job-1/${command}`,
        undefined,
        { headers: { 'Idempotency-Key': expect.any(String) } },
      )
    }
  })

  it('routes overview, timeline, and vector work through backend jobs', async () => {
    postMock
      .mockResolvedValueOnce(accepted('overview-job'))
      .mockResolvedValueOnce(accepted('timeline-job'))
      .mockResolvedValueOnce(accepted('vector-job'))
    const {
      rebuildEmbeddings,
      regenerateOverview,
      regenerateTimeline,
    } = await import('@/api/insight')

    await regenerateOverview('book/id one', 'character_growth', true)
    await regenerateTimeline('book/id one')
    const vectorResult = await rebuildEmbeddings('book/id one')

    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/artifacts/overviews/character_growth',
      { bookId: 'book/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/insight/timeline',
      { bookId: 'book/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      3,
      '/api/v2/insight/books/book%2Fid%20one/vector-rebuild',
      {},
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(vectorResult).toMatchObject({ success: true, task_id: 'vector-job' })
  })

  it('loads complete notes in one page and updates citations with stable page IDs', async () => {
    getMock
      .mockResolvedValueOnce({ items: [note], nextCursor: null })
      .mockResolvedValueOnce({ items: pages, nextCursor: null })
    patchMock.mockResolvedValueOnce({
      ...note,
      revision: 8,
      citations: [{
        ...note.citations[0],
        excerpt: 'updated evidence',
      }],
    })
    deleteMock.mockResolvedValueOnce({ deleted: true })
    const {
      deleteNote,
      getNotes,
      updateNote,
    } = await import('@/api/insight')

    const loaded = await getNotes('book/id one', 'text')
    await updateNote('book/id one', 'note/id one', {
      citations: [{ page: 2, content: 'updated evidence' }],
    })
    await deleteNote('book/id one', 'note/id one')

    expect(loaded.notes?.[0]).toMatchObject({
      id: 'note/id one',
      content: 'note body',
      revision: 7,
    })
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/v2/insight/notes', {
      params: {
        bookId: 'book/id one',
        limit: 200,
        detail: 1,
        kind: 'text',
      },
    })
    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one',
      expect.objectContaining({
        baseRevision: 7,
        citations: [{ pageId: 'page-2', excerpt: 'updated evidence' }],
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one?baseRevision=8',
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })
})
