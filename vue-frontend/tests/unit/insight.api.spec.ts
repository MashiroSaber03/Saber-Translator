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
  comments: [{ text: 'note body', question: '', answer: '', comment: '' }],
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
    const timelineResult = await regenerateTimeline('book/id one')
    const vectorResult = await rebuildEmbeddings('book/id one')

    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/artifacts/overviews/character_growth',
      { bookId: 'book/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(timelineResult).toMatchObject({
      success: true,
      task_id: 'timeline-job',
    })
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

  it('serializes precise QA with the backend exact-mode contract', async () => {
    const body = [
      'event: context',
      'data: {"mode":"exact","citations":[{"pageId":"page-2","pageNumber":2}]}',
      '',
      'event: chunk',
      'data: {"text":"回答"}',
      '',
      'event: done',
      'data: {"suggestedQuestions":[]}',
      '',
      '',
    ].join('\n')
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }))
    vi.stubGlobal('fetch', fetchMock)
    const { sendChat } = await import('@/api/insight')

    const result = await sendChat('book/id one', '发生了什么？', {
      use_parent_child: true,
      use_reasoning: true,
      use_reranker: true,
      top_k: 5,
      threshold: 0,
    })

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(JSON.parse(String(request.body))).toMatchObject({
      question: '发生了什么？',
      mode: 'exact',
      useParentChild: true,
      useReasoning: true,
      useReranker: true,
      topK: 5,
      threshold: 0,
    })
    expect(result).toMatchObject({
      success: true,
      answer: '回答',
      citations: [{ page: 2 }],
    })
  })

  it('loads an existing overview without creating a duplicate rebuild job', async () => {
    getMock.mockResolvedValueOnce({
      artifactId: 'overview-1',
      bookId: 'book/id one',
      dependencyFingerprint: 'fingerprint',
      kind: 'overview',
      payload: { content: 'cached overview' },
      revision: 1,
      runId: null,
      status: 'ready',
      template: 'no_spoiler',
    })
    const { regenerateOverview } = await import('@/api/insight')

    const result = await regenerateOverview('book/id one', 'no_spoiler', false)

    expect(result).toMatchObject({
      success: true,
      content: 'cached overview',
      cached: true,
    })
    expect(postMock).not.toHaveBeenCalled()
  })

  it('does not project derived jobs as active page-analysis tasks', async () => {
    getMock.mockResolvedValueOnce({
      books: [{
        activeRun: 'run-1',
        analyzedPageCount: 14,
        bookId: 'book/id one',
        pageCount: 14,
      }],
      activeJobs: [{
        bookId: 'book/id one',
        jobId: 'timeline-job',
        kind: 'derived_rebuild',
        progress: { completedItems: 0, totalItems: 1 },
        status: 'running',
      }],
    })
    const { getAnalysisStatus } = await import('@/api/insight')

    const result = await getAnalysisStatus('book/id one')

    expect(result).toMatchObject({
      success: true,
      analyzed_pages_count: 14,
      fully_analyzed: true,
      status: 'completed',
    })
    expect(result.current_task).toBeUndefined()
  })

  it('maps durable timeline page IDs into renderable page-number groups', async () => {
    getMock
      .mockResolvedValueOnce({
        content: { overview: '故事概述' },
        mode: 'enhanced',
        events: [{
          eventId: 'event-1',
          page_ids: ['page-1', 'page-2'],
          summary: '关键事件',
        }],
        characters: [{
          name: '主角',
          first_page: 1,
          key_moments: ['首次登场（第1页）'],
        }],
      })
      .mockResolvedValueOnce({ items: pages, nextCursor: null })
    const { getTimeline } = await import('@/api/insight')

    const result = await getTimeline('book/id one')
    const timeline = result.timeline as unknown as Record<string, unknown>

    expect(timeline).toMatchObject({
      story_summary: '故事概述',
      groups: [{
        id: 'event-1',
        page_range: { start: 1, end: 2 },
        events: ['关键事件'],
      }],
      main_characters: [{
        name: '主角',
        first_appearance: 1,
      }],
      stats: {
        total_events: 1,
        total_pages: 2,
        total_characters: 1,
      },
    })
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
        comments: [{
          text: 'note body',
          question: '',
          answer: '',
          comment: '',
        }],
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one?baseRevision=8',
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })

  it('creates backend-valid note metadata comments with stable page citations', async () => {
    getMock.mockResolvedValueOnce({ items: pages, nextCursor: null })
    postMock.mockResolvedValueOnce(note)
    const { createNote } = await import('@/api/insight')

    await createNote('book/id one', {
      type: 'text',
      title: 'Note',
      content: 'note body',
      page_num: 2,
    })

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes',
      {
        bookId: 'book/id one',
        title: 'Note',
        content: 'note body',
        kind: 'text',
        tags: [],
        citations: [{ pageId: 'page-2', excerpt: '' }],
        comments: [{
          text: 'note body',
          question: '',
          answer: '',
          comment: '',
        }],
      },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })
})
