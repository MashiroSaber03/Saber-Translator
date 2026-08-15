import { beforeEach, describe, expect, it, vi } from 'vitest'

const { deleteMock, getMock, patchMock, postMock } = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  patchMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  class ApiClientError extends Error {
    constructor(
      public status: number,
      message = 'API request failed'
    ) {
      super(message)
    }
  }
  const apiClient = {
    delete: deleteMock,
    get: getMock,
    patch: patchMock,
    post: postMock,
    put: vi.fn(),
    upload: vi.fn(),
  }
  return { ApiClientError, apiClient, default: apiClient }
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
  citations: [
    {
      excerpt: 'evidence',
      pageId: 'page-2',
      pageIdSnapshot: 'page-2',
      pageNumberSnapshot: 2,
      score: null,
      sourceAnalysisId: null,
    },
  ],
  comment: null,
  content: 'note body',
  createdAt: '2026-07-01T00:00:00Z',
  excerpt: 'note body',
  kind: 'text',
  noteId: 'note/id one',
  question: null,
  revision: 7,
  tags: [],
  title: 'Note',
  updatedAt: '2026-07-01T00:00:00Z',
}

function accepted(jobId: string) {
  return { batchId: 'batch-1', jobIds: [jobId], runId: 'run-1', status: 'queued' }
}

const preciseChatOptions = {
  mode: 'precise' as const,
  threshold: 0,
  topK: 5,
  useParentChild: true,
  useReasoning: true,
  useReranker: true,
}

function timelineResponse(plotArcs: unknown[]) {
  return {
    timelineVersionId: 'timeline-plot-arcs',
    bookId: 'book/id one',
    runId: 'run-1',
    mode: 'enhanced',
    status: 'ready',
    content: {
      requested_mode: 'enhanced',
      actual_mode: 'enhanced',
      fallback_reason: null,
      degraded: false,
      story_summary: '完整剧情',
      plot_arcs: plotArcs,
    },
    events: [
      {
        eventId: 'event-1',
        summary: '事件',
        page_ids: ['page-1'],
        page_numbers: [1],
      },
    ],
    characters: [],
    eventPage: { nextCursor: null, totalCount: 1 },
    characterPage: { nextCursor: null, totalCount: 0 },
    pageCount: 1,
    pageThumbnails: { '1': '/api/v2/assets/thumb-1' },
    dependencyFingerprint: 'd'.repeat(64),
  }
}

describe('insight v2 api', () => {
  beforeEach(() => {
    vi.resetModules()
    deleteMock.mockReset()
    getMock.mockReset()
    patchMock.mockReset()
    postMock.mockReset()
  })

  it('resolves display page numbers to stable IDs and controls durable jobs', async () => {
    getMock
      .mockResolvedValueOnce({ items: [pages[0]], nextCursor: 1 })
      .mockResolvedValueOnce({ items: [pages[1]], nextCursor: null })
    postMock.mockResolvedValueOnce(accepted('job-1')).mockResolvedValue({})
    const { cancelAnalysis, pauseAnalysis, resumeAnalysis, startAnalysis } =
      await import('@/api/insight')

    const result = await startAnalysis('book/id one', {
      mode: 'pages',
      pages: [1, 2],
    })
    await pauseAnalysis('job-1')
    await resumeAnalysis('job-1')
    await cancelAnalysis('job-1')

    expect(result).toEqual({ jobId: 'job-1', runId: 'run-1' })
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/v2/insight/books/book%2Fid%20one/pages', {
      params: { cursor: 0, limit: 1 },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/v2/insight/books/book%2Fid%20one/pages', {
      params: { cursor: 1, limit: 1 },
    })
    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/analysis-jobs',
      {
        bookId: 'book/id one',
        scope: 'page',
        pageIds: ['page-1', 'page-2'],
      },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    for (const [index, command] of ['pause', 'resume', 'cancel'].entries()) {
      expect(postMock).toHaveBeenNthCalledWith(
        index + 2,
        `/api/v2/jobs/job-1/${command}`,
        undefined,
        { headers: { 'Idempotency-Key': expect.any(String) } }
      )
    }
  })

  it('re-resolves page numbers for every command instead of reusing stale page IDs', async () => {
    getMock.mockResolvedValueOnce({ items: [pages[0]], nextCursor: null }).mockResolvedValueOnce({
      items: [{ ...pages[0], pageId: 'page-1-after-reorder' }],
      nextCursor: null,
    })
    postMock
      .mockResolvedValueOnce(accepted('job-before-reorder'))
      .mockResolvedValueOnce(accepted('job-after-reorder'))
    const { reanalyzePage } = await import('@/api/insight')

    await reanalyzePage('book/id one', 1)
    await reanalyzePage('book/id one', 1)

    expect(getMock).toHaveBeenCalledTimes(2)
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/insight/analysis-jobs',
      expect.objectContaining({ pageIds: ['page-1-after-reorder'] }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })

  it('rejects an unresolved page selection instead of silently submitting a partial job', async () => {
    getMock
      .mockResolvedValueOnce({ items: [pages[0]], nextCursor: 1 })
      .mockResolvedValueOnce({ items: [], nextCursor: null })
    const { startAnalysis } = await import('@/api/insight')

    await expect(startAnalysis('book/id one', { mode: 'pages', pages: [1, 2] })).rejects.toThrow(
      '第 2 页不存在'
    )
    expect(postMock).not.toHaveBeenCalled()
  })

  it('routes overview, timeline, and vector work through backend jobs', async () => {
    postMock
      .mockResolvedValueOnce(accepted('overview-job'))
      .mockResolvedValueOnce(accepted('timeline-job'))
      .mockResolvedValueOnce(accepted('vector-job'))
    const { rebuildEmbeddings, regenerateOverview, regenerateTimeline } =
      await import('@/api/insight')

    await regenerateOverview('book/id one', 'character_guide', true)
    const timelineResult = await regenerateTimeline('book/id one')
    const vectorResult = await rebuildEmbeddings('book/id one')

    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/artifacts/overviews/character_guide',
      { bookId: 'book/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(timelineResult).toBe('timeline-job')
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/insight/timeline',
      { bookId: 'book/id one' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(postMock).toHaveBeenNthCalledWith(
      3,
      '/api/v2/insight/books/book%2Fid%20one/vector-rebuild',
      {},
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(vectorResult).toBe('vector-job')
  })

  it('reads mode-specific QA readiness and rebuilds global compressed context', async () => {
    getMock.mockResolvedValueOnce({
      available: false,
      reason: 'compressed_context_missing',
      repairAction: 'compressed_context_rebuild',
    })
    postMock.mockResolvedValueOnce(accepted('compressed-context-job'))
    const { getQAStatus, rebuildCompressedContext } = await import('@/api/insight')

    const status = await getQAStatus('book/id one', 'global')
    const rebuild = await rebuildCompressedContext('book/id one')

    expect(status).toMatchObject({
      available: false,
      repairAction: 'compressed_context_rebuild',
    })
    expect(getMock).toHaveBeenCalledWith('/api/v2/insight/qa/status', {
      params: { bookId: 'book/id one', mode: 'global' },
    })
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/compressed-context/rebuild',
      {},
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(rebuild).toBe('compressed-context-job')
  })

  it('serializes precise QA with the backend exact-mode contract', async () => {
    const body = [
      'event: status',
      'data: {"requestId":"request-1","status":"retrieving"}',
      '',
      'event: context',
      'data: {"mode":"exact","citations":[{"pageId":"page-2","pageNumber":2,"excerpt":"证据","score":0.9}]}',
      '',
      'event: chunk',
      'data: {"text":"回答"}',
      '',
      'event: done',
      'data: {}',
      '',
      '',
    ].join('\n')
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    )
    vi.stubGlobal('fetch', fetchMock)
    const { sendChat } = await import('@/api/insight')
    const streamed: string[] = []
    const abortController = new AbortController()

    const result = await sendChat('book/id one', '发生了什么？', {
      ...preciseChatOptions,
      onChunk: content => streamed.push(content),
      signal: abortController.signal,
    })

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(request.signal).toBe(abortController.signal)
    expect(JSON.parse(String(request.body))).toMatchObject({
      question: '发生了什么？',
      mode: 'exact',
      useParentChild: true,
      useReasoning: true,
      useReranker: true,
      topK: 5,
      threshold: 0,
    })
    expect(result).toMatchObject({ answer: '回答', citations: [{ page: 2 }] })
    expect(streamed).toEqual(['回答'])
  })

  it('omits precise retrieval settings from a global QA command', async () => {
    const body = [
      'event: status',
      'data: {"requestId":"request-global","status":"retrieving"}',
      '',
      'event: context',
      'data: {"mode":"global","citations":[]}',
      '',
      'event: chunk',
      'data: {"text":"全局回答"}',
      '',
      'event: done',
      'data: {}',
      '',
      '',
    ].join('\n')
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { sendChat } = await import('@/api/insight')

    const result = await sendChat('book-1', '概括故事', { mode: 'global' })

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(JSON.parse(String(request.body))).toEqual({
      question: '概括故事',
      mode: 'global',
    })
    expect(result).toEqual({ answer: '全局回答', citations: [], mode: 'global' })
  })

  it('rejects answer chunks that arrive before the context event', async () => {
    const body = [
      'event: status',
      'data: {"requestId":"request-1","status":"retrieving"}',
      '',
      'event: chunk',
      'data: {"text":"乱序回答"}',
      '',
      'event: context',
      'data: {"mode":"exact","citations":[]}',
      '',
      'event: done',
      'data: {}',
      '',
      '',
    ].join('\n')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status: 200 })))
    const { sendChat } = await import('@/api/insight')

    await expect(sendChat('book-1', '问题', preciseChatOptions)).rejects.toThrow(
      '问答 chunk 事件字段无效'
    )
  })

  it('rejects malformed QA event fields instead of coercing them', async () => {
    const body = [
      'event: status',
      'data: {"requestId":"request-1","status":"retrieving"}',
      '',
      'event: context',
      'data: {"mode":"exact","citations":[{"pageId":"page-2","pageNumber":"2","excerpt":"证据","score":0.9}]}',
      '',
      '',
    ].join('\n')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )
    )
    const { sendChat } = await import('@/api/insight')

    await expect(sendChat('book-1', '问题', preciseChatOptions)).rejects.toThrow('问答引用字段无效')
  })

  it('rejects a QA stream that closes without a done event', async () => {
    const body = [
      'event: status',
      'data: {"requestId":"request-1","status":"retrieving"}',
      '',
      'event: context',
      'data: {"mode":"exact","citations":[]}',
      '',
      'event: chunk',
      'data: {"text":"未完成回答"}',
      '',
      '',
    ].join('\n')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        })
      )
    )
    const { sendChat } = await import('@/api/insight')

    await expect(sendChat('book-1', '问题', preciseChatOptions)).rejects.toThrow('问答响应意外中断')
  })

  it('loads an existing overview without creating a duplicate rebuild job', async () => {
    getMock.mockResolvedValueOnce({
      artifactId: 'overview-1',
      bookId: 'book/id one',
      dependencyFingerprint: 'fingerprint',
      kind: 'overview',
      payload: { title: 'Overview', content: 'cached overview' },
      revision: 1,
      runId: null,
      status: 'ready',
      template: 'no_spoiler',
    })
    const { regenerateOverview } = await import('@/api/insight')

    const result = await regenerateOverview('book/id one', 'no_spoiler', false)

    expect(result).toEqual({ kind: 'cached', content: 'cached overview' })
    expect(postMock).not.toHaveBeenCalled()
  })

  it('loads generated overview template names with one backend request', async () => {
    getMock.mockResolvedValueOnce({
      items: ['story_summary', 'no_spoiler'],
    })
    const { getGeneratedTemplates } = await import('@/api/insight')

    const templates = await getGeneratedTemplates('book/id one')

    expect(templates).toEqual(['story_summary', 'no_spoiler'])
    expect(getMock).toHaveBeenCalledTimes(1)
    expect(getMock).toHaveBeenCalledWith('/api/v2/insight/artifacts/overviews', {
      params: { bookId: 'book/id one' },
    })
  })

  it('rejects unknown overview templates instead of silently hiding them', async () => {
    getMock.mockResolvedValueOnce({ items: ['no_spoiler', 'legacy_template'] })
    const { getGeneratedTemplates } = await import('@/api/insight')

    await expect(getGeneratedTemplates('book/id one')).rejects.toThrow('漫画概览模板响应格式无效')
  })

  it('projects a strict timeline response without synthetic fallback values', async () => {
    getMock.mockResolvedValueOnce({
      timelineVersionId: 'timeline-1',
      bookId: 'book/id one',
      runId: 'run-1',
      mode: 'enhanced',
      status: 'ready',
      content: {
        requested_mode: 'enhanced',
        actual_mode: 'enhanced',
        fallback_reason: null,
        degraded: false,
        story_summary: '完整剧情',
      },
      events: [
        {
          eventId: 'event-1',
          summary: '事件',
          page_ids: ['page-2'],
          page_numbers: [2],
        },
      ],
      characters: [
        {
          characterId: 'character-1',
          name: 'Saber',
          description: '主角',
          first_page: 2,
          key_moments: [{ summary: '登场', page: 2 }],
        },
      ],
      eventPage: { nextCursor: null, totalCount: 1 },
      characterPage: { nextCursor: null, totalCount: 1 },
      pageCount: 2,
      pageThumbnails: { '2': '/api/v2/assets/thumb-2' },
      dependencyFingerprint: 'a'.repeat(64),
    })
    const { getTimeline } = await import('@/api/insight')

    const timeline = await getTimeline('book/id one')

    expect(timeline).toMatchObject({
      timeline_version_id: 'timeline-1',
      groups: [
        {
          id: 'event-1',
          page_range: { start: 2, end: 2 },
          events: ['事件'],
        },
      ],
      main_characters: [
        {
          character_id: 'character-1',
          name: 'Saber',
          first_appearance: 2,
        },
      ],
      stats: { total_events: 1, total_pages: 2, total_characters: 1 },
      page_thumbnails: { 2: '/api/v2/assets/thumb-2' },
    })
  })

  it('rejects a timeline event with incomplete page references', async () => {
    getMock.mockResolvedValueOnce({
      timelineVersionId: 'timeline-1',
      bookId: 'book/id one',
      runId: null,
      mode: 'simple',
      status: 'ready',
      content: {
        requested_mode: 'enhanced',
        actual_mode: 'simple',
        fallback_reason: 'provider output invalid',
        degraded: true,
        story_summary: '',
      },
      events: [
        {
          eventId: 'event-1',
          summary: '事件',
          page_ids: ['page-1'],
        },
      ],
      characters: [],
      eventPage: { nextCursor: null, totalCount: 1 },
      characterPage: { nextCursor: null, totalCount: 0 },
      pageCount: 1,
      pageThumbnails: {},
      dependencyFingerprint: 'b'.repeat(64),
    })
    const { getTimeline } = await import('@/api/insight')

    await expect(getTimeline('book/id one')).rejects.toThrow('页面引用格式无效')
  })

  it('rejects missing or duplicate plot arc identities', async () => {
    getMock
      .mockResolvedValueOnce(
        timelineResponse([{ name: '缺少身份', page_range: { start: 1, end: 1 } }])
      )
      .mockResolvedValueOnce(
        timelineResponse([
          { id: 'arc-1', name: '开端', description: '开端描述', page_range: { start: 1, end: 1 } },
          { id: 'arc-1', name: '重复', description: '重复描述', page_range: { start: 1, end: 1 } },
        ])
      )
    const { getTimeline } = await import('@/api/insight')

    await expect(getTimeline('book/id one')).rejects.toThrow('plot_arcs[0].id')
    await expect(getTimeline('book/id one')).rejects.toThrow('plot_arcs[1].id 重复')
  })

  it('loads real recent page analyses from one bounded backend query', async () => {
    getMock.mockResolvedValueOnce({
      items: [
        {
          pageId: 'page-13',
          displayPageNumber: 13,
          summary: '真实最近页面摘要',
          generatedAt: '2026-08-04T06:00:00Z',
        },
      ],
    })
    const { getRecentAnalyzedPages } = await import('@/api/insight')

    const pages = await getRecentAnalyzedPages('book/id one')

    expect(pages).toEqual([
      {
        page_num: 13,
        summary: '真实最近页面摘要',
      },
    ])
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/recent-page-analyses',
      { params: { limit: 5 } }
    )
  })

  it('does not project derived jobs as active page-analysis tasks', async () => {
    getMock.mockResolvedValueOnce({
      books: [
        {
          activeRun: 'run-1',
          analyzedPageCount: 14,
          bookId: 'book/id one',
          pageCount: 14,
        },
      ],
      activeJobs: [
        {
          bookId: 'book/id one',
          jobId: 'timeline-job',
          kind: 'derived_rebuild',
          progress: { completedItems: 0, totalItems: 1 },
          status: 'running',
        },
      ],
    })
    const { getAnalysisStatus } = await import('@/api/insight')

    const result = await getAnalysisStatus('book/id one')

    expect(result).toMatchObject({
      analyzedPagesCount: 14,
      fullyAnalyzed: true,
    })
    expect(result.currentTask).toBeUndefined()
  })

  it('rejects an analysis status snapshot that omits the requested book', async () => {
    getMock.mockResolvedValueOnce({ books: [], activeJobs: [] })
    const { getAnalysisStatus } = await import('@/api/insight')

    await expect(getAnalysisStatus('missing-book')).rejects.toThrow('漫画分析书籍不存在')
  })

  it('rejects impossible analysis counts and unknown task states', async () => {
    const { getAnalysisStatus } = await import('@/api/insight')
    getMock.mockResolvedValueOnce({
      books: [{ bookId: 'book-1', pageCount: 2, analyzedPageCount: 3, activeRun: null }],
      activeJobs: [],
    })
    await expect(getAnalysisStatus('book-1')).rejects.toThrow('已分析页数超过总页数')

    getMock.mockResolvedValueOnce({
      books: [{ bookId: 'book-1', pageCount: 2, analyzedPageCount: 1, activeRun: null }],
      activeJobs: [{
        bookId: 'book-1',
        jobId: 'job-1',
        kind: 'insight_analysis',
        progress: {
          completedItems: 0,
          totalItems: 1,
          pools: [],
        },
        status: 'unknown',
      }],
    })
    await expect(getAnalysisStatus('book-1')).rejects.toThrow('任务状态格式无效')
  })

  it('does not mark a zero-page chapter as analyzed', async () => {
    getMock.mockResolvedValueOnce({
      items: [
        {
          chapterId: 'chapter-empty',
          title: '空章节',
          pageCount: 0,
          analysisCounts: {
            notAnalyzed: 0,
            running: 0,
            failed: 0,
            stale: 0,
            ready: 0,
          },
        },
      ],
    })
    const { getInsightChapters } = await import('@/api/insight')

    await expect(getInsightChapters('book/id one')).resolves.toEqual([
      {
        id: 'chapter-empty',
        title: '空章节',
        startPage: 0,
        endPage: 0,
        analyzed: false,
        analyzedCount: 0,
      },
    ])
  })

  it('maps durable timeline page IDs into renderable page-number groups', async () => {
    getMock.mockResolvedValueOnce({
      timelineVersionId: 'timeline-2',
      bookId: 'book/id one',
      runId: 'run-1',
      content: {
        requested_mode: 'enhanced',
        actual_mode: 'enhanced',
        fallback_reason: null,
        degraded: false,
        story_summary: '故事概述',
      },
      mode: 'enhanced',
      status: 'ready',
      events: [
        {
          eventId: 'event-1',
          page_ids: ['page-1', 'page-2'],
          page_numbers: [1, 2],
          summary: '关键事件',
        },
      ],
      characters: [
        {
          characterId: 'character-1',
          name: '主角',
          description: '主角描述',
          first_page: 1,
          key_moments: [{ summary: '首次登场', page: 1 }],
        },
      ],
      eventPage: { nextCursor: null, totalCount: 1 },
      characterPage: { nextCursor: null, totalCount: 1 },
      pageCount: 2,
      pageThumbnails: { '1': '/api/v2/assets/thumb-1' },
      dependencyFingerprint: 'c'.repeat(64),
    })
    const { getTimeline } = await import('@/api/insight')

    const result = await getTimeline('book/id one')
    expect(result).toMatchObject({
      story_summary: '故事概述',
      groups: [
        {
          id: 'event-1',
          page_range: { start: 1, end: 2 },
          events: ['关键事件'],
        },
      ],
      main_characters: [
        {
          name: '主角',
          first_appearance: 1,
        },
      ],
      stats: {
        total_events: 1,
        total_pages: 2,
        total_characters: 1,
      },
      page_thumbnails: { 1: '/api/v2/assets/thumb-1' },
    })
  })

  it('reads the current note revision before every mutation and reuses stable citation IDs', async () => {
    getMock
      .mockResolvedValueOnce({ items: [note], nextCursor: null })
      .mockResolvedValueOnce({ ...note, revision: 9 })
      .mockResolvedValueOnce({ ...note, revision: 10 })
    patchMock.mockResolvedValueOnce({
      ...note,
      revision: 10,
      citations: [
        {
          ...note.citations[0],
          excerpt: 'updated evidence',
        },
      ],
    })
    deleteMock.mockResolvedValueOnce({ deleted: true })
    const { deleteNote, getNotes, updateNote } = await import('@/api/insight')

    const loaded = await getNotes('book/id one', 'text')
    await updateNote('book/id one', 'note/id one', {
      citations: [{ page: 2, content: 'updated evidence' }],
    })
    await deleteNote('note/id one')

    expect(loaded.items[0]).toMatchObject({
      id: 'note/id one',
      content: 'note body',
      revision: 7,
    })
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/v2/insight/notes', {
      params: {
        bookId: 'book/id one',
        limit: 50,
        kind: 'text',
      },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/v2/insight/notes/note%2Fid%20one')
    expect(getMock).toHaveBeenNthCalledWith(3, '/api/v2/insight/notes/note%2Fid%20one')
    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one',
      expect.objectContaining({
        baseRevision: 9,
        citations: [{ pageId: 'page-2', excerpt: 'updated evidence' }],
        question: null,
        comment: null,
      }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one?baseRevision=10',
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })

  it('updates a manual note page by resolving the new page to its stable ID', async () => {
    getMock
      .mockResolvedValueOnce(note)
      .mockResolvedValueOnce({ items: [pages[0]], nextCursor: null })
    patchMock.mockResolvedValueOnce({
      ...note,
      revision: 8,
      citations: [
        {
          excerpt: '',
          pageId: 'page-1',
          pageIdSnapshot: 'page-1',
          pageNumberSnapshot: 1,
          score: null,
          sourceAnalysisId: null,
        },
      ],
    })
    const { updateNote } = await import('@/api/insight')

    await updateNote('book/id one', 'note/id one', { pageNum: 1 })

    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/insight/notes/note%2Fid%20one',
      expect.objectContaining({ citations: [{ pageId: 'page-1', excerpt: '' }] }),
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })

  it('rejects malformed current note DTOs at the API boundary', async () => {
    getMock.mockResolvedValueOnce({
      items: [{ ...note, revision: 0 }],
      nextCursor: null,
    })
    const { getNotes } = await import('@/api/insight')

    await expect(getNotes('book/id one')).rejects.toThrow('笔记响应格式无效')
  })

  it('creates a note with explicit metadata and stable page citations', async () => {
    getMock.mockResolvedValueOnce({ items: [pages[1]], nextCursor: null })
    postMock.mockResolvedValueOnce(note)
    const { createNote } = await import('@/api/insight')

    await createNote('book/id one', {
      type: 'text',
      title: 'Note',
      content: 'note body',
      pageNum: 2,
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
        question: null,
        comment: null,
      },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })

  it('rejects a missing citation page instead of publishing a note without it', async () => {
    getMock.mockResolvedValueOnce({ items: [], nextCursor: null })
    const { createNote } = await import('@/api/insight')

    await expect(
      createNote('book/id one', {
        type: 'text',
        title: 'Note',
        content: 'note body',
        citations: [{ page: 99, content: 'missing evidence' }],
      })
    ).rejects.toThrow('引用的第 99 页不存在')
    expect(postMock).not.toHaveBeenCalled()
  })
})
