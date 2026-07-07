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

describe('insight api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    deleteMock.mockReset()
  })

  it('routes analysis and page endpoints through encoded book paths', async () => {
    const {
      cancelAnalysis,
      exportAnalysis,
      getAnalysisStatus,
      getAnalyzedPages,
      getInsightChapters,
      getPageData,
      getPageImageUrl,
      getThumbnailUrl,
      pauseAnalysis,
      previewAnalysis,
      reanalyzeChapter,
      reanalyzePage,
      resumeAnalysis,
      startAnalysis,
    } = await import('@/api/insight')

    const bookId = 'book/id one'
    const base = '/api/manga-insight/book%2Fid%20one'

    await startAnalysis(bookId, { mode: 'pages', pages: [1, 2], force: true })
    await pauseAnalysis(bookId, 'task/id one')
    await resumeAnalysis(bookId)
    await cancelAnalysis(bookId, 'task/id one')
    await getAnalysisStatus(bookId)
    await previewAnalysis(bookId, [1])
    await reanalyzePage(bookId, 3)
    await reanalyzeChapter(bookId, 'chapter/id one')
    await getPageData(bookId, 5)
    await getAnalyzedPages(bookId)
    await getInsightChapters(bookId)
    await exportAnalysis(bookId)

    expect(postMock).toHaveBeenNthCalledWith(1, `${base}/analyze/start`, {
      mode: 'pages',
      pages: [1, 2],
      force: true,
    }, { timeout: 0 })
    expect(postMock).toHaveBeenNthCalledWith(2, `${base}/analyze/pause`, { task_id: 'task/id one' })
    expect(postMock).toHaveBeenNthCalledWith(3, `${base}/analyze/resume`, { task_id: undefined })
    expect(postMock).toHaveBeenNthCalledWith(4, `${base}/analyze/cancel`, { task_id: 'task/id one' })
    expect(getMock).toHaveBeenNthCalledWith(1, `${base}/analyze/status`)
    expect(postMock).toHaveBeenNthCalledWith(5, `${base}/preview`, { pages: [1] }, { timeout: 0 })
    expect(postMock).toHaveBeenNthCalledWith(6, `${base}/reanalyze/page/3`, {}, { timeout: 0 })
    expect(postMock).toHaveBeenNthCalledWith(7, `${base}/reanalyze/chapter/chapter%2Fid%20one`, {}, { timeout: 0 })
    expect(getMock).toHaveBeenNthCalledWith(2, `${base}/pages/5`)
    expect(getMock).toHaveBeenNthCalledWith(3, `${base}/pages`)
    expect(getMock).toHaveBeenNthCalledWith(4, `${base}/chapters`)
    expect(getMock).toHaveBeenNthCalledWith(5, `${base}/export`)
    expect(getPageImageUrl(bookId, 6)).toBe(`${base}/page-image/6`)
    expect(getThumbnailUrl(bookId, 7)).toBe(`${base}/thumbnail/7`)
  })

  it('routes overview, timeline, QA, and embedding endpoints consistently', async () => {
    const {
      getChatStreamUrl,
      getOverview,
      getOverviewBasic,
      getRebuildEmbeddingsStatus,
      getTimeline,
      rebuildEmbeddings,
      regenerateOverview,
      regenerateTimeline,
      sendChat,
    } = await import('@/api/insight')

    const bookId = 'book/id one'
    const base = '/api/manga-insight/book%2Fid%20one'

    await getOverviewBasic(bookId)
    await getOverview(bookId, 'character_growth')
    await getOverview(bookId)
    await regenerateOverview(bookId, 'chapter_summary', true)
    await getTimeline(bookId)
    await regenerateTimeline(bookId)
    await sendChat(bookId, 'Who is the lead?', { top_k: 8, use_reranker: true })
    await rebuildEmbeddings(bookId)
    await getRebuildEmbeddingsStatus(bookId, 'task/id one')

    expect(getMock).toHaveBeenNthCalledWith(1, `${base}/overview`)
    expect(getMock).toHaveBeenNthCalledWith(2, `${base}/overview/character_growth`)
    expect(getMock).toHaveBeenNthCalledWith(3, `${base}/overview`)
    expect(postMock).toHaveBeenNthCalledWith(1, `${base}/overview/generate`, {
      template: 'chapter_summary',
      force: true,
    }, { timeout: 0 })
    expect(getMock).toHaveBeenNthCalledWith(4, `${base}/timeline`)
    expect(postMock).toHaveBeenNthCalledWith(2, `${base}/regenerate/timeline`, {}, { timeout: 0 })
    expect(postMock).toHaveBeenNthCalledWith(3, `${base}/chat`, {
      question: 'Who is the lead?',
      top_k: 8,
      use_reranker: true,
    }, { timeout: 0 })
    expect(postMock).toHaveBeenNthCalledWith(4, `${base}/rebuild-embeddings`, {}, { timeout: 0 })
    expect(getMock).toHaveBeenNthCalledWith(
      5,
      `${base}/rebuild-embeddings/status?task_id=task%2Fid+one`,
    )
    expect(getChatStreamUrl(bookId)).toBe(`${base}/chat`)
  })

  it('routes notes and prompt library endpoints through helpers', async () => {
    const {
      createNote,
      deleteNote,
      deletePromptFromLibrary,
      exportPageAnalysis,
      getDefaultPrompts,
      getNotes,
      getPromptsLibrary,
      importPromptsLibrary,
      savePromptToLibrary,
      updateNote,
    } = await import('@/api/insight')

    const bookId = 'book/id one'
    const base = '/api/manga-insight/book%2Fid%20one'

    await getNotes(bookId, 'qa')
    await createNote(bookId, { type: 'text', content: 'note', page_num: 2 })
    await updateNote(bookId, 'note/id one', { content: 'updated' })
    await deleteNote(bookId, 'note/id one')
    await getDefaultPrompts()
    await getPromptsLibrary()
    await savePromptToLibrary({
      id: 'prompt-1',
      name: 'Prompt',
      type: 'qa_response',
      content: 'content',
      created_at: 'now',
    })
    await deletePromptFromLibrary('prompt/id one')
    await importPromptsLibrary([])
    await exportPageAnalysis(bookId, 4)

    expect(getMock).toHaveBeenNthCalledWith(1, `${base}/notes`, {
      params: { type: 'qa' },
    })
    expect(postMock).toHaveBeenNthCalledWith(1, `${base}/notes`, {
      type: 'text',
      content: 'note',
      page_num: 2,
    })
    expect(putMock).toHaveBeenNthCalledWith(1, `${base}/notes/note%2Fid%20one`, { content: 'updated' })
    expect(deleteMock).toHaveBeenNthCalledWith(1, `${base}/notes/note%2Fid%20one`)
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/manga-insight/prompts/defaults')
    expect(getMock).toHaveBeenNthCalledWith(3, '/api/manga-insight/prompts/library')
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/manga-insight/prompts/library', {
      id: 'prompt-1',
      name: 'Prompt',
      type: 'qa_response',
      content: 'content',
      created_at: 'now',
    })
    expect(deleteMock).toHaveBeenNthCalledWith(2, '/api/manga-insight/prompts/library/prompt%2Fid%20one')
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/manga-insight/prompts/library/import', { library: [] })
    expect(getMock).toHaveBeenNthCalledWith(4, `${base}/pages/4`)
  })
})
