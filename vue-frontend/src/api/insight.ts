import type { FetchModelsResponse } from '@/types'
import type {
  InsightAnalysisSnapshot,
  InsightTaskStatus,
  OverviewTemplateType,
  TimelineData,
} from '@/types/insight'
import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'
import { projectInsightPageProgress } from '@/utils/insightJobProgress'
import { readApiErrorMessage } from '@/api/download'
import { readSseStream } from '@/api/sse'
import { ApiClientError } from '@/api/client'
import { jobsApi } from '@/api/v2/jobs'
import {
  createInsightAnalysisJob,
  createInsightExport,
  createInsightNote,
  deleteInsightNote,
  getInsightBootstrap,
  getInsightOverview,
  getInsightNote,
  getInsightPage,
  getInsightQaStatus,
  getInsightTimeline,
  insightCurrentExportUrl,
  insightPageExportUrl,
  insightQaUrl,
  listInsightNotes,
  listInsightOverviewTemplates,
  listInsightPages,
  listInsightChapters,
  listRecentInsightPageAnalyses,
  rebuildInsightOverview,
  rebuildInsightCompressedContext,
  rebuildInsightTimeline,
  rebuildInsightVectors,
  updateInsightNote,
  type V2InsightNote,
  type V2InsightPageSummary,
} from '@/api/v2/insight'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import {
  createV2Prompt,
  deleteV2Prompt,
  fetchV2ModelCatalog,
  getV2Settings,
  listV2Prompts,
  runV2ConnectionTest,
  saveV2SettingsTransaction,
  updateV2Prompt,
  type V2CredentialEdit,
  type V2CredentialSummary,
  type V2Prompt,
  type V2ProviderSettingEntry,
  type V2ProviderSettingMutation,
  type V2SettingsDocument,
  type V2ConnectionTestResult,
} from '@/api/v2/settings'

export interface PageAnalysisData {
  analyzed?: boolean
  analyzed_at?: string
  continuity_notes?: string
  key_events?: Array<{
    event_type?: string
    importance: string
    summary: string
  }>
  page_num?: number
  page_summary?: string
  warnings?: Array<{ code: string; message: string }>
}

export interface PageData {
  analysis: PageAnalysisData
  sourceUrl: string
}

export interface InsightChapter {
  analyzed: boolean
  end_page: number
  id: string
  start_page: number
  title: string
}

export interface NoteData {
  answer?: string
  citations?: Array<{ content: string; page: number }>
  comment?: string
  content: string
  createdAt: string
  id: string
  pageNum?: number
  question?: string
  revision?: number
  tags?: string[]
  title?: string
  type: 'text' | 'qa'
  updatedAt: string
}

export interface VlmConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  openai_options?: OpenAICompatibleOptionsWire
  image_max_size?: number
}

export interface LlmConfig {
  use_same_as_vlm: boolean
  provider?: string
  api_key?: string
  model?: string
  base_url?: string
  openai_options?: OpenAICompatibleOptionsWire
}

export interface EmbeddingConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface RerankerConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  top_k?: number
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface ImageGenConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface BatchAnalysisConfig {
  pages_per_batch: number
  context_batch_count: number
  architecture_preset: string
  custom_layers?: Array<{
    name: string
    units_per_group: number
    align_to_chapter: boolean
  }>
}

export interface AnalysisConfig {
  analysis?: { batch?: BatchAnalysisConfig }
  chat_llm?: LlmConfig
  embedding?: EmbeddingConfig
  image_gen?: ImageGenConfig
  prompts?: Record<string, string>
  provider_settings?: Record<string, Record<string, Record<string, unknown>>>
  reranker?: RerankerConfig
  vlm?: VlmConfig
}

export interface StartAnalysisOptions {
  mode?: 'full' | 'incremental' | 'chapters' | 'pages'
  chapters?: string[]
  pages?: number[]
  force?: boolean
}

export interface AnalysisJobSubmission {
  jobId: string
  runId?: string
}

export type OverviewGenerationResult =
  | { kind: 'cached'; content: string }
  | { kind: 'queued'; jobId: string }

export interface ChatResult {
  answer: string
  mode: string
  citations: Array<{ page: number }>
  suggestedQuestions: string[]
}

export interface QAStatusResponse {
  available: boolean
  coverage?: {
    events: number
    pages: number
  }
  generation?: number
  reason: string | null
  repairAction?: 'analyze' | 'vector_rebuild' | 'overview_rebuild' | 'compressed_context_rebuild'
}

export type PromptType = 'batch_analysis' | 'segment_summary' | 'chapter_summary' | 'qa_response'

export interface PromptMetadata {
  label: string
  hint: string
}

export const PROMPT_METADATA: Record<PromptType, PromptMetadata> = {
  batch_analysis: {
    label: '批量分析提示词',
    hint: '用于批量分析多个页面。支持变量：{page_count}, {start_page}, {end_page}',
  },
  segment_summary: {
    label: '段落总结提示词',
    hint: '用于汇总多个批次的分析结果生成段落总结。',
  },
  chapter_summary: {
    label: '章节总结提示词',
    hint: '用于生成章节级别的完整总结。',
  },
  qa_response: {
    label: '问答响应提示词',
    hint: '用于回答用户关于漫画内容的问题。',
  },
}

export interface SavedPromptItem {
  id: string
  name: string
  type: PromptType
  content: string
  created_at: string
}

const INSIGHT_DOMAINS = [
  'insight',
  'insight_vlm',
  'insight_chat',
  'insight_embedding',
  'insight_reranker',
  'insight_image_gen',
]
const PROVIDER_GROUPS = {
  vlmProvider: 'insight_vlm',
  llmProvider: 'insight_chat',
  embeddingProvider: 'insight_embedding',
  rerankerProvider: 'insight_reranker',
  imageGenProvider: 'insight_image_gen',
} as const
const SECTION_DOMAINS = {
  vlm: 'insight_vlm',
  chat_llm: 'insight_chat',
  embedding: 'insight_embedding',
  reranker: 'insight_reranker',
  image_gen: 'insight_image_gen',
} as const
const OVERVIEW_TEMPLATES = [
  'no_spoiler',
  'story_summary',
  'recap',
  'character_guide',
  'world_setting',
  'highlights',
  'reading_notes',
] as const satisfies readonly OverviewTemplateType[]

let settingsDocument: V2SettingsDocument | null = null
let promptCache: V2Prompt[] = []
let credentialSummaries: V2CredentialSummary[] = []
const PAGE_CACHE_LIMIT = 300
const NOTE_CACHE_LIMIT = 500
const pageCache = new Map<string, Map<string, V2InsightPageSummary>>()
const timelineThumbnailCache = new Map<number, string>()
const noteCache = new Map<string, NoteData>()
const noteCitationPageIds = new Map<string, Map<number, string>>()

async function boundedMap<T, R>(
  items: readonly T[],
  mapper: (item: T, index: number) => Promise<R>,
  concurrency = 4,
): Promise<R[]> {
  if (items.length === 0) return []
  const results = new Array<R>(items.length)
  let nextIndex = 0
  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await mapper(items[index] as T, index)
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, () => worker()),
  )
  return results
}

let cachedPageBookId: string | null = null
let cachedTimelineThumbnailBookId: string | null = null

function rememberPages(bookId: string, pages: V2InsightPageSummary[]): void {
  if (cachedPageBookId !== bookId) {
    pageCache.clear()
    cachedPageBookId = bookId
  }
  const byId = pageCache.get(bookId) ?? new Map<string, V2InsightPageSummary>()
  pages.forEach(page => {
    byId.delete(page.pageId)
    byId.set(page.pageId, page)
  })
  while (byId.size > PAGE_CACHE_LIMIT) {
    const oldestId = byId.keys().next().value
    if (!oldestId) break
    byId.delete(oldestId)
  }
  pageCache.set(bookId, byId)
}

function rememberTimelineThumbnails(bookId: string, thumbnails: Record<string, string>): void {
  if (cachedTimelineThumbnailBookId !== bookId) {
    timelineThumbnailCache.clear()
    cachedTimelineThumbnailBookId = bookId
  }
  Object.entries(thumbnails).forEach(([pageNumber, url]) => {
    const parsed = Number(pageNumber)
    if (Number.isInteger(parsed) && parsed > 0 && url) {
      timelineThumbnailCache.delete(parsed)
      timelineThumbnailCache.set(parsed, url)
    }
  })
  while (timelineThumbnailCache.size > PAGE_CACHE_LIMIT) {
    const oldestPage = timelineThumbnailCache.keys().next().value
    if (oldestPage === undefined) break
    timelineThumbnailCache.delete(oldestPage)
  }
}

function cachedPageForNumber(
  bookId: string,
  pageNum: number,
): V2InsightPageSummary | undefined {
  for (const page of pageCache.get(bookId)?.values() ?? []) {
    if (page.displayPageNumber === pageNum) return page
  }
  return undefined
}

export async function getInsightPagesPage(
  bookId: string,
  options: { chapterId?: string; cursor?: number; limit?: number } = {}
): Promise<{ items: V2InsightPageSummary[]; nextCursor: number | null }> {
  const response = await listInsightPages(bookId, options)
  rememberPages(bookId, response.items)
  return response
}

async function pageForNumber(
  bookId: string,
  pageNum: number
): Promise<V2InsightPageSummary | undefined> {
  const cached = cachedPageForNumber(bookId, pageNum)
  if (cached) return cached
  if (!Number.isInteger(pageNum) || pageNum < 1) return undefined
  const response = await getInsightPagesPage(bookId, { cursor: pageNum - 1, limit: 1 })
  return response.items.find(page => page.displayPageNumber === pageNum)
}

function mapJobStatus(status: string): InsightTaskStatus {
  if (
    status === 'queued' ||
    status === 'running' ||
    status === 'pausing' ||
    status === 'paused' ||
    status === 'cancelling' ||
    status === 'interrupted' ||
    status === 'completed' ||
    status === 'completed_with_errors' ||
    status === 'cancelled' ||
    status === 'failed'
  ) {
    return status
  }
  return 'failed'
}

export async function startAnalysis(
  bookId: string,
  options: StartAnalysisOptions = {}
): Promise<AnalysisJobSubmission> {
  const mode = options.mode ?? 'full'
  const scope = mode === 'chapters' ? 'chapter' : mode === 'pages' ? 'page' : mode
  let pageIds: string[] | undefined
  if (scope === 'page') {
    const pages = await boundedMap(
      options.pages ?? [],
      page => pageForNumber(bookId, page),
    )
    pageIds = pages.flatMap(page => page ? [page.pageId] : [])
  }
  const accepted = await createInsightAnalysisJob({
    bookId,
    scope,
    ...(scope === 'chapter' ? { chapterIds: options.chapters ?? [] } : {}),
    ...(scope === 'page' ? { pageIds: pageIds ?? [] } : {}),
    force: options.force,
  })
  return {
    jobId: accepted.jobIds[0],
    ...(accepted.runId ? { runId: accepted.runId } : {}),
  }
}

export async function pauseAnalysis(taskId: string): Promise<void> {
  await jobsApi.pause(taskId)
}

export async function resumeAnalysis(taskId: string): Promise<void> {
  await jobsApi.resume(taskId)
}

export async function continueAnalysis(taskId: string): Promise<void> {
  await jobsApi.continue(taskId)
}

export async function cancelAnalysis(taskId: string): Promise<void> {
  await jobsApi.cancel(taskId)
}

export async function getAnalysisStatus(bookId: string): Promise<InsightAnalysisSnapshot> {
  const bootstrap = await getInsightBootstrap()
  const book = bootstrap.books.find(item => item.bookId === bookId)
  const job = bootstrap.activeJobs.find(
    item => item.bookId === bookId && item.kind === 'insight_analysis'
  )
  const pageProgress = job ? projectInsightPageProgress(job.progress) : undefined
  return {
    fullyAnalyzed: Boolean(book && book.pageCount > 0 && book.analyzedPageCount >= book.pageCount),
    analyzedPagesCount: book?.analyzedPageCount ?? 0,
    currentTask: job
      ? {
          jobId: job.jobId,
          status: mapJobStatus(job.status),
          progress: {
            analyzedPages: pageProgress?.current ?? 0,
            totalPages: pageProgress?.total ?? 0,
          },
        }
      : undefined,
  }
}

export function reanalyzePage(bookId: string, pageNum: number): Promise<AnalysisJobSubmission> {
  return startAnalysis(bookId, { mode: 'pages', pages: [pageNum], force: true })
}

export function reanalyzeChapter(
  bookId: string,
  chapterId: string
): Promise<AnalysisJobSubmission> {
  return startAnalysis(bookId, { mode: 'chapters', chapters: [chapterId], force: true })
}

export async function getPageData(bookId: string, pageNum: number): Promise<PageData> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) throw new Error('页面不存在')
  const detail = await getInsightPage(page.pageId)
  if (!detail.analysis) {
    return {
      analysis: { page_num: pageNum, analyzed: false },
      sourceUrl: detail.sourceUrl,
    }
  }
  return {
    analysis: {
      ...(detail.analysis as PageAnalysisData),
      page_num: pageNum,
      analyzed: detail.analysisState === 'ready' || detail.analysisState === 'stale',
      analyzed_at: detail.generatedAt ?? undefined,
    },
    sourceUrl: detail.sourceUrl,
  }
}

export function getThumbnailUrl(bookId: string, pageNum: number): string {
  const page = cachedPageForNumber(bookId, pageNum)
  if (page?.thumbnailUrl) return page.thumbnailUrl
  return cachedTimelineThumbnailBookId === bookId
    ? timelineThumbnailCache.get(pageNum) ?? ''
    : ''
}

export async function getInsightChapters(bookId: string): Promise<InsightChapter[]> {
  const chapters = await listInsightChapters(bookId)
  let offset = 0
  return chapters.items.map(chapter => {
    const startPage = offset + 1
    offset += chapter.pageCount
    return {
      id: chapter.chapterId,
      title: chapter.title,
      start_page: startPage,
      end_page: offset,
      analyzed: chapter.analysisCounts.ready + chapter.analysisCounts.stale === chapter.pageCount,
    }
  })
}

function artifactContent(payload: Record<string, unknown>): string {
  if (typeof payload.content === 'string') return payload.content
  return JSON.stringify(payload, null, 2)
}

function isNotFound(error: unknown): boolean {
  return error instanceof ApiClientError && error.status === 404
}

export async function getOverview(
  bookId: string,
  templateType = 'story_summary'
): Promise<string | null> {
  try {
    const artifact = await getInsightOverview(bookId, templateType)
    return artifactContent(artifact.payload)
  } catch (error) {
    if (isNotFound(error)) return null
    throw error
  }
}

export async function regenerateOverview(
  bookId: string,
  templateType: string,
  force = false
): Promise<OverviewGenerationResult> {
  if (!force) {
    const cached = await getOverview(bookId, templateType)
    if (cached !== null) return { kind: 'cached', content: cached }
  }
  const accepted = await rebuildInsightOverview(bookId, templateType)
  return { kind: 'queued', jobId: accepted.jobIds[0] }
}

export async function getGeneratedTemplates(bookId: string): Promise<OverviewTemplateType[]> {
  const response = await listInsightOverviewTemplates(bookId)
  const available = new Set(response.items)
  return OVERVIEW_TEMPLATES.filter(template => available.has(template))
}

export async function getRecentAnalyzedPages(bookId: string): Promise<Array<{
  page_num: number
  summary?: string
  analyzed_at?: string
}>> {
  const response = await listRecentInsightPageAnalyses(bookId, 5)
  return response.items.map(item => ({
    page_num: item.displayPageNumber,
    ...(item.summary ? { summary: item.summary } : {}),
    ...(item.generatedAt ? { analyzed_at: item.generatedAt } : {}),
  }))
}

export async function getTimeline(
  bookId: string,
  options: { eventCursor?: number; characterCursor?: string } = {}
): Promise<TimelineData | null> {
  try {
    const timeline = await getInsightTimeline(bookId, options)
    rememberTimelineThumbnails(bookId, timeline.pageThumbnails ?? {})
    const rawEvents = Array.isArray(timeline.events)
      ? timeline.events.filter(value => Boolean(value) && typeof value === 'object')
      : []
    const groups = rawEvents.map((event, index) => {
      const pageNumbers = (Array.isArray(event.page_numbers) ? event.page_numbers : [])
        .map(Number)
        .filter(value => Number.isInteger(value) && value > 0)
      const fallbackPage = Math.max(1, index + 1)
      const start = pageNumbers.length ? Math.min(...pageNumbers) : fallbackPage
      const end = pageNumbers.length ? Math.max(...pageNumbers) : start
      const summary = typeof event.summary === 'string' ? event.summary : ''
      return {
        id: String(event.eventId),
        page_range: { start, end },
        thumbnail_page: start,
        summary,
        events: summary ? [summary] : [],
      }
    })
    const rawCharacters = Array.isArray(timeline.characters)
      ? timeline.characters.filter(value => Boolean(value) && typeof value === 'object')
      : []
    const characters = rawCharacters.map((character, index) => {
      const keyMoments = Array.isArray(character.key_moments)
        ? character.key_moments.filter(
            (value): value is Record<string, unknown> => Boolean(value) && typeof value === 'object'
          )
        : []
      const firstAppearance = Math.max(1, Number(character.first_page ?? 1))
      return {
        name: String(character.name ?? `角色 ${index + 1}`),
        description:
          typeof character.description === 'string'
            ? character.description
            : String(keyMoments[0]?.summary ?? `首次出现于第 ${firstAppearance} 页`),
        first_appearance: firstAppearance,
        key_moments: keyMoments.map(moment => ({
          summary: String(moment.summary ?? ''),
          ...(Number(moment.page) > 0 ? { page: Number(moment.page) } : {}),
        })),
      }
    })
    const content = (
      timeline.content && typeof timeline.content === 'object' ? timeline.content : {}
    ) as Record<string, unknown>
    const lastPage = groups.reduce((maximum, group) => Math.max(maximum, group.page_range.end), 0)
    return {
      ...content,
      mode: timeline.mode,
      events: rawEvents,
      groups,
      story_summary: typeof content.story_summary === 'string' ? content.story_summary : '',
      main_characters: characters,
      stats: {
        total_events: timeline.eventPage.totalCount ?? groups.length,
        total_pages: timeline.pageCount ?? lastPage,
        total_characters: timeline.characterPage.totalCount ?? characters.length,
      },
      next_event_cursor: timeline.eventPage.nextCursor,
      next_character_cursor: timeline.characterPage.nextCursor,
    } as TimelineData
  } catch (error) {
    if (isNotFound(error)) return null
    throw error
  }
}

export async function regenerateTimeline(bookId: string): Promise<string> {
  const accepted = await rebuildInsightTimeline(bookId)
  return accepted.jobIds[0]
}

export async function sendChat(
  bookId: string,
  question: string,
  options: {
    use_parent_child?: boolean
    use_reasoning?: boolean
    use_reranker?: boolean
    top_k?: number
    threshold?: number
    use_global_context?: boolean
    on_chunk?: (content: string) => void
  } = {}
): Promise<ChatResult> {
  assertBackendActionAllowed()
  const response = await fetch(insightQaUrl(bookId), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      mode: options.use_global_context ? 'global' : 'exact',
      useParentChild: options.use_parent_child,
      useReasoning: options.use_reasoning,
      useReranker: options.use_reranker,
      topK: options.top_k,
      threshold: options.threshold,
    }),
  })
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, `HTTP ${response.status}`))
  }
  let answer = ''
  let mode = options.use_global_context ? 'global' : 'precise'
  let citations: Array<{ page: number }> = []
  let suggestedQuestions: string[] = []
  let streamError = ''
  await readSseStream<Record<string, unknown>>(response, {
    missingBodyMessage: '无法读取问答响应流',
    parseErrorMessage: '问答响应格式无效',
    onMessage(message) {
      if (message.event === 'chunk') {
        answer += String(message.data.text ?? '')
        options.on_chunk?.(answer)
      } else if (message.event === 'context') {
        mode = String(message.data.mode ?? mode)
        const values = Array.isArray(message.data.citations) ? message.data.citations : []
        citations = values.map(value => {
          const citation = value as Record<string, unknown>
          return {
            page: Number(citation.pageNumber ?? 0),
          }
        })
      } else if (message.event === 'done') {
        suggestedQuestions = Array.isArray(message.data.suggestedQuestions)
          ? message.data.suggestedQuestions.map(String)
          : []
      } else if (message.event === 'error') {
        streamError = String(message.data.message ?? '问答失败')
      }
    },
  })
  if (streamError) throw new Error(streamError)
  return { answer, mode, citations, suggestedQuestions }
}

export async function rebuildEmbeddings(bookId: string): Promise<string> {
  const accepted = await rebuildInsightVectors(bookId)
  return accepted.jobIds[0]
}

export function getQAStatus(
  bookId: string,
  mode: 'precise' | 'global' = 'precise'
): Promise<QAStatusResponse> {
  return getInsightQaStatus(bookId, mode === 'global' ? 'global' : 'exact')
}

export async function rebuildCompressedContext(bookId: string): Promise<string> {
  const accepted = await rebuildInsightCompressedContext(bookId)
  return accepted.jobIds[0]
}

function noteMetadata(note: NoteData): Record<string, unknown> {
  const text =
    [note.comment, note.question, note.answer, note.content, note.title]
      .find(value => typeof value === 'string' && value.trim())
      ?.trim() || '笔记'
  return {
    text,
    question: note.question ?? '',
    answer: note.answer ?? '',
    comment: note.comment ?? '',
  }
}

function mapNote(note: V2InsightNote): NoteData {
  const metadata = note.comments?.find(value => typeof value === 'object') as
    | Record<string, unknown>
    | undefined
  const mapped: NoteData = {
    id: note.noteId,
    type: note.kind,
    content: note.content ?? note.excerpt ?? '',
    title: note.title,
    tags: note.tags,
    question: typeof metadata?.question === 'string' ? metadata.question : undefined,
    answer: typeof metadata?.answer === 'string' ? metadata.answer : undefined,
    comment: typeof metadata?.comment === 'string' ? metadata.comment : undefined,
    citations: note.citations.map(citation => ({
      page: citation.pageNumberSnapshot,
      content: citation.excerpt,
    })),
    pageNum: note.citations[0]?.pageNumberSnapshot,
    revision: note.revision,
    createdAt: note.createdAt,
    updatedAt: note.updatedAt,
  }
  noteCache.delete(mapped.id)
  noteCache.set(mapped.id, mapped)
  noteCitationPageIds.set(
    mapped.id,
    new Map(
      note.citations.flatMap(citation => {
        const pageId = citation.pageId ?? citation.pageIdSnapshot
        return pageId ? [[citation.pageNumberSnapshot, pageId] as const] : []
      })
    )
  )
  while (noteCache.size > NOTE_CACHE_LIMIT) {
    const oldestId = noteCache.keys().next().value
    if (!oldestId) break
    noteCache.delete(oldestId)
    noteCitationPageIds.delete(oldestId)
  }
  return mapped
}

export async function getNotes(
  bookId: string,
  type?: 'text' | 'qa',
  cursor?: string
): Promise<{ items: NoteData[]; nextCursor: string | null }> {
  const response = await listInsightNotes(bookId, { cursor, kind: type, limit: 50 })
  return {
    items: response.items.map(mapNote),
    nextCursor: response.nextCursor,
  }
}

export async function getNoteDetail(noteId: string): Promise<NoteData> {
  return mapNote(await getInsightNote(noteId))
}

async function resolvePageCitations(
  bookId: string,
  citations: Array<{ page: number; content: string }>
): Promise<Array<{ pageId: string; excerpt: string }>> {
  const resolved = await boundedMap(citations, async citation => {
    const page = await pageForNumber(bookId, citation.page)
    return page ? { pageId: page.pageId, excerpt: citation.content } : null
  })
  return resolved.filter(
    (value): value is { pageId: string; excerpt: string } => value !== null
  )
}

export async function createNote(
  bookId: string,
  note: {
    type: 'text' | 'qa'
    content: string
    pageNum?: number
    title?: string
    tags?: string[]
    question?: string
    answer?: string
    citations?: Array<{ page: number; content: string }>
    comment?: string
  }
): Promise<NoteData> {
  const citations = await resolvePageCitations(
    bookId,
    note.citations ?? (note.pageNum ? [{ page: note.pageNum, content: '' }] : [])
  )
  const created = await createInsightNote({
    bookId,
    title: note.title?.trim() || (note.type === 'qa' ? note.question?.trim() : '') || '未命名笔记',
    content: note.content,
    kind: note.type,
    tags: note.tags ?? [],
    citations,
    comments: [
      noteMetadata({
        ...note,
        id: '',
        createdAt: '',
        updatedAt: '',
      }),
    ],
  })
  return mapNote(created)
}

export async function updateNote(
  bookId: string,
  noteId: string,
  updates: Partial<NoteData>
): Promise<NoteData> {
  const current = noteCache.get(noteId)
  if (!current?.revision) throw new Error('笔记版本缺失，请重新加载')
  const merged = { ...current, ...updates }
  const knownPageIds = noteCitationPageIds.get(noteId) ?? new Map<number, string>()
  const requestedCitations = merged.citations ?? []
  const unresolved = requestedCitations.filter(value => !knownPageIds.has(value.page))
  await boundedMap(unresolved, async value => {
    const page = await pageForNumber(bookId, value.page)
    if (page) knownPageIds.set(value.page, page.pageId)
  })
  const citations = requestedCitations.flatMap(value => {
    const pageId = knownPageIds.get(value.page)
    return pageId ? [{ pageId, excerpt: value.content }] : []
  })
  const updated = await updateInsightNote(noteId, {
    baseRevision: current.revision,
    title: merged.title?.trim() || '未命名笔记',
    content: merged.content,
    kind: merged.type,
    tags: merged.tags ?? [],
    citations,
    comments: [noteMetadata(merged)],
  })
  return mapNote(updated)
}

export async function deleteNote(noteId: string): Promise<void> {
  const current = noteCache.get(noteId)
  if (!current?.revision) throw new Error('笔记版本缺失，请重新加载')
  await deleteInsightNote(noteId, current.revision)
  noteCache.delete(noteId)
  noteCitationPageIds.delete(noteId)
}

function providerWire(
  row: V2ProviderSettingEntry | undefined,
  provider: string
): Record<string, unknown> {
  const payload = row?.payload ?? {}
  return {
    provider,
    api_key: '',
    model: payload.modelName ?? '',
    base_url: payload.customBaseUrl ?? '',
    openai_options: payload.openaiOptions ?? {},
    image_max_size: payload.imageMaxSize,
    rpm_limit: payload.rpmLimit,
    top_k: payload.topK,
    transport_retries: payload.transportRetries,
    business_retries: payload.businessRetries,
    timeout_seconds: payload.timeoutSeconds,
  }
}

function providerSettingsWire(document: V2SettingsDocument): AnalysisConfig['provider_settings'] {
  return Object.fromEntries(
    Object.entries(PROVIDER_GROUPS).map(([group, domain]) => [
      group,
      Object.fromEntries(
        document.providerSettings
          .filter(row => row.domain === domain)
          .map(row => [row.provider, providerWire(row, row.provider)])
      ),
    ])
  )
}

function requireInsightAppPayload(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('后端 Insight 设置格式无效')
  }
  const payload = value as Record<string, unknown>
  const required = ['analysis', 'vlm', 'chat', 'embedding', 'reranker', 'imageGen']
  if (
    Object.keys(payload).length !== required.length ||
    required.some(key => !Object.prototype.hasOwnProperty.call(payload, key))
  ) {
    throw new Error('后端 Insight 设置字段不完整')
  }
  const analysis = payload.analysis
  if (!analysis || typeof analysis !== 'object' || Array.isArray(analysis)) {
    throw new Error('后端 Insight 分析设置格式无效')
  }
  const batch = (analysis as Record<string, unknown>).batch
  if (!batch || typeof batch !== 'object' || Array.isArray(batch)) {
    throw new Error('后端 Insight 批量设置格式无效')
  }
  return payload
}

export function hasInsightCredential(domain: string, provider: string): boolean {
  return credentialSummaries.some(
    row => row.domain === domain && row.provider === provider && row.hasKey
  )
}

export async function getGlobalConfig(): Promise<AnalysisConfig> {
  const [document, prompts] = await Promise.all([getV2Settings(INSIGHT_DOMAINS), listV2Prompts()])
  settingsDocument = document
  credentialSummaries = document.credentials
  promptCache = prompts
  const appEntry = document.settings.find(row => row.domain === 'insight')
  if (!appEntry) throw new Error('后端 Insight 设置缺失')
  const app = requireInsightAppPayload(appEntry.payload)
  const section = (
    key: keyof typeof SECTION_DOMAINS,
    appKey: string = key
  ): Record<string, unknown> => {
    const selected = (app[appKey] as Record<string, unknown> | undefined) ?? {}
    const provider = String(selected.provider ?? '')
    const row = document.providerSettings.find(
      value => value.domain === SECTION_DOMAINS[key] && value.provider === provider
    )
    return { ...providerWire(row, provider), ...selected, api_key: '' }
  }
  const factoryPrompts = Object.fromEntries(
    prompts.filter(prompt => prompt.isFactoryDefault).map(prompt => [prompt.type, prompt.content])
  )
  const batch = ((app.analysis as Record<string, unknown> | undefined)?.batch ?? {}) as Record<
    string,
    unknown
  >
  return {
    vlm: section('vlm') as unknown as VlmConfig,
    chat_llm: {
      ...section('chat_llm', 'chat'),
      use_same_as_vlm: Boolean((app.chat as Record<string, unknown> | undefined)?.useSameAsVlm),
    } as unknown as LlmConfig,
    embedding: section('embedding') as unknown as EmbeddingConfig,
    reranker: section('reranker') as unknown as RerankerConfig,
    image_gen: section('image_gen', 'imageGen') as unknown as ImageGenConfig,
    analysis: {
      batch: {
        pages_per_batch: Number(batch.pagesPerBatch ?? 5),
        context_batch_count: Number(batch.contextBatchCount ?? 3),
        architecture_preset: String(batch.architecturePreset ?? 'standard'),
        custom_layers: ((batch.customLayers ?? []) as Array<Record<string, unknown>>).map(
          layer => ({
            name: String(layer.name ?? ''),
            units_per_group: Number(layer.unitsPerGroup ?? 0),
            align_to_chapter: Boolean(layer.alignToChapter),
          })
        ),
      },
    },
    prompts: factoryPrompts,
    provider_settings: providerSettingsWire(document),
  }
}

function providerPayload(section: Record<string, unknown>): Record<string, unknown> {
  return {
    modelName: String(section.model ?? ''),
    customBaseUrl: String(section.base_url ?? ''),
    ...(section.openai_options && typeof section.openai_options === 'object'
      ? { openaiOptions: section.openai_options }
      : {}),
    imageMaxSize: section.image_max_size,
    rpmLimit: section.rpm_limit,
    topK: section.top_k,
    transportRetries: section.transport_retries,
    businessRetries: section.business_retries,
    timeoutSeconds: section.timeout_seconds,
  }
}

export async function saveGlobalConfig(config: AnalysisConfig): Promise<void> {
  const document = settingsDocument ?? (await getV2Settings(INSIGHT_DOMAINS))
  settingsDocument = document
  credentialSummaries = document.credentials
  const currentApp = document.settings.find(row => row.domain === 'insight')
  if (!currentApp) throw new Error('后端 Insight 设置缺失')
  const providerSettings: V2ProviderSettingMutation[] = []
  const credentialEdits: V2CredentialEdit[] = []
  const sections = [
    ['vlm', 'insight_vlm', config.vlm],
    ['chat', 'insight_chat', config.chat_llm],
    ['embedding', 'insight_embedding', config.embedding],
    ['reranker', 'insight_reranker', config.reranker],
    ['imageGen', 'insight_image_gen', config.image_gen],
  ] as const
  const appPayload: Record<string, unknown> = {
    analysis: {
      batch: {
        pagesPerBatch: config.analysis?.batch?.pages_per_batch ?? 5,
        contextBatchCount: config.analysis?.batch?.context_batch_count ?? 3,
        architecturePreset: config.analysis?.batch?.architecture_preset ?? 'standard',
        customLayers: (config.analysis?.batch?.custom_layers ?? []).map(layer => ({
          name: layer.name,
          unitsPerGroup: layer.units_per_group,
          alignToChapter: layer.align_to_chapter,
        })),
      },
    },
  }
  for (const [appKey, domain, rawSection] of sections) {
    const section = (rawSection ?? {}) as Record<string, unknown>
    const provider = String(section.provider ?? '')
    appPayload[appKey] = {
      provider,
      ...(appKey === 'chat' ? { useSameAsVlm: Boolean(section.use_same_as_vlm) } : {}),
    }
    if (!provider) continue
    const existingRow = document.providerSettings.find(
      row => row.domain === domain && row.provider === provider
    )
    const existingCredential = document.credentials.find(
      row => row.domain === domain && row.provider === provider
    )
    const mutation: V2ProviderSettingMutation = {
      domain,
      provider,
      payload: providerPayload(section),
      baseRevision: existingRow?.revision ?? 0,
      schemaVersion: 1,
      ...(existingRow?.credentialVersionId
        ? { credentialVersionId: existingRow.credentialVersionId }
        : {}),
    }
    const secret = String(section.api_key ?? '').trim()
    if (secret) {
      const clientRef = `insight:${domain}:${provider}`
      credentialEdits.push({
        domain,
        provider,
        secret: { api_key: secret },
        baseRevision: existingCredential?.revision ?? 0,
        credentialId: existingCredential?.credentialId,
        clientRef,
      })
      mutation.credentialEditRef = clientRef
      delete mutation.credentialVersionId
    }
    providerSettings.push(mutation)
  }
  const prompts = config.prompts ?? {}
  if (Object.keys(prompts).length > 0) {
    const currentPrompts = promptCache.length ? promptCache : await listV2Prompts()
    for (const [type, content] of Object.entries(prompts)) {
      const factory = currentPrompts.find(prompt => prompt.type === type && prompt.isFactoryDefault)
      if (factory && factory.content !== content) {
        const updated = await updateV2Prompt({ ...factory, content })
        promptCache = promptCache.map(prompt => (prompt.id === updated.id ? updated : prompt))
      }
    }
  }
  await saveV2SettingsTransaction({
    settings: [
      {
        domain: 'insight',
        payload: appPayload,
        baseRevision: currentApp.revision,
        schemaVersion: 1,
      },
    ],
    providerSettings,
    credentialEdits,
  })
  await getGlobalConfig()
}

function diagnosticRequest(
  kind: string,
  domain: string,
  config: { provider: string; api_key: string; model: string; base_url?: string }
) {
  return runV2ConnectionTest(kind, {
    provider: config.provider,
    model: config.model,
    baseUrl: config.base_url,
    ...(config.api_key ? { secret: { api_key: config.api_key } } : { domain }),
  })
}

export function testVlmConnection(config: VlmConfig): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('vlm', 'insight_vlm', config)
}

export function testEmbeddingConnection(config: EmbeddingConfig): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('embedding', 'insight_embedding', config)
}

export function testRerankerConnection(config: RerankerConfig): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('reranker', 'insight_reranker', config)
}

export function testLlmConnection(
  config: LlmConfig & { provider: string; api_key: string; model: string }
): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('llm', 'insight_chat', config)
}

export function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string,
  domain = 'insight_chat'
): Promise<FetchModelsResponse> {
  return fetchV2ModelCatalog({
    provider,
    baseUrl,
    ...(apiKey ? { secret: { api_key: apiKey } } : { domain }),
  })
}

export async function getDefaultPrompts(): Promise<Record<PromptType, string>> {
  const prompts = promptCache.length > 0 ? promptCache : await listV2Prompts()
  promptCache = prompts
  return Object.fromEntries(
    prompts.filter(prompt => prompt.isFactoryDefault).map(prompt => [prompt.type, prompt.content])
  ) as Record<PromptType, string>
}

function savedPrompt(prompt: V2Prompt): SavedPromptItem {
  return {
    id: prompt.id,
    name: prompt.name,
    type: prompt.type as PromptType,
    content: prompt.content,
    created_at: '',
  }
}

export async function getPromptsLibrary(): Promise<SavedPromptItem[]> {
  const prompts = promptCache.length > 0 ? promptCache : await listV2Prompts()
  promptCache = prompts
  return prompts.filter(prompt => !prompt.isFactoryDefault).map(savedPrompt)
}

export async function savePromptToLibrary(prompt: SavedPromptItem): Promise<SavedPromptItem> {
  const existing = promptCache.find(value => value.id === prompt.id)
  const saved = existing
    ? await updateV2Prompt({ ...existing, name: prompt.name, content: prompt.content })
    : await createV2Prompt(prompt.type, prompt.name, prompt.content)
  promptCache = [...promptCache.filter(value => value.id !== saved.id), saved]
  return savedPrompt(saved)
}

export async function deletePromptFromLibrary(promptId: string): Promise<void> {
  await deleteV2Prompt(promptId)
  promptCache = promptCache.filter(prompt => prompt.id !== promptId)
}

export async function importPromptsLibrary(
  library: SavedPromptItem[]
): Promise<SavedPromptItem[]> {
  const current = await listV2Prompts()
  promptCache = current
  for (const prompt of library) {
    const existing = current.find(
      value => !value.isFactoryDefault && value.type === prompt.type && value.name === prompt.name
    )
    if (existing) {
      await updateV2Prompt({ ...existing, content: prompt.content })
    } else {
      await createV2Prompt(prompt.type, prompt.name, prompt.content)
    }
  }
  promptCache = await listV2Prompts()
  return promptCache
    .filter(prompt => !prompt.isFactoryDefault)
    .map(savedPrompt)
}

export async function exportAnalysis(bookId: string): Promise<string> {
  const accepted = await createInsightExport(bookId)
  return accepted.jobIds[0]
}

export async function downloadPageAnalysis(bookId: string, pageNum: number): Promise<Blob> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) throw new Error('页面不存在')
  const response = await fetch(insightPageExportUrl(page.pageId, 'markdown'))
  if (!response.ok) throw new Error(`导出失败: HTTP ${response.status}`)
  return response.blob()
}

export async function downloadCurrentOverview(bookId: string, template: string): Promise<Blob> {
  const response = await fetch(insightCurrentExportUrl(bookId, template, 'markdown'))
  if (!response.ok) throw new Error(`导出失败: HTTP ${response.status}`)
  return response.blob()
}
