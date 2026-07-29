import type { ApiResponse, FetchModelsResponse } from '@/types'
import type {
  InsightOverviewResponse,
  InsightStatusResponse,
  InsightTimelineResponse,
} from '@/types/insight'
import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'
import { readSseStream } from '@/api/sse'
import { jobsApi } from '@/api/v2/jobs'
import {
  createInsightAnalysisJob,
  createInsightExport,
  createInsightNote,
  deleteInsightNote,
  getInsightBootstrap,
  getInsightJob,
  getInsightOverview,
  getInsightPage,
  getInsightTimeline,
  insightCurrentExportUrl,
  insightPageExportUrl,
  insightQaUrl,
  listAllInsightNotes,
  listAllInsightPages,
  listInsightChapters,
  rebuildInsightOverview,
  rebuildInsightTimeline,
  rebuildInsightVectors,
  updateInsightNote,
  type V2InsightNote,
  type V2InsightPageSummary,
} from '@/api/v2/insight'
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
} from '@/api/v2/settings'

export type { InsightOverviewResponse, InsightTimelineResponse }

export interface PageDialogueData {
  speaker_name?: string
  character?: string
  text?: string
  translated_text?: string
}

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
  page_number?: number
  page_summary?: string
  warnings?: Array<{ code: string; message: string }>
}

export interface PageDataResponse {
  success: boolean
  page?: PageAnalysisData
  analysis?: PageAnalysisData
  source_url?: string
  error?: string
}

export interface InsightPagesResponse {
  success: boolean
  pages?: number[]
  error?: string
}

export interface InsightChapterListResponse {
  success: boolean
  chapters?: Array<{
    analyzed?: boolean
    end_page: number
    id: string
    start_page: number
    title: string
  }>
  error?: string
}

export interface GeneratedTemplatesResponse {
  success: boolean
  templates?: Record<string, unknown> | string[]
  generated?: string[]
  generated_details?: Array<{ template_key: string; template_name?: string }>
  error?: string
}

export interface NoteData {
  answer?: string
  citations?: Array<{ content: string; page: number }>
  comment?: string
  content: string
  created_at: string
  id: string
  page_num?: number
  question?: string
  revision?: number
  tags?: string[]
  title?: string
  type: 'text' | 'qa'
  updated_at: string
}

export interface NoteListResponse {
  success: boolean
  notes?: NoteData[]
  error?: string
}

export interface NoteDetailResponse {
  success: boolean
  note?: NoteData
  error?: string
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

export interface ConnectionTestResponse {
  success: boolean
  error?: string
  message?: string
}

export interface StartAnalysisOptions {
  mode?: 'full' | 'incremental' | 'chapters' | 'pages'
  chapters?: string[]
  pages?: number[]
  force?: boolean
}

export interface StartAnalysisResponse {
  success: boolean
  task_id?: string
  run_id?: string
  error?: string
  message?: string
}

export type ReanalyzeResponse = StartAnalysisResponse

export interface ExportAnalysisResponse {
  success: boolean
  task_id?: string
  markdown?: string
  message?: string
  error?: string
}

export interface PreviewAnalysisResponse {
  success: boolean
  preview?: unknown
  persisted?: boolean
  message?: string
  error?: string
}

export interface OverviewContentResponse {
  success: boolean
  content?: string
  cached?: boolean
  task_id?: string
  error?: string
  message?: string
}

export interface ChatResponse {
  success: boolean
  answer?: string
  mode?: string
  citations?: Array<{ page: number }>
  suggested_questions?: string[]
  error?: string
}

export interface RebuildEmbeddingsResponse {
  success: boolean
  task_id?: string
  status?: string
  message?: string
  error?: string
}

export interface RebuildEmbeddingsStatusResponse {
  success: boolean
  task?: {
    task_id: string
    task_type: string
    status: string
    progress?: {
      current_phase?: string
      analyzed_pages?: number
      total_pages?: number
      percentage?: number
    }
    error_message?: string
  } | null
  stats?: {
    available?: boolean
    pages_count?: number
    events_count?: number
  }
  error?: string
}

export interface GlobalConfigResponse {
  success: boolean
  config?: AnalysisConfig
  error?: string
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

export interface PromptsLibraryResponse {
  success: boolean
  library?: SavedPromptItem[]
  error?: string
}

export interface DefaultPromptsResponse {
  success: boolean
  prompts?: Record<PromptType, string>
  error?: string
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
]

let settingsDocument: V2SettingsDocument | null = null
let promptCache: V2Prompt[] = []
let credentialSummaries: V2CredentialSummary[] = []
const pageCache = new Map<string, V2InsightPageSummary[]>()
const pageSourceUrls = new Map<string, string>()
const noteCache = new Map<string, NoteData>()
const noteCitationPageIds = new Map<string, Map<number, string>>()

function pageCacheKey(bookId: string, pageNum: number): string {
  return `${bookId}\u0000${pageNum}`
}

async function pagesForBook(bookId: string, force = false): Promise<V2InsightPageSummary[]> {
  if (!force && pageCache.has(bookId)) return pageCache.get(bookId)!
  const pages = await listAllInsightPages(bookId)
  pageCache.set(bookId, pages)
  return pages
}

async function pageForNumber(bookId: string, pageNum: number): Promise<V2InsightPageSummary | undefined> {
  return (await pagesForBook(bookId)).find(page => page.displayPageNumber === pageNum)
}

function mapJobStatus(status: string): string {
  if (status === 'queued') return 'pending'
  if (status === 'pausing' || status === 'cancelling') return 'running'
  if (status === 'completed_with_errors') return 'completed'
  if (status === 'interrupted') return 'failed'
  return status
}

export async function startAnalysis(
  bookId: string,
  options: StartAnalysisOptions = {},
): Promise<StartAnalysisResponse> {
  const mode = options.mode ?? 'full'
  const scope = mode === 'chapters' ? 'chapter' : mode === 'pages' ? 'page' : mode
  let pageIds: string[] | undefined
  if (scope === 'page') {
    const pages = await pagesForBook(bookId)
    const requested = new Set(options.pages ?? [])
    pageIds = pages
      .filter(page => requested.has(page.displayPageNumber))
      .map(page => page.pageId)
  }
  const accepted = await createInsightAnalysisJob({
    bookId,
    scope,
    ...(scope === 'chapter' ? { chapterIds: options.chapters ?? [] } : {}),
    ...(scope === 'page' ? { pageIds: pageIds ?? [] } : {}),
    force: options.force,
  })
  return {
    success: true,
    task_id: accepted.jobIds[0],
    run_id: accepted.runId,
    message: '分析任务已进入任务中心',
  }
}

export async function pauseAnalysis(_bookId: string, taskId?: string): Promise<ApiResponse> {
  if (!taskId) return { success: false, error: '未找到运行中的任务' }
  await jobsApi.pause(taskId)
  return { success: true }
}

export async function resumeAnalysis(_bookId: string, taskId?: string): Promise<ApiResponse> {
  if (!taskId) return { success: false, error: '未找到已暂停的任务' }
  await jobsApi.resume(taskId)
  return { success: true }
}

export async function cancelAnalysis(_bookId: string, taskId?: string): Promise<ApiResponse> {
  if (!taskId) return { success: false, error: '未找到任务' }
  await jobsApi.cancel(taskId)
  return { success: true }
}

export async function getAnalysisStatus(bookId: string): Promise<InsightStatusResponse> {
  const bootstrap = await getInsightBootstrap()
  const book = bootstrap.books.find(item => item.bookId === bookId)
  const job = bootstrap.activeJobs.find(item => item.bookId === bookId)
  const progress = job?.progress ?? {}
  return {
    success: true,
    book_id: bookId,
    analyzed: (book?.analyzedPageCount ?? 0) > 0,
    fully_analyzed: Boolean(book && book.pageCount > 0 && book.analyzedPageCount >= book.pageCount),
    analyzed_pages_count: book?.analyzedPageCount ?? 0,
    total_pages: book?.pageCount ?? 0,
    status: (job ? mapJobStatus(job.status) : book?.activeRun ? 'completed' : 'pending') as InsightStatusResponse['status'],
    current_task: job ? {
      task_id: job.jobId,
      book_id: bookId,
      task_type: 'full_book',
      status: mapJobStatus(job.status) as never,
      progress: {
        current_phase: String(progress.phase ?? progress.currentPhase ?? ''),
        current_page: Number(progress.current ?? progress.completed ?? 0),
        analyzed_pages: Number(progress.completed ?? progress.current ?? 0),
        total_pages: Number(progress.total ?? book?.pageCount ?? 0),
      },
      created_at: '',
    } : undefined,
  }
}

export async function previewAnalysis(): Promise<PreviewAnalysisResponse> {
  return {
    success: false,
    persisted: false,
    error: '新版架构不再提供非持久化分析预览，请创建单页分析任务',
  }
}

export function reanalyzePage(bookId: string, pageNum: number): Promise<ReanalyzeResponse> {
  return startAnalysis(bookId, { mode: 'pages', pages: [pageNum], force: true })
}

export function reanalyzeChapter(bookId: string, chapterId: string): Promise<ReanalyzeResponse> {
  return startAnalysis(bookId, { mode: 'chapters', chapters: [chapterId], force: true })
}

export async function getPageData(bookId: string, pageNum: number): Promise<PageDataResponse> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) return { success: false, error: '页面不存在' }
  const detail = await getInsightPage(page.pageId)
  pageSourceUrls.set(pageCacheKey(bookId, pageNum), detail.sourceUrl)
  if (!detail.analysis) {
    return {
      success: true,
      analysis: { page_num: pageNum, analyzed: false },
      source_url: detail.sourceUrl,
    }
  }
  return {
    success: true,
    analysis: {
      ...(detail.analysis as PageAnalysisData),
      page_num: pageNum,
      analyzed: detail.analysisState === 'ready' || detail.analysisState === 'stale',
      analyzed_at: detail.generatedAt ?? undefined,
    },
    source_url: detail.sourceUrl,
  }
}

export async function getAnalyzedPages(bookId: string): Promise<InsightPagesResponse> {
  const pages = await pagesForBook(bookId, true)
  return {
    success: true,
    pages: pages
      .filter(page => page.analysisState !== 'not_analyzed')
      .map(page => page.displayPageNumber),
  }
}

export function getPageImageUrl(bookId: string, pageNum: number): string {
  return pageSourceUrls.get(pageCacheKey(bookId, pageNum)) ?? ''
}

export function getThumbnailUrl(bookId: string, pageNum: number): string {
  return pageCache.get(bookId)?.find(page => page.displayPageNumber === pageNum)?.thumbnailUrl ?? ''
}

export async function getInsightChapters(bookId: string): Promise<InsightChapterListResponse> {
  const [chapters] = await Promise.all([
    listInsightChapters(bookId),
    pagesForBook(bookId, true),
  ])
  let offset = 0
  return {
    success: true,
    chapters: chapters.items.map(chapter => {
      const startPage = offset + 1
      offset += chapter.pageCount
      return {
        id: chapter.chapterId,
        title: chapter.title,
        start_page: startPage,
        end_page: offset,
        analyzed: chapter.analysisCounts.ready + chapter.analysisCounts.stale === chapter.pageCount,
      }
    }),
  }
}

function artifactContent(payload: Record<string, unknown>): string {
  if (typeof payload.content === 'string') return payload.content
  if (typeof payload.summary === 'string') return payload.summary
  return JSON.stringify(payload, null, 2)
}

export async function getOverviewBasic(bookId: string): Promise<InsightOverviewResponse> {
  const response = await getOverview(bookId, 'story_summary')
  return { success: response.success, content: response.content, error: response.error }
}

export async function getOverview(
  bookId: string,
  templateType = 'story_summary',
): Promise<OverviewContentResponse> {
  try {
    const artifact = await getInsightOverview(bookId, templateType)
    return {
      success: true,
      content: artifactContent(artifact.payload),
      cached: true,
    }
  } catch {
    return { success: false, error: '概览尚未生成' }
  }
}

export async function regenerateOverview(
  bookId: string,
  templateType: string,
  _force = false,
): Promise<OverviewContentResponse> {
  const accepted = await rebuildInsightOverview(bookId, templateType)
  return {
    success: true,
    task_id: accepted.jobIds[0],
    message: '概览重建已进入任务中心',
  }
}

export async function getGeneratedTemplates(bookId: string): Promise<GeneratedTemplatesResponse> {
  const results = await Promise.all(
    OVERVIEW_TEMPLATES.map(async template => {
      const response = await getOverview(bookId, template)
      return response.success ? template : null
    }),
  )
  return { success: true, generated: results.filter((value): value is string => Boolean(value)) }
}

export async function getTimeline(bookId: string): Promise<InsightTimelineResponse> {
  try {
    const timeline = await getInsightTimeline(bookId)
    return {
      success: true,
      timeline: {
        ...(timeline.content as Record<string, unknown> ?? {}),
        mode: timeline.mode,
        events: timeline.events,
        characters: timeline.characters,
      } as never,
    }
  } catch {
    return { success: false, error: '时间线尚未生成' }
  }
}

export async function regenerateTimeline(bookId: string): Promise<InsightTimelineResponse> {
  await rebuildInsightTimeline(bookId)
  return { success: true, timeline: undefined }
}

export function getChatStreamUrl(bookId: string): string {
  return insightQaUrl(bookId)
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
  } = {},
): Promise<ChatResponse> {
  const response = await fetch(insightQaUrl(bookId), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      mode: options.use_global_context ? 'global' : 'precise',
      useParentChild: options.use_parent_child,
      useReasoning: options.use_reasoning,
      useReranker: options.use_reranker,
      topK: options.top_k,
      threshold: options.threshold,
    }),
  })
  if (!response.ok) return { success: false, error: `HTTP ${response.status}` }
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
      } else if (message.event === 'context') {
        mode = String(message.data.mode ?? mode)
        const values = Array.isArray(message.data.citations) ? message.data.citations : []
        citations = values.map(value => ({
          page: Number((value as Record<string, unknown>).pageNumberSnapshot ?? 0),
        }))
      } else if (message.event === 'done') {
        suggestedQuestions = Array.isArray(message.data.suggestedQuestions)
          ? message.data.suggestedQuestions.map(String)
          : []
      } else if (message.event === 'error') {
        streamError = String(message.data.message ?? '问答失败')
      }
    },
  })
  return streamError
    ? { success: false, error: streamError }
    : { success: true, answer, mode, citations, suggested_questions: suggestedQuestions }
}

export async function rebuildEmbeddings(bookId: string): Promise<RebuildEmbeddingsResponse> {
  const accepted = await rebuildInsightVectors(bookId)
  return {
    success: true,
    task_id: accepted.jobIds[0],
    status: accepted.status,
    message: '向量重建已进入任务中心',
  }
}

export async function getRebuildEmbeddingsStatus(
  _bookId: string,
  taskId?: string,
): Promise<RebuildEmbeddingsStatusResponse> {
  if (!taskId) return { success: false, task: null, error: '缺少任务 ID' }
  const job = await getInsightJob(taskId)
  const progress = job.progress as Record<string, unknown>
  return {
    success: true,
    task: {
      task_id: job.jobId,
      task_type: job.kind,
      status: mapJobStatus(job.status),
      progress: {
        current_phase: String(progress.phase ?? progress.currentPhase ?? ''),
        analyzed_pages: Number(progress.completed ?? progress.current ?? 0),
        total_pages: Number(progress.total ?? 0),
        percentage: Number(progress.percent ?? 0),
      },
      error_message: String(
        ((job.progress as Record<string, unknown>).error as Record<string, unknown> | undefined)
          ?.message ?? '',
      ) || undefined,
    },
  }
}

function noteMetadata(note: NoteData): Record<string, unknown> {
  return {
    question: note.question ?? '',
    answer: note.answer ?? '',
    comment: note.comment ?? '',
  }
}

function mapNote(note: V2InsightNote): NoteData {
  const metadata = note.comments?.find(value => typeof value === 'object') as Record<string, unknown> | undefined
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
    page_num: note.citations[0]?.pageNumberSnapshot,
    revision: note.revision,
    created_at: note.createdAt,
    updated_at: note.updatedAt,
  }
  noteCache.set(mapped.id, mapped)
  noteCitationPageIds.set(
    mapped.id,
    new Map(
      note.citations.flatMap(citation => {
        const pageId = citation.pageId ?? citation.pageIdSnapshot
        return pageId ? [[citation.pageNumberSnapshot, pageId] as const] : []
      }),
    ),
  )
  return mapped
}

export async function getNotes(bookId: string, type?: 'text' | 'qa'): Promise<NoteListResponse> {
  const notes = await listAllInsightNotes(bookId, type)
  return { success: true, notes: notes.map(mapNote) }
}

export async function createNote(
  bookId: string,
  note: {
    type: 'text' | 'qa'
    content: string
    page_num?: number
    title?: string
    tags?: string[]
    question?: string
    answer?: string
    citations?: Array<{ page: number; content: string }>
    comment?: string
  },
): Promise<NoteDetailResponse> {
  const pages = await pagesForBook(bookId)
  const citations = (note.citations ?? (note.page_num ? [{ page: note.page_num, content: '' }] : []))
    .map(citation => {
      const page = pages.find(item => item.displayPageNumber === citation.page)
      return page ? { pageId: page.pageId, excerpt: citation.content } : null
    })
    .filter((value): value is { pageId: string; excerpt: string } => value !== null)
  const created = await createInsightNote({
    bookId,
    title: note.title?.trim() || (note.type === 'qa' ? note.question?.trim() : '') || '未命名笔记',
    content: note.content,
    kind: note.type,
    tags: note.tags ?? [],
    citations,
    comments: [noteMetadata({
      ...note,
      id: '',
      created_at: '',
      updated_at: '',
    })],
  })
  return { success: true, note: mapNote(created) }
}

export async function updateNote(
  bookId: string,
  noteId: string,
  updates: Partial<NoteData> & { page_num?: number },
): Promise<NoteDetailResponse> {
  const current = noteCache.get(noteId)
  if (!current?.revision) return { success: false, error: '笔记版本缺失，请重新加载' }
  const merged = { ...current, ...updates }
  const knownPageIds = noteCitationPageIds.get(noteId) ?? new Map<number, string>()
  const requestedCitations = merged.citations ?? []
  const requiresPageLookup = requestedCitations.some(value => !knownPageIds.has(value.page))
  if (requiresPageLookup) {
    for (const page of await pagesForBook(bookId)) {
      knownPageIds.set(page.displayPageNumber, page.pageId)
    }
  }
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
  return { success: true, note: mapNote(updated) }
}

export async function deleteNote(_bookId: string, noteId: string): Promise<ApiResponse> {
  const current = noteCache.get(noteId)
  if (!current?.revision) return { success: false, error: '笔记版本缺失，请重新加载' }
  await deleteInsightNote(noteId, current.revision)
  noteCache.delete(noteId)
  noteCitationPageIds.delete(noteId)
  return { success: true }
}

function providerWire(row: V2ProviderSettingEntry | undefined, provider: string): Record<string, unknown> {
  const payload = row?.payload ?? {}
  return {
    provider,
    api_key: '',
    model: payload.modelName ?? payload.model_name ?? '',
    base_url: payload.customBaseUrl ?? payload.custom_base_url ?? '',
    openai_options: payload.openaiOptions ?? payload.openai_options ?? {},
    image_max_size: payload.imageMaxSize ?? payload.image_max_size,
    rpm_limit: payload.rpmLimit ?? payload.rpm_limit,
    top_k: payload.topK ?? payload.top_k,
    transport_retries: payload.transportRetries ?? payload.transport_retries,
    business_retries: payload.businessRetries ?? payload.business_retries,
    timeout_seconds: payload.timeoutSeconds ?? payload.timeout_seconds,
  }
}

function providerSettingsWire(document: V2SettingsDocument): AnalysisConfig['provider_settings'] {
  return Object.fromEntries(
    Object.entries(PROVIDER_GROUPS).map(([group, domain]) => [
      group,
      Object.fromEntries(
        document.providerSettings
          .filter(row => row.domain === domain)
          .map(row => [row.provider, providerWire(row, row.provider)]),
      ),
    ]),
  )
}

export function hasInsightCredential(domain: string, provider: string): boolean {
  return credentialSummaries.some(
    row => row.domain === domain && row.provider === provider && row.hasKey,
  )
}

export async function getGlobalConfig(): Promise<GlobalConfigResponse> {
  const [document, prompts] = await Promise.all([
    getV2Settings(INSIGHT_DOMAINS),
    listV2Prompts(),
  ])
  settingsDocument = document
  credentialSummaries = document.credentials
  promptCache = prompts
  const app = document.settings.find(row => row.domain === 'insight')?.payload ?? {}
  const section = (
    key: keyof typeof SECTION_DOMAINS,
    appKey: string = key,
  ): Record<string, unknown> => {
    const selected = (app[appKey] as Record<string, unknown> | undefined) ?? {}
    const provider = String(selected.provider ?? '')
    const row = document.providerSettings.find(
      value => value.domain === SECTION_DOMAINS[key] && value.provider === provider,
    )
    return { ...providerWire(row, provider), ...selected, api_key: '' }
  }
  const factoryPrompts = Object.fromEntries(
    prompts.filter(prompt => prompt.isFactoryDefault).map(prompt => [prompt.type, prompt.content]),
  )
  const batch = ((app.analysis as Record<string, unknown> | undefined)?.batch ?? {}) as Record<string, unknown>
  return {
    success: true,
    config: {
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
          pages_per_batch: Number(batch.pagesPerBatch ?? batch.pages_per_batch ?? 5),
          context_batch_count: Number(batch.contextBatchCount ?? batch.context_batch_count ?? 3),
          architecture_preset: String(batch.architecturePreset ?? batch.architecture_preset ?? 'standard'),
          custom_layers: ((batch.customLayers ?? batch.custom_layers ?? []) as Array<Record<string, unknown>>)
            .map(layer => ({
              name: String(layer.name ?? ''),
              units_per_group: Number(layer.unitsPerGroup ?? layer.units_per_group ?? 0),
              align_to_chapter: Boolean(layer.alignToChapter ?? layer.align_to_chapter),
            })),
        },
      },
      prompts: factoryPrompts,
      provider_settings: providerSettingsWire(document),
    },
  }
}

function providerPayload(section: Record<string, unknown>): Record<string, unknown> {
  return {
    modelName: String(section.model ?? ''),
    customBaseUrl: String(section.base_url ?? ''),
    openaiOptions: section.openai_options ?? {},
    imageMaxSize: section.image_max_size,
    rpmLimit: section.rpm_limit,
    topK: section.top_k,
    transportRetries: section.transport_retries,
    businessRetries: section.business_retries,
    timeoutSeconds: section.timeout_seconds,
  }
}

export async function saveGlobalConfig(config: AnalysisConfig): Promise<ApiResponse> {
  const document = settingsDocument ?? await getV2Settings(INSIGHT_DOMAINS)
  settingsDocument = document
  credentialSummaries = document.credentials
  const currentApp = document.settings.find(row => row.domain === 'insight')
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
      row => row.domain === domain && row.provider === provider,
    )
    const existingCredential = document.credentials.find(
      row => row.domain === domain && row.provider === provider,
    )
    const mutation: V2ProviderSettingMutation = {
      domain,
      provider,
      payload: providerPayload(section),
      baseRevision: existingRow?.revision ?? 0,
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
        promptCache = promptCache.map(prompt => prompt.id === updated.id ? updated : prompt)
      }
    }
  }
  await saveV2SettingsTransaction({
    settings: [{
      domain: 'insight',
      payload: appPayload,
      baseRevision: currentApp?.revision ?? 0,
      schemaVersion: 1,
    }],
    providerSettings,
    credentialEdits,
  })
  await getGlobalConfig()
  return { success: true }
}

function diagnosticRequest(
  kind: string,
  domain: string,
  config: { provider: string; api_key: string; model: string; base_url?: string },
) {
  return runV2ConnectionTest(kind, {
    provider: config.provider,
    model: config.model,
    baseUrl: config.base_url,
    ...(config.api_key
      ? { secret: { apiKey: config.api_key } }
      : { domain }),
  })
}

export function testVlmConnection(config: VlmConfig): Promise<ConnectionTestResponse> {
  return diagnosticRequest('vlm', 'insight_vlm', config)
}

export function testEmbeddingConnection(config: EmbeddingConfig): Promise<ConnectionTestResponse> {
  return diagnosticRequest('embedding', 'insight_embedding', config)
}

export function testRerankerConnection(config: RerankerConfig): Promise<ConnectionTestResponse> {
  return diagnosticRequest('reranker', 'insight_reranker', config)
}

export function testLlmConnection(config: LlmConfig & { provider: string; api_key: string; model: string }): Promise<ConnectionTestResponse> {
  return diagnosticRequest('llm', 'insight_chat', config)
}

export function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string,
  domain = 'insight_chat',
): Promise<FetchModelsResponse> {
  return fetchV2ModelCatalog({
    provider,
    baseUrl,
    ...(apiKey ? { secret: { apiKey } } : { domain }),
  })
}

export async function getDefaultPrompts(): Promise<DefaultPromptsResponse> {
  const prompts = await listV2Prompts()
  promptCache = prompts
  return {
    success: true,
    prompts: Object.fromEntries(
      prompts
        .filter(prompt => prompt.isFactoryDefault)
        .map(prompt => [prompt.type, prompt.content]),
    ) as Record<PromptType, string>,
  }
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

export async function getPromptsLibrary(): Promise<PromptsLibraryResponse> {
  const prompts = await listV2Prompts()
  promptCache = prompts
  return {
    success: true,
    library: prompts.filter(prompt => !prompt.isFactoryDefault).map(savedPrompt),
  }
}

export async function savePromptToLibrary(prompt: SavedPromptItem): Promise<ApiResponse> {
  const existing = promptCache.find(value => value.id === prompt.id)
  const saved = existing
    ? await updateV2Prompt({ ...existing, name: prompt.name, content: prompt.content })
    : await createV2Prompt(prompt.type, prompt.name, prompt.content)
  promptCache = [...promptCache.filter(value => value.id !== saved.id), saved]
  return { success: true }
}

export async function deletePromptFromLibrary(promptId: string): Promise<ApiResponse> {
  await deleteV2Prompt(promptId)
  promptCache = promptCache.filter(prompt => prompt.id !== promptId)
  return { success: true }
}

export async function importPromptsLibrary(library: SavedPromptItem[]): Promise<ApiResponse> {
  const current = await listV2Prompts()
  promptCache = current
  for (const prompt of library) {
    const existing = current.find(value =>
      !value.isFactoryDefault && value.type === prompt.type && value.name === prompt.name,
    )
    if (existing) {
      await updateV2Prompt({ ...existing, content: prompt.content })
    } else {
      await createV2Prompt(prompt.type, prompt.name, prompt.content)
    }
  }
  promptCache = await listV2Prompts()
  return { success: true }
}

export async function exportAnalysis(bookId: string): Promise<ExportAnalysisResponse> {
  const accepted = await createInsightExport(bookId)
  return {
    success: true,
    task_id: accepted.jobIds[0],
    message: '完整分析导出已进入任务中心',
  }
}

export async function exportPageAnalysis(
  bookId: string,
  pageNum: number,
): Promise<PageDataResponse> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) return { success: false, error: '页面不存在' }
  const response = await fetch(insightPageExportUrl(page.pageId, 'json'))
  if (!response.ok) return { success: false, error: `HTTP ${response.status}` }
  return { success: true, analysis: await response.json() }
}

export async function downloadPageAnalysis(
  bookId: string,
  pageNum: number,
): Promise<Blob> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) throw new Error('页面不存在')
  const response = await fetch(insightPageExportUrl(page.pageId, 'markdown'))
  if (!response.ok) throw new Error(`导出失败: HTTP ${response.status}`)
  return response.blob()
}

export async function downloadCurrentOverview(
  bookId: string,
  template: string,
): Promise<Blob> {
  const response = await fetch(insightCurrentExportUrl(bookId, template, 'markdown'))
  if (!response.ok) throw new Error(`导出失败: HTTP ${response.status}`)
  return response.blob()
}
