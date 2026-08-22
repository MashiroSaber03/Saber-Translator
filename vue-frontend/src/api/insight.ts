import type { FetchModelsResponse } from '@/types'
import type {
  InsightAnalysisSnapshot,
  ChapterInfo,
  InsightEmbeddingProviderDraft,
  InsightImageGenProviderDraft,
  InsightLlmProviderDraft,
  InsightProviderDrafts,
  InsightRerankerProviderDraft,
  InsightSettingsSnapshot,
  InsightTaskStatus,
  InsightVlmProviderDraft,
  NoteData,
  NoteType,
  NoteUpdateInput,
  OverviewTemplateType,
  QAMode,
  TimelineData,
} from '@/types/insight'
import {
  deserializeOpenAICompatibleOptionsFromApi,
  serializeOpenAICompatibleOptionsForApi,
  type OpenAICompatibleOptionsWire,
} from '@/utils/openaiOptions'
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
  type V2InsightPageDetail,
  type V2InsightPageSummary,
} from '@/api/v2/insight'
import {
  prepareBrowserCredentialTransaction,
  restoreBrowserCredentialLeases,
} from '@/services/browserCredentials'
import { deepClone } from '@/utils/deepClone'
import { getProviderDefaultModel } from '@/config/aiProviders'
import {
  createV2Prompt,
  deleteV2Prompt,
  fetchV2ModelCatalog,
  getV2Settings,
  listV2Prompts,
  resetV2Prompt,
  runV2ConnectionTest,
  saveV2SettingsTransaction,
  updateV2Prompt,
  type V2CredentialEdit,
  type V2CredentialSummary,
  type V2Prompt,
  type V2PromptMutation,
  type V2ProviderSettingEntry,
  type V2ProviderSettingMutation,
  type V2SettingsDocument,
  type V2SettingsTransaction,
  type V2ConnectionTestResult,
} from '@/api/v2/settings'

export interface PageAnalysisData {
  analysisState: V2InsightPageDetail['analysisState']
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

export interface StartAnalysisOptions {
  mode?: 'full' | 'incremental' | 'chapters' | 'pages'
  chapters?: string[]
  pages?: number[]
}

export interface AnalysisJobSubmission {
  jobId: string
  runId: string
}

export type OverviewGenerationResult =
  | { kind: 'cached'; content: string }
  | { kind: 'queued'; jobId: string }

export interface ChatResult {
  answer: string
  mode: QAMode
  citations: Array<{ page: number }>
}

interface ChatStreamOptions {
  onChunk?: (content: string) => void
  signal?: AbortSignal
}

export type SendChatOptions = ChatStreamOptions &
  (
    | {
        mode: 'global'
      }
    | {
        mode: 'precise'
        threshold: number
        topK: number
        useParentChild: boolean
        useReasoning: boolean
        useReranker: boolean
      }
  )

function requireQaEventData(
  value: unknown,
  keys: readonly string[],
  event: string
): Record<string, unknown> {
  if (!isRecord(value) || !hasExactKeys(value, keys)) {
    throw new Error(`问答 ${event} 事件格式无效`)
  }
  return value
}

function requireQaCitations(value: unknown): Array<{ page: number }> {
  if (!Array.isArray(value)) throw new Error('问答引用格式无效')
  return value.map(raw => {
    const citation = requireQaEventData(
      raw,
      ['pageId', 'pageNumber', 'excerpt', 'score'],
      'context'
    )
    if (
      typeof citation.pageId !== 'string' ||
      !citation.pageId ||
      typeof citation.pageNumber !== 'number' ||
      !Number.isInteger(citation.pageNumber) ||
      citation.pageNumber < 1 ||
      typeof citation.excerpt !== 'string' ||
      typeof citation.score !== 'number' ||
      !Number.isFinite(citation.score)
    ) {
      throw new Error('问答引用字段无效')
    }
    return { page: citation.pageNumber }
  })
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

export const INSIGHT_PROMPT_TYPES = [
  'batch_analysis',
  'segment_summary',
  'chapter_summary',
  'qa_response',
] as const

export type PromptType = (typeof INSIGHT_PROMPT_TYPES)[number]

export function isInsightPromptType(value: unknown): value is PromptType {
  return typeof value === 'string' && INSIGHT_PROMPT_TYPES.some(type => type === value)
}

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
}

export type SavedPromptInput = Pick<SavedPromptItem, 'name' | 'type' | 'content'>

const INSIGHT_DOMAINS = [
  'insight',
  'insight_vlm',
  'insight_chat',
  'insight_embedding',
  'insight_reranker',
  'insight_image_gen',
]
const INSIGHT_PROVIDER_DOMAINS = {
  vlm: 'insight_vlm',
  llm: 'insight_chat',
  embedding: 'insight_embedding',
  reranker: 'insight_reranker',
  imageGen: 'insight_image_gen',
} as const
type InsightProviderDomain =
  (typeof INSIGHT_PROVIDER_DOMAINS)[keyof typeof INSIGHT_PROVIDER_DOMAINS]
const INSIGHT_PROVIDER_PAYLOAD_FIELDS: Record<InsightProviderDomain, readonly string[]> = {
  insight_vlm: ['modelName', 'customBaseUrl', 'openaiOptions', 'imageMaxSize'],
  insight_chat: ['modelName', 'customBaseUrl', 'openaiOptions'],
  insight_embedding: [
    'modelName',
    'customBaseUrl',
    'rpmLimit',
    'transportRetries',
    'businessRetries',
    'timeoutSeconds',
  ],
  insight_reranker: [
    'modelName',
    'customBaseUrl',
    'transportRetries',
    'businessRetries',
    'timeoutSeconds',
  ],
  insight_image_gen: [
    'modelName',
    'customBaseUrl',
    'transportRetries',
    'businessRetries',
    'timeoutSeconds',
  ],
}
const OVERVIEW_TEMPLATES = [
  'no_spoiler',
  'story_summary',
  'recap',
  'character_guide',
  'world_setting',
  'highlights',
  'reading_notes',
] as const satisfies readonly OverviewTemplateType[]

function requireOverviewTemplate(value: string): OverviewTemplateType {
  if (!OVERVIEW_TEMPLATES.includes(value as OverviewTemplateType)) {
    throw new Error('不支持的漫画概览模板')
  }
  return value as OverviewTemplateType
}

let credentialSummaries: V2CredentialSummary[] = []

async function boundedMap<T, R>(
  items: readonly T[],
  mapper: (item: T, index: number) => Promise<R>,
  concurrency = 4
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
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()))
  return results
}

export async function getInsightPagesPage(
  bookId: string,
  options: { chapterId?: string; cursor?: number; limit?: number } = {}
): Promise<{ items: V2InsightPageSummary[]; nextCursor: number | null }> {
  return listInsightPages(bookId, options)
}

async function pageForNumber(
  bookId: string,
  pageNum: number
): Promise<V2InsightPageSummary | undefined> {
  if (!Number.isInteger(pageNum) || pageNum < 1) return undefined
  const response = await getInsightPagesPage(bookId, { cursor: pageNum - 1, limit: 1 })
  return response.items.find(page => page.displayPageNumber === pageNum)
}

export async function startAnalysis(
  bookId: string,
  options: StartAnalysisOptions = {}
): Promise<AnalysisJobSubmission> {
  const mode = options.mode ?? 'full'
  const scope = mode === 'chapters' ? 'chapter' : mode === 'pages' ? 'page' : mode
  let pageIds: string[] | undefined
  if (scope === 'page') {
    const requestedPages = options.pages ?? []
    const pages = await boundedMap(requestedPages, page => pageForNumber(bookId, page))
    pageIds = pages.map((page, index) => {
      if (!page) throw new Error(`第 ${requestedPages[index]} 页不存在`)
      return page.pageId
    })
  }
  const accepted = await createInsightAnalysisJob({
    bookId,
    scope,
    ...(scope === 'chapter' ? { chapterIds: options.chapters ?? [] } : {}),
    ...(scope === 'page' ? { pageIds: pageIds ?? [] } : {}),
  })
  return {
    jobId: accepted.jobIds[0],
    runId: accepted.runId,
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

function requireAnalysisCount(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`漫画分析 ${field} 格式无效`)
  }
  return value
}

function requireAnalysisString(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value) throw new Error(`漫画分析 ${field} 格式无效`)
  return value
}

function requireInsightTaskStatus(value: unknown): InsightTaskStatus {
  if (
    value === 'queued' ||
    value === 'running' ||
    value === 'pausing' ||
    value === 'paused' ||
    value === 'cancelling' ||
    value === 'interrupted' ||
    value === 'completed' ||
    value === 'completed_with_errors' ||
    value === 'cancelled' ||
    value === 'failed'
  )
    return value
  throw new Error('漫画分析任务状态格式无效')
}

export async function getAnalysisStatus(bookId: string): Promise<InsightAnalysisSnapshot> {
  const bootstrap = await getInsightBootstrap()
  const book = bootstrap.books.find(item => item.bookId === bookId)
  if (!book) throw new Error('漫画分析书籍不存在')
  const pageCount = requireAnalysisCount(book.pageCount, '总页数')
  const analyzedPageCount = requireAnalysisCount(book.analyzedPageCount, '已分析页数')
  if (analyzedPageCount > pageCount) throw new Error('漫画分析已分析页数超过总页数')
  const job = bootstrap.activeJobs.find(
    item => item.bookId === bookId && item.kind === 'insight_analysis'
  )
  const pageProgress = job ? projectInsightPageProgress(job.progress) : undefined
  if (pageProgress) {
    requireAnalysisCount(pageProgress.current, '任务已处理页数')
    requireAnalysisCount(pageProgress.total, '任务总页数')
  }
  return {
    fullyAnalyzed: pageCount > 0 && analyzedPageCount === pageCount,
    analyzedPagesCount: analyzedPageCount,
    currentTask: job
      ? {
          jobId: requireAnalysisString(job.jobId, '任务 ID'),
          status: requireInsightTaskStatus(job.status),
          progress: {
            analyzedPages: pageProgress?.current ?? 0,
            totalPages: pageProgress?.total ?? 0,
          },
        }
      : undefined,
  }
}

export function reanalyzePage(bookId: string, pageNum: number): Promise<AnalysisJobSubmission> {
  return startAnalysis(bookId, { mode: 'pages', pages: [pageNum] })
}

export function reanalyzeChapter(
  bookId: string,
  chapterId: string
): Promise<AnalysisJobSubmission> {
  return startAnalysis(bookId, { mode: 'chapters', chapters: [chapterId] })
}

export async function getPageData(bookId: string, pageNum: number): Promise<PageData> {
  const page = await pageForNumber(bookId, pageNum)
  if (!page) throw new Error('页面不存在')
  const detail = await getInsightPage(page.pageId)
  if (!detail.analysis) {
    return {
      analysis: {
        analysisState: detail.analysisState,
        page_num: pageNum,
      },
      sourceUrl: detail.sourceUrl,
    }
  }
  return {
    analysis: {
      analysisState: detail.analysisState,
      continuity_notes: detail.analysis.continuity_notes,
      key_events: detail.analysis.key_events,
      page_summary: detail.analysis.page_summary,
      warnings: detail.analysis.warnings,
      page_num: pageNum,
    },
    sourceUrl: detail.sourceUrl,
  }
}

export async function getInsightChapters(bookId: string): Promise<ChapterInfo[]> {
  const chapters = await listInsightChapters(bookId)
  let offset = 0
  return chapters.items.map(chapter => {
    const pageCount = requireAnalysisCount(chapter.pageCount, '章节总页数')
    const analyzedCount = requireAnalysisCount(
      chapter.analysisCounts.ready + chapter.analysisCounts.stale,
      '章节已分析页数'
    )
    if (analyzedCount > pageCount) throw new Error('漫画分析章节已分析页数超过章节总页数')
    const startPage = pageCount > 0 ? offset + 1 : 0
    offset += pageCount
    requireAnalysisCount(offset, '章节累计页数')
    return {
      id: requireAnalysisString(chapter.chapterId, '章节 ID'),
      title: requireAnalysisString(chapter.title, '章节标题'),
      startPage,
      endPage: pageCount > 0 ? offset : 0,
      analyzed: pageCount > 0 && analyzedCount === pageCount,
      analyzedCount,
    }
  })
}

function artifactContent(payload: Record<string, unknown>): string {
  if (
    typeof payload.title !== 'string' ||
    !payload.title.trim() ||
    typeof payload.content !== 'string' ||
    !payload.content.trim()
  ) {
    throw new Error('漫画概览响应格式无效')
  }
  return payload.content
}

function isNotFound(error: unknown): boolean {
  return error instanceof ApiClientError && error.status === 404
}

export async function getOverview(
  bookId: string,
  templateType = 'story_summary'
): Promise<string | null> {
  try {
    const artifact = await getInsightOverview(bookId, requireOverviewTemplate(templateType))
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
  const template = requireOverviewTemplate(templateType)
  if (!force) {
    const cached = await getOverview(bookId, template)
    if (cached !== null) return { kind: 'cached', content: cached }
  }
  const accepted = await rebuildInsightOverview(bookId, template)
  return { kind: 'queued', jobId: accepted.jobIds[0] }
}

export async function getGeneratedTemplates(bookId: string): Promise<OverviewTemplateType[]> {
  const response = await listInsightOverviewTemplates(bookId)
  if (
    !Array.isArray(response.items) ||
    response.items.some(
      template => !OVERVIEW_TEMPLATES.includes(template as OverviewTemplateType)
    ) ||
    new Set(response.items).size !== response.items.length
  ) {
    throw new Error('漫画概览模板响应格式无效')
  }
  return [...response.items] as OverviewTemplateType[]
}

export async function getRecentAnalyzedPages(bookId: string): Promise<
  Array<{
    page_num: number
    summary?: string
  }>
> {
  const response = await listRecentInsightPageAnalyses(bookId, 5)
  return response.items.map(item => ({
    page_num: item.displayPageNumber,
    ...(item.summary ? { summary: item.summary } : {}),
  }))
}

function requirePositiveInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new Error(`时间线 ${field} 格式无效`)
  }
  return value
}

function requireNonnegativeInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new Error(`时间线 ${field} 格式无效`)
  }
  return value
}

function requireTimelineString(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`时间线 ${field} 格式无效`)
  }
  return value
}

function optionalTimelineString(value: unknown, field: string): string | undefined {
  if (value === undefined) return undefined
  return requireTimelineString(value, field)
}

function requireTimelinePageRange(value: unknown, field: string): { start: number; end: number } {
  if (!isRecord(value)) throw new Error(`时间线 ${field} 格式无效`)
  const start = requirePositiveInteger(value.start, `${field}.start`)
  const end = requirePositiveInteger(value.end, `${field}.end`)
  if (end < start) throw new Error(`时间线 ${field} 范围无效`)
  return { start, end }
}

function requireTimelineArcs(value: unknown): TimelineData['plot_arcs'] {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) throw new Error('时间线 plot_arcs 格式无效')
  const arcIds = new Set<string>()
  return value.map((raw, index) => {
    if (!isRecord(raw)) throw new Error(`时间线 plot_arcs[${index}] 格式无效`)
    const id = requireTimelineString(raw.id, `plot_arcs[${index}].id`)
    if (arcIds.has(id)) throw new Error(`时间线 plot_arcs[${index}].id 重复`)
    arcIds.add(id)
    const eventIds = raw.event_ids
    if (
      eventIds !== undefined &&
      (!Array.isArray(eventIds) || eventIds.some(id => typeof id !== 'string' || !id))
    ) {
      throw new Error(`时间线 plot_arcs[${index}].event_ids 格式无效`)
    }
    return {
      id,
      name: requireTimelineString(raw.name, `plot_arcs[${index}].name`),
      description: requireTimelineString(raw.description, `plot_arcs[${index}].description`),
      page_range: requireTimelinePageRange(raw.page_range, `plot_arcs[${index}].page_range`),
      ...(optionalTimelineString(raw.mood, `plot_arcs[${index}].mood`)
        ? { mood: raw.mood as string }
        : {}),
      ...(eventIds ? { event_ids: [...eventIds] as string[] } : {}),
    }
  })
}

function requireTimelineThreads(value: unknown): TimelineData['plot_threads'] {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) throw new Error('时间线 plot_threads 格式无效')
  const threadIds = new Set<string>()
  return value.map((raw, index) => {
    if (!isRecord(raw)) throw new Error(`时间线 plot_threads[${index}] 格式无效`)
    const id = requireTimelineString(raw.id, `plot_threads[${index}].id`)
    if (threadIds.has(id)) throw new Error(`时间线 plot_threads[${index}].id 重复`)
    threadIds.add(id)
    const introduced =
      raw.introduced_at === undefined
        ? undefined
        : requirePositiveInteger(raw.introduced_at, `plot_threads[${index}].introduced_at`)
    const resolved =
      raw.resolved_at === undefined || raw.resolved_at === null
        ? raw.resolved_at
        : requirePositiveInteger(raw.resolved_at, `plot_threads[${index}].resolved_at`)
    return {
      id,
      name: requireTimelineString(raw.name, `plot_threads[${index}].name`),
      type: requireTimelineString(raw.type, `plot_threads[${index}].type`),
      status: requireTimelineString(raw.status, `plot_threads[${index}].status`),
      ...(optionalTimelineString(raw.description, `plot_threads[${index}].description`)
        ? { description: raw.description as string }
        : {}),
      ...(introduced === undefined ? {} : { introduced_at: introduced }),
      ...(resolved === undefined ? {} : { resolved_at: resolved }),
    }
  })
}

function requireTimelineThumbnails(value: unknown): Record<number, string> {
  if (!isRecord(value)) throw new Error('时间线 pageThumbnails 格式无效')
  const thumbnails: Record<number, string> = {}
  for (const [pageNumber, url] of Object.entries(value)) {
    if (!/^[1-9][0-9]*$/.test(pageNumber) || typeof url !== 'string' || !url.trim()) {
      throw new Error('时间线 pageThumbnails 格式无效')
    }
    thumbnails[Number(pageNumber)] = url
  }
  return thumbnails
}

function requireTimelinePayload(value: unknown, expectedBookId: string): TimelineData {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'timelineVersionId',
      'bookId',
      'runId',
      'mode',
      'status',
      'content',
      'events',
      'characters',
      'eventPage',
      'characterPage',
      'pageCount',
      'pageThumbnails',
      'dependencyFingerprint',
    ])
  ) {
    throw new Error('时间线响应格式无效')
  }
  if (value.bookId !== expectedBookId) {
    throw new Error('时间线 bookId 与请求不一致')
  }
  const timelineVersionId = requireTimelineString(value.timelineVersionId, 'timelineVersionId')
  if (value.runId !== null) requireTimelineString(value.runId, 'runId')
  if (value.status !== 'ready' && value.status !== 'degraded' && value.status !== 'stale') {
    throw new Error('时间线 status 格式无效')
  }
  if (
    typeof value.dependencyFingerprint !== 'string' ||
    !/^[0-9a-f]{64}$/.test(value.dependencyFingerprint)
  ) {
    throw new Error('时间线 dependencyFingerprint 格式无效')
  }
  const mode = value.mode
  if (mode !== 'enhanced' && mode !== 'compressed' && mode !== 'simple') {
    throw new Error('时间线 mode 格式无效')
  }
  if (!isRecord(value.content)) throw new Error('时间线 content 格式无效')
  const content = value.content
  if (
    content.requested_mode !== 'enhanced' ||
    content.actual_mode !== mode ||
    typeof content.degraded !== 'boolean' ||
    content.degraded !== (mode !== 'enhanced') ||
    (mode === 'enhanced' && content.fallback_reason !== null) ||
    (mode !== 'enhanced' &&
      (typeof content.fallback_reason !== 'string' || !content.fallback_reason.trim())) ||
    typeof content.story_summary !== 'string' ||
    (mode !== 'simple' && !content.story_summary.trim())
  ) {
    throw new Error('时间线 content 元数据格式无效')
  }
  if (!Array.isArray(value.events)) throw new Error('时间线 events 格式无效')
  const events = value.events.map((raw, index) => {
    if (!isRecord(raw)) throw new Error(`时间线 events[${index}] 格式无效`)
    const pageIds = raw.page_ids
    const pageNumbers = raw.page_numbers
    if (
      !Array.isArray(pageIds) ||
      !pageIds.length ||
      pageIds.some(pageId => typeof pageId !== 'string' || !pageId) ||
      !Array.isArray(pageNumbers) ||
      pageNumbers.length !== pageIds.length
    ) {
      throw new Error(`时间线 events[${index}] 页面引用格式无效`)
    }
    const normalizedPageNumbers = pageNumbers.map((pageNumber, pageIndex) =>
      requirePositiveInteger(pageNumber, `events[${index}].page_numbers[${pageIndex}]`)
    )
    return {
      eventId: requireTimelineString(raw.eventId, `events[${index}].eventId`),
      summary: requireTimelineString(raw.summary, `events[${index}].summary`),
      page_ids: [...pageIds] as string[],
      page_numbers: normalizedPageNumbers,
      ...(optionalTimelineString(raw.importance, `events[${index}].importance`)
        ? { importance: raw.importance as string }
        : {}),
    }
  })
  if (!Array.isArray(value.characters)) throw new Error('时间线 characters 格式无效')
  const characters = value.characters.map((raw, index) => {
    if (!isRecord(raw) || !Array.isArray(raw.key_moments)) {
      throw new Error(`时间线 characters[${index}] 格式无效`)
    }
    const keyMoments = raw.key_moments.map((moment, momentIndex) => {
      if (!isRecord(moment)) {
        throw new Error(`时间线 characters[${index}].key_moments[${momentIndex}] 格式无效`)
      }
      const page =
        moment.page === undefined
          ? undefined
          : requirePositiveInteger(
              moment.page,
              `characters[${index}].key_moments[${momentIndex}].page`
            )
      return {
        summary: requireTimelineString(
          moment.summary,
          `characters[${index}].key_moments[${momentIndex}].summary`
        ),
        ...(page === undefined ? {} : { page }),
      }
    })
    return {
      character_id: requireTimelineString(raw.characterId, `characters[${index}].characterId`),
      name: requireTimelineString(raw.name, `characters[${index}].name`),
      description: requireTimelineString(raw.description, `characters[${index}].description`),
      first_appearance: requirePositiveInteger(raw.first_page, `characters[${index}].first_page`),
      key_moments: keyMoments,
      ...(optionalTimelineString(raw.arc, `characters[${index}].arc`)
        ? { arc: raw.arc as string }
        : {}),
    }
  })
  if (!isRecord(value.eventPage) || !isRecord(value.characterPage)) {
    throw new Error('时间线分页响应格式无效')
  }
  const nextEventCursor =
    value.eventPage.nextCursor === null
      ? null
      : requirePositiveInteger(value.eventPage.nextCursor, 'eventPage.nextCursor')
  const nextCharacterCursor = value.characterPage.nextCursor
  if (
    nextCharacterCursor !== null &&
    (typeof nextCharacterCursor !== 'string' || !nextCharacterCursor)
  ) {
    throw new Error('时间线 characterPage.nextCursor 格式无效')
  }
  const pageCount = requireNonnegativeInteger(value.pageCount, 'pageCount')
  const totalEvents = requireNonnegativeInteger(value.eventPage.totalCount, 'eventPage.totalCount')
  const totalCharacters = requireNonnegativeInteger(
    value.characterPage.totalCount,
    'characterPage.totalCount'
  )
  const pageThumbnails = requireTimelineThumbnails(value.pageThumbnails)
  const groups = events.map(event => {
    const start = Math.min(...event.page_numbers)
    const end = Math.max(...event.page_numbers)
    return {
      id: event.eventId,
      page_range: { start, end },
      thumbnail_page: start,
      summary: event.summary,
      events: [event.summary],
    }
  })
  const plotArcs = requireTimelineArcs(content.plot_arcs)
  const plotThreads = requireTimelineThreads(content.plot_threads)
  return {
    timeline_version_id: timelineVersionId,
    mode,
    events,
    groups,
    story_summary: content.story_summary,
    main_characters: characters,
    page_thumbnails: pageThumbnails,
    stats: {
      total_events: totalEvents,
      total_pages: pageCount,
      total_characters: totalCharacters,
      ...(plotArcs ? { total_arcs: plotArcs.length } : {}),
      ...(plotThreads ? { total_threads: plotThreads.length } : {}),
    },
    ...(plotArcs ? { plot_arcs: plotArcs } : {}),
    ...(plotThreads ? { plot_threads: plotThreads } : {}),
    next_event_cursor: nextEventCursor,
    next_character_cursor: nextCharacterCursor,
  }
}

export async function getTimeline(
  bookId: string,
  options: { eventCursor?: number; characterCursor?: string } = {}
): Promise<TimelineData | null> {
  try {
    const timeline = await getInsightTimeline(bookId, options)
    return requireTimelinePayload(timeline, bookId)
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
  options: SendChatOptions
): Promise<ChatResult> {
  const preciseOptions =
    options.mode === 'precise'
      ? {
          useParentChild: options.useParentChild,
          useReasoning: options.useReasoning,
          useReranker: options.useReranker,
          topK: options.topK,
          threshold: options.threshold,
        }
      : {}
  const response = await fetch(insightQaUrl(bookId), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: options.signal,
    body: JSON.stringify({
      question,
      mode: options.mode === 'global' ? 'global' : 'exact',
      ...preciseOptions,
    }),
  })
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, `HTTP ${response.status}`))
  }
  let answer = ''
  let mode: QAMode | null = null
  let citations: Array<{ page: number }> = []
  let streamError = ''
  let statusSeen = false
  let contextSeen = false
  let doneSeen = false
  await readSseStream<Record<string, unknown>>(response, {
    missingBodyMessage: '无法读取问答响应流',
    parseErrorMessage: '问答响应格式无效',
    onMessage(message) {
      if (doneSeen || streamError) throw new Error('问答响应在结束后仍包含事件')
      if (message.event === 'status') {
        const data = requireQaEventData(message.data, ['requestId', 'status'], 'status')
        if (
          statusSeen ||
          typeof data.requestId !== 'string' ||
          !data.requestId ||
          data.status !== 'retrieving'
        ) {
          throw new Error('问答 status 事件字段无效')
        }
        statusSeen = true
      } else if (message.event === 'chunk') {
        const data = requireQaEventData(message.data, ['text'], 'chunk')
        if (!statusSeen || !contextSeen || typeof data.text !== 'string' || !data.text) {
          throw new Error('问答 chunk 事件字段无效')
        }
        answer += data.text
        options.onChunk?.(answer)
      } else if (message.event === 'context') {
        const data = requireQaEventData(message.data, ['mode', 'citations'], 'context')
        if (!statusSeen || contextSeen || (data.mode !== 'exact' && data.mode !== 'global')) {
          throw new Error('问答 context 事件字段无效')
        }
        mode = data.mode === 'exact' ? 'precise' : 'global'
        citations = requireQaCitations(data.citations)
        contextSeen = true
      } else if (message.event === 'done') {
        requireQaEventData(message.data, [], 'done')
        if (!statusSeen || !contextSeen || !answer) {
          throw new Error('问答响应未完整结束')
        }
        doneSeen = true
      } else if (message.event === 'error') {
        const data = requireQaEventData(message.data, ['code', 'message'], 'error')
        if (
          !statusSeen ||
          typeof data.code !== 'string' ||
          !data.code ||
          typeof data.message !== 'string' ||
          !data.message
        ) {
          throw new Error('问答 error 事件字段无效')
        }
        streamError = data.message
      } else {
        throw new Error(`未知的问答事件: ${message.event}`)
      }
    },
  })
  if (streamError) throw new Error(streamError)
  if (!doneSeen || mode === null) throw new Error('问答响应意外中断')
  return { answer, mode, citations }
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

function mapNote(note: V2InsightNote): NoteData {
  if (
    !note ||
    typeof note.noteId !== 'string' ||
    !note.noteId ||
    (note.kind !== 'text' && note.kind !== 'qa') ||
    typeof note.title !== 'string' ||
    !note.title.trim() ||
    (note.content !== null && typeof note.content !== 'string') ||
    (note.excerpt !== null && typeof note.excerpt !== 'string') ||
    !Array.isArray(note.tags) ||
    note.tags.some(tag => typeof tag !== 'string') ||
    !Number.isInteger(note.revision) ||
    note.revision < 1 ||
    !Array.isArray(note.citations) ||
    typeof note.createdAt !== 'string' ||
    !Number.isFinite(Date.parse(note.createdAt)) ||
    typeof note.updatedAt !== 'string' ||
    !Number.isFinite(Date.parse(note.updatedAt)) ||
    (note.kind === 'qa' && (typeof note.question !== 'string' || !note.question.trim())) ||
    (note.kind === 'text' && note.question !== null) ||
    (note.comment !== null && (typeof note.comment !== 'string' || !note.comment.trim()))
  ) {
    throw new Error('笔记响应格式无效')
  }
  for (const citation of note.citations) {
    if (
      !Number.isInteger(citation.pageNumberSnapshot) ||
      citation.pageNumberSnapshot < 1 ||
      typeof citation.excerpt !== 'string'
    ) {
      throw new Error('笔记响应格式无效')
    }
  }
  const content = note.content ?? note.excerpt ?? ''
  return {
    id: note.noteId,
    type: note.kind,
    content,
    title: note.title,
    tags: note.tags,
    question: note.question ?? undefined,
    comment: note.comment ?? undefined,
    citations: note.citations.map(citation => ({
      page: citation.pageNumberSnapshot,
      content: citation.excerpt,
    })),
    pageNum: note.citations[0]?.pageNumberSnapshot,
    revision: note.revision,
    createdAt: note.createdAt,
    updatedAt: note.updatedAt,
  }
}

export async function getNotes(
  bookId: string,
  type?: NoteType,
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
  return boundedMap(citations, async citation => {
    const page = await pageForNumber(bookId, citation.page)
    if (!page) throw new Error(`引用的第 ${citation.page} 页不存在`)
    return { pageId: page.pageId, excerpt: citation.content }
  })
}

export async function createNote(
  bookId: string,
  note: {
    type: NoteType
    content: string
    pageNum?: number
    title?: string
    tags?: string[]
    question?: string
    citations?: Array<{ page: number; content: string }>
    comment?: string
  }
): Promise<NoteData> {
  const citations = await resolvePageCitations(
    bookId,
    note.citations ?? (note.pageNum === undefined ? [] : [{ page: note.pageNum, content: '' }])
  )
  const created = await createInsightNote({
    bookId,
    title: note.title?.trim() || (note.type === 'qa' ? note.question?.trim() : '') || '未命名笔记',
    content: note.content,
    kind: note.type,
    tags: note.tags ?? [],
    citations,
    question: note.type === 'qa' ? note.question?.trim() || null : null,
    comment: note.comment?.trim() || null,
  })
  return mapNote(created)
}

export async function updateNote(
  bookId: string,
  noteId: string,
  updates: NoteUpdateInput
): Promise<NoteData> {
  const currentWire = await getInsightNote(noteId)
  const current = mapNote(currentWire)
  const merged = { ...current, ...updates }
  const knownPageIds = new Map(
    currentWire.citations.flatMap(citation => {
      const pageId = citation.pageId ?? citation.pageIdSnapshot
      return pageId ? [[citation.pageNumberSnapshot, pageId] as const] : []
    })
  )
  const pageNumberWasUpdated = Object.prototype.hasOwnProperty.call(updates, 'pageNum')
  const requestedCitations =
    updates.citations ??
    (pageNumberWasUpdated
      ? updates.pageNum === undefined
        ? []
        : [{ page: updates.pageNum, content: '' }]
      : current.citations)
  const unresolved = requestedCitations.filter(value => !knownPageIds.has(value.page))
  await boundedMap(unresolved, async value => {
    const page = await pageForNumber(bookId, value.page)
    if (!page) throw new Error(`引用的第 ${value.page} 页不存在`)
    knownPageIds.set(value.page, page.pageId)
  })
  const citations = requestedCitations.map(value => {
    const pageId = knownPageIds.get(value.page)
    if (!pageId) throw new Error(`引用的第 ${value.page} 页不存在`)
    return { pageId, excerpt: value.content }
  })
  const updated = await updateInsightNote(noteId, {
    baseRevision: currentWire.revision,
    title: merged.title?.trim() || '未命名笔记',
    content: merged.content,
    kind: merged.type,
    tags: merged.tags ?? [],
    citations,
    question: merged.type === 'qa' ? merged.question?.trim() || null : null,
    comment: merged.comment?.trim() || null,
  })
  return mapNote(updated)
}

export async function deleteNote(noteId: string): Promise<void> {
  const current = await getInsightNote(noteId)
  await deleteInsightNote(noteId, current.revision)
}

function requireProviderString(
  payload: Record<string, unknown>,
  key: string,
  label: string
): string {
  const value = payload[key]
  if (typeof value !== 'string') throw new Error(`${label}.${key} 格式无效`)
  return value
}

function requireProviderInteger(
  payload: Record<string, unknown>,
  key: string,
  label: string,
  maximum?: number
): number {
  const value = payload[key]
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    (maximum !== undefined && value > maximum)
  ) {
    throw new Error(`${label}.${key} 格式无效`)
  }
  return value
}

function requireProviderNumber(
  payload: Record<string, unknown>,
  key: string,
  label: string
): number {
  const value = payload[key]
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label}.${key} 格式无效`)
  }
  return value
}

function requireOpenAiOptionsWire(
  payload: Record<string, unknown>,
  label: string
): OpenAICompatibleOptionsWire {
  const value = payload.openaiOptions
  if (!isRecord(value) || !hasExactKeys(value, ['request', 'execution'])) {
    throw new Error(`${label}.openaiOptions 格式无效`)
  }
  const request = value.request
  const execution = value.execution
  if (
    !isRecord(request) ||
    !isRecord(execution) ||
    !hasExactKeys(request, ['force_json_output', 'temperature', 'extra_body']) ||
    typeof request.force_json_output !== 'boolean' ||
    (request.temperature !== null &&
      (typeof request.temperature !== 'number' ||
        !Number.isFinite(request.temperature) ||
        request.temperature < 0 ||
        request.temperature > 2)) ||
    !isRecord(request.extra_body) ||
    !hasExactKeys(execution, [
      'use_stream',
      'rpm_limit',
      'transport_retries',
      'business_retries',
    ]) ||
    typeof execution.use_stream !== 'boolean'
  ) {
    throw new Error(`${label}.openaiOptions 字段无效`)
  }
  requireProviderInteger(execution, 'rpm_limit', `${label}.openaiOptions.execution`, 100_000)
  requireProviderInteger(execution, 'transport_retries', `${label}.openaiOptions.execution`, 100)
  requireProviderInteger(execution, 'business_retries', `${label}.openaiOptions.execution`, 100)
  return value as unknown as OpenAICompatibleOptionsWire
}

function requireProviderPayload(
  domain: InsightProviderDomain,
  row: V2ProviderSettingEntry
): Record<string, unknown> {
  if (
    !isRecord(row.payload) ||
    !hasExactKeys(row.payload, INSIGHT_PROVIDER_PAYLOAD_FIELDS[domain])
  ) {
    throw new Error(`后端 ${domain} provider 设置字段不完整`)
  }
  return row.payload
}

function providerCommonDraft(payload: Record<string, unknown>, label: string) {
  return {
    apiKey: '',
    model: requireProviderString(payload, 'modelName', label),
    baseUrl: requireProviderString(payload, 'customBaseUrl', label),
  }
}

function defaultVlmDraft(provider: string): InsightVlmProviderDraft {
  return {
    apiKey: '',
    model: getProviderDefaultModel(provider, 'vlm'),
    baseUrl: '',
    openaiOptions: deserializeOpenAICompatibleOptionsFromApi({
      request: { force_json_output: false, temperature: 0.3, extra_body: {} },
      execution: {
        use_stream: true,
        rpm_limit: 0,
        transport_retries: 1,
        business_retries: 0,
      },
    }),
    imageMaxSize: 0,
  }
}

function defaultLlmDraft(provider: string): InsightLlmProviderDraft {
  return {
    apiKey: '',
    model: getProviderDefaultModel(provider, 'chat'),
    baseUrl: '',
    openaiOptions: deserializeOpenAICompatibleOptionsFromApi({
      request: { force_json_output: false, temperature: null, extra_body: {} },
      execution: {
        use_stream: true,
        rpm_limit: 0,
        transport_retries: 1,
        business_retries: 0,
      },
    }),
  }
}

function defaultEmbeddingDraft(provider: string): InsightEmbeddingProviderDraft {
  return {
    apiKey: '',
    model: getProviderDefaultModel(provider, 'embedding'),
    baseUrl: '',
    rpmLimit: 0,
    transportRetries: 1,
    businessRetries: 0,
    timeoutSeconds: 0,
  }
}

function defaultRerankerDraft(provider: string): InsightRerankerProviderDraft {
  return {
    apiKey: '',
    model: getProviderDefaultModel(provider, 'reranker'),
    baseUrl: '',
    transportRetries: 1,
    businessRetries: 0,
    timeoutSeconds: 0,
  }
}

function defaultImageGenDraft(provider: string): InsightImageGenProviderDraft {
  return {
    apiKey: '',
    model: getProviderDefaultModel(provider, 'imageGen'),
    baseUrl: '',
    transportRetries: 1,
    businessRetries: 0,
    timeoutSeconds: 0,
  }
}

function readVlmDraft(row: V2ProviderSettingEntry): InsightVlmProviderDraft {
  const label = `${row.domain}.${row.provider}`
  const payload = requireProviderPayload('insight_vlm', row)
  return {
    ...providerCommonDraft(payload, label),
    openaiOptions: deserializeOpenAICompatibleOptionsFromApi(
      requireOpenAiOptionsWire(payload, label)
    ),
    imageMaxSize: requireProviderInteger(payload, 'imageMaxSize', label),
  }
}

function readLlmDraft(row: V2ProviderSettingEntry): InsightLlmProviderDraft {
  const label = `${row.domain}.${row.provider}`
  const payload = requireProviderPayload('insight_chat', row)
  return {
    ...providerCommonDraft(payload, label),
    openaiOptions: deserializeOpenAICompatibleOptionsFromApi(
      requireOpenAiOptionsWire(payload, label)
    ),
  }
}

function readEmbeddingDraft(row: V2ProviderSettingEntry): InsightEmbeddingProviderDraft {
  const label = `${row.domain}.${row.provider}`
  const payload = requireProviderPayload('insight_embedding', row)
  return {
    ...providerCommonDraft(payload, label),
    rpmLimit: requireProviderInteger(payload, 'rpmLimit', label),
    transportRetries: requireProviderInteger(payload, 'transportRetries', label),
    businessRetries: requireProviderInteger(payload, 'businessRetries', label),
    timeoutSeconds: requireProviderNumber(payload, 'timeoutSeconds', label),
  }
}

function readRerankerDraft(row: V2ProviderSettingEntry): InsightRerankerProviderDraft {
  const label = `${row.domain}.${row.provider}`
  const payload = requireProviderPayload('insight_reranker', row)
  return {
    ...providerCommonDraft(payload, label),
    transportRetries: requireProviderInteger(payload, 'transportRetries', label),
    businessRetries: requireProviderInteger(payload, 'businessRetries', label),
    timeoutSeconds: requireProviderNumber(payload, 'timeoutSeconds', label),
  }
}

function readImageGenDraft(row: V2ProviderSettingEntry): InsightImageGenProviderDraft {
  const label = `${row.domain}.${row.provider}`
  const payload = requireProviderPayload('insight_image_gen', row)
  return {
    ...providerCommonDraft(payload, label),
    transportRetries: requireProviderInteger(payload, 'transportRetries', label),
    businessRetries: requireProviderInteger(payload, 'businessRetries', label),
    timeoutSeconds: requireProviderNumber(payload, 'timeoutSeconds', label),
  }
}

function readProviderDrafts(document: V2SettingsDocument): InsightProviderDrafts {
  if (!Array.isArray(document.providerSettings)) {
    throw new Error('后端 Insight provider 设置格式无效')
  }
  const drafts: InsightProviderDrafts = {
    vlm: {},
    llm: {},
    embedding: {},
    reranker: {},
    imageGen: {},
  }
  for (const row of document.providerSettings) {
    if (typeof row.provider !== 'string' || !row.provider || row.provider !== row.provider.trim()) {
      throw new Error('后端 Insight provider 名称格式无效')
    }
    const provider = row.provider
    if (row.domain === 'insight_vlm') {
      if (drafts.vlm[provider]) throw new Error(`后端 insight_vlm provider 重复：${provider}`)
      drafts.vlm[provider] = readVlmDraft(row)
    } else if (row.domain === 'insight_chat') {
      if (drafts.llm[provider]) throw new Error(`后端 insight_chat provider 重复：${provider}`)
      drafts.llm[provider] = readLlmDraft(row)
    } else if (row.domain === 'insight_embedding') {
      if (drafts.embedding[provider]) {
        throw new Error(`后端 insight_embedding provider 重复：${provider}`)
      }
      drafts.embedding[provider] = readEmbeddingDraft(row)
    } else if (row.domain === 'insight_reranker') {
      if (drafts.reranker[provider]) {
        throw new Error(`后端 insight_reranker provider 重复：${provider}`)
      }
      drafts.reranker[provider] = readRerankerDraft(row)
    } else if (row.domain === 'insight_image_gen') {
      if (drafts.imageGen[provider]) {
        throw new Error(`后端 insight_image_gen provider 重复：${provider}`)
      }
      drafts.imageGen[provider] = readImageGenDraft(row)
    } else {
      throw new Error(`后端 Insight provider domain 格式无效：${row.domain}`)
    }
  }
  return drafts
}

function activeProviderDraft<T>(
  provider: string,
  drafts: Record<string, T>,
  createDefault: (provider: string) => T
): T {
  const draft = drafts[provider] ?? createDefault(provider)
  if (provider && !drafts[provider]) drafts[provider] = deepClone(draft)
  return deepClone(draft)
}

type InsightProviderSelection = {
  provider: string
}

type InsightAppPayload = {
  analysis: {
    batch: {
      pagesPerBatch: number
      contextBatchCount: number
      architecturePreset: string
      customLayers: Array<{
        name: string
        unitsPerGroup: number
        alignToChapter: boolean
      }>
    }
  }
  vlm: InsightProviderSelection
  chat: InsightProviderSelection & { useSameAsVlm: boolean }
  embedding: InsightProviderSelection
  reranker: InsightProviderSelection
  imageGen: InsightProviderSelection
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value)
  return (
    actual.length === expected.length &&
    expected.every(key => Object.prototype.hasOwnProperty.call(value, key))
  )
}

function requireInsightAppPayload(value: unknown): InsightAppPayload {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('后端 Insight 设置格式无效')
  }
  const payload = value as Record<string, unknown>
  const required = ['analysis', 'vlm', 'chat', 'embedding', 'reranker', 'imageGen']
  if (!hasExactKeys(payload, required)) {
    throw new Error('后端 Insight 设置字段不完整')
  }
  if (!isRecord(payload.analysis) || !hasExactKeys(payload.analysis, ['batch'])) {
    throw new Error('后端 Insight 分析设置格式无效')
  }
  const batch = payload.analysis.batch
  if (
    !isRecord(batch) ||
    !hasExactKeys(batch, [
      'pagesPerBatch',
      'contextBatchCount',
      'architecturePreset',
      'customLayers',
    ])
  ) {
    throw new Error('后端 Insight 批量设置格式无效')
  }
  const architecturePresets = new Set(['simple', 'standard', 'chapter_based', 'full', 'custom'])
  if (
    typeof batch.pagesPerBatch !== 'number' ||
    !Number.isSafeInteger(batch.pagesPerBatch) ||
    batch.pagesPerBatch < 1 ||
    typeof batch.contextBatchCount !== 'number' ||
    !Number.isSafeInteger(batch.contextBatchCount) ||
    batch.contextBatchCount < 0 ||
    typeof batch.architecturePreset !== 'string' ||
    !architecturePresets.has(batch.architecturePreset) ||
    !Array.isArray(batch.customLayers) ||
    (batch.architecturePreset === 'custom' && batch.customLayers.length < 2) ||
    batch.customLayers.some(
      layer =>
        !isRecord(layer) ||
        !hasExactKeys(layer, ['name', 'unitsPerGroup', 'alignToChapter']) ||
        typeof layer.name !== 'string' ||
        !layer.name.trim() ||
        typeof layer.unitsPerGroup !== 'number' ||
        !Number.isSafeInteger(layer.unitsPerGroup) ||
        layer.unitsPerGroup < 0 ||
        typeof layer.alignToChapter !== 'boolean'
    )
  ) {
    throw new Error('后端 Insight 批量设置字段类型无效')
  }
  for (const key of ['vlm', 'embedding', 'reranker', 'imageGen'] as const) {
    const section = payload[key]
    if (
      !isRecord(section) ||
      !hasExactKeys(section, ['provider']) ||
      typeof section.provider !== 'string' ||
      section.provider !== section.provider.trim()
    ) {
      throw new Error(`后端 Insight ${key} 设置格式无效`)
    }
  }
  if (
    !isRecord(payload.chat) ||
    !hasExactKeys(payload.chat, ['provider', 'useSameAsVlm']) ||
    typeof payload.chat.provider !== 'string' ||
    payload.chat.provider !== payload.chat.provider.trim() ||
    typeof payload.chat.useSameAsVlm !== 'boolean'
  ) {
    throw new Error('后端 Insight chat 设置格式无效')
  }
  return payload as InsightAppPayload
}

export function hasInsightCredential(domain: string, provider: string): boolean {
  return credentialSummaries.some(
    row => row.domain === domain && row.provider === provider && row.hasKey
  )
}

export async function getGlobalConfig(): Promise<InsightSettingsSnapshot> {
  const [document, prompts] = await Promise.all([getV2Settings(INSIGHT_DOMAINS), listV2Prompts()])
  credentialSummaries = mergeCredentialSummaries(
    document.credentials,
    await restoreBrowserCredentialLeases(),
  )
  const appEntry = document.settings.find(row => row.domain === 'insight')
  if (!appEntry) throw new Error('后端 Insight 设置缺失')
  const app = requireInsightAppPayload(appEntry.payload)
  const providerDrafts = readProviderDrafts(document)
  const vlm = activeProviderDraft(app.vlm.provider, providerDrafts.vlm, defaultVlmDraft)
  const llm = activeProviderDraft(app.chat.provider, providerDrafts.llm, defaultLlmDraft)
  const embedding = activeProviderDraft(
    app.embedding.provider,
    providerDrafts.embedding,
    defaultEmbeddingDraft
  )
  const reranker = activeProviderDraft(
    app.reranker.provider,
    providerDrafts.reranker,
    defaultRerankerDraft
  )
  const imageGen = activeProviderDraft(
    app.imageGen.provider,
    providerDrafts.imageGen,
    defaultImageGenDraft
  )
  const batch = app.analysis.batch
  return {
    config: {
      vlm: { provider: app.vlm.provider, ...vlm },
      llm: {
        provider: app.chat.provider,
        useSameAsVlm: app.chat.useSameAsVlm,
        ...llm,
      },
      embedding: { provider: app.embedding.provider, ...embedding },
      reranker: { provider: app.reranker.provider, ...reranker },
      imageGen: { provider: app.imageGen.provider, ...imageGen },
      batch: {
        pagesPerBatch: batch.pagesPerBatch,
        contextBatchCount: batch.contextBatchCount,
        architecturePreset: batch.architecturePreset,
        customLayers: batch.customLayers.map(layer => ({
          name: layer.name,
          units: layer.unitsPerGroup,
          align: layer.alignToChapter,
        })),
      },
      prompts: readInsightFactoryPrompts(prompts),
    },
    providerDrafts,
  }
}

function providerCommonPayload(
  draft: { model: string; baseUrl: string },
  label: string
): { modelName: string; customBaseUrl: string } {
  if (typeof draft.model !== 'string') throw new Error(`${label}.model 格式无效`)
  if (typeof draft.baseUrl !== 'string') throw new Error(`${label}.baseUrl 格式无效`)
  return { modelName: draft.model, customBaseUrl: draft.baseUrl }
}

function serializeVlmDraft(draft: InsightVlmProviderDraft, label: string): Record<string, unknown> {
  const openaiOptions = serializeOpenAICompatibleOptionsForApi(draft.openaiOptions)
  requireOpenAiOptionsWire({ openaiOptions }, label)
  requireProviderInteger({ imageMaxSize: draft.imageMaxSize }, 'imageMaxSize', label)
  return {
    ...providerCommonPayload(draft, label),
    openaiOptions,
    imageMaxSize: draft.imageMaxSize,
  }
}

function serializeLlmDraft(draft: InsightLlmProviderDraft, label: string): Record<string, unknown> {
  const openaiOptions = serializeOpenAICompatibleOptionsForApi(draft.openaiOptions)
  requireOpenAiOptionsWire({ openaiOptions }, label)
  return { ...providerCommonPayload(draft, label), openaiOptions }
}

function serializeEmbeddingDraft(
  draft: InsightEmbeddingProviderDraft,
  label: string
): Record<string, unknown> {
  requireProviderInteger({ rpmLimit: draft.rpmLimit }, 'rpmLimit', label)
  requireProviderInteger({ transportRetries: draft.transportRetries }, 'transportRetries', label)
  requireProviderInteger({ businessRetries: draft.businessRetries }, 'businessRetries', label)
  requireProviderNumber({ timeoutSeconds: draft.timeoutSeconds }, 'timeoutSeconds', label)
  return {
    ...providerCommonPayload(draft, label),
    rpmLimit: draft.rpmLimit,
    transportRetries: draft.transportRetries,
    businessRetries: draft.businessRetries,
    timeoutSeconds: draft.timeoutSeconds,
  }
}

function serializeRetriedProviderDraft(
  draft: InsightRerankerProviderDraft | InsightImageGenProviderDraft,
  label: string
): Record<string, unknown> {
  requireProviderInteger({ transportRetries: draft.transportRetries }, 'transportRetries', label)
  requireProviderInteger({ businessRetries: draft.businessRetries }, 'businessRetries', label)
  requireProviderNumber({ timeoutSeconds: draft.timeoutSeconds }, 'timeoutSeconds', label)
  return {
    ...providerCommonPayload(draft, label),
    transportRetries: draft.transportRetries,
    businessRetries: draft.businessRetries,
    timeoutSeconds: draft.timeoutSeconds,
  }
}

function requireActiveProvider(domain: InsightProviderDomain, provider: unknown): string {
  if (typeof provider !== 'string' || !provider || provider !== provider.trim()) {
    throw new Error(`${domain} provider 不能为空且不能包含首尾空格`)
  }
  return provider
}

function mergeCredentialSummaries(
  current: V2CredentialSummary[],
  changed: V2CredentialSummary[]
): V2CredentialSummary[] {
  const byIdentity = new Map(current.map(row => [`${row.domain}\0${row.provider}`, row]))
  for (const row of changed) byIdentity.set(`${row.domain}\0${row.provider}`, row)
  return [...byIdentity.values()]
}

function withoutInsightApiKeys(snapshot: InsightSettingsSnapshot): InsightSettingsSnapshot {
  const saved = deepClone(snapshot)
  saved.config.vlm.apiKey = ''
  saved.config.llm.apiKey = ''
  saved.config.embedding.apiKey = ''
  saved.config.reranker.apiKey = ''
  saved.config.imageGen.apiKey = ''
  const clearDraftKeys = <TDraft extends { apiKey: string }>(
    drafts: Record<string, TDraft>
  ): void => {
    for (const draft of Object.values(drafts)) draft.apiKey = ''
  }
  clearDraftKeys(saved.providerDrafts.vlm)
  clearDraftKeys(saved.providerDrafts.llm)
  clearDraftKeys(saved.providerDrafts.embedding)
  clearDraftKeys(saved.providerDrafts.reranker)
  clearDraftKeys(saved.providerDrafts.imageGen)
  return saved
}

export async function saveGlobalConfig(
  snapshot: InsightSettingsSnapshot
): Promise<InsightSettingsSnapshot> {
  const [document, currentPrompts] = await Promise.all([
    getV2Settings(INSIGHT_DOMAINS),
    listV2Prompts(),
  ])
  credentialSummaries = document.credentials
  const currentApp = document.settings.find(row => row.domain === 'insight')
  if (!currentApp) throw new Error('后端 Insight 设置缺失')
  const { config, providerDrafts } = snapshot
  const providerSettings: V2ProviderSettingMutation[] = []
  const credentialEdits: V2CredentialEdit[] = []
  const promptEdits: V2PromptMutation[] = []

  const vlmProvider = requireActiveProvider(INSIGHT_PROVIDER_DOMAINS.vlm, config.vlm.provider)
  const llmProvider = requireActiveProvider(INSIGHT_PROVIDER_DOMAINS.llm, config.llm.provider)
  const embeddingProvider = requireActiveProvider(
    INSIGHT_PROVIDER_DOMAINS.embedding,
    config.embedding.provider
  )
  const rerankerProvider = requireActiveProvider(
    INSIGHT_PROVIDER_DOMAINS.reranker,
    config.reranker.provider
  )
  const imageGenProvider = requireActiveProvider(
    INSIGHT_PROVIDER_DOMAINS.imageGen,
    config.imageGen.provider
  )
  const appPayload = requireInsightAppPayload({
    analysis: {
      batch: {
        pagesPerBatch: config.batch.pagesPerBatch,
        contextBatchCount: config.batch.contextBatchCount,
        architecturePreset: config.batch.architecturePreset,
        customLayers: config.batch.customLayers.map(layer => ({
          name: layer.name,
          unitsPerGroup: layer.units,
          alignToChapter: layer.align,
        })),
      },
    },
    vlm: { provider: vlmProvider },
    chat: { provider: llmProvider, useSameAsVlm: config.llm.useSameAsVlm },
    embedding: { provider: embeddingProvider },
    reranker: { provider: rerankerProvider },
    imageGen: { provider: imageGenProvider },
  })

  function appendProviderMutations<TDraft extends { apiKey: string }>(
    domain: InsightProviderDomain,
    drafts: Record<string, TDraft>,
    activeProvider: string,
    activeDraft: TDraft,
    serialize: (draft: TDraft, label: string) => Record<string, unknown>
  ): void {
    const mergedDrafts = new Map(Object.entries(drafts))
    mergedDrafts.set(activeProvider, activeDraft)
    for (const [provider, draft] of mergedDrafts) {
      if (!provider || provider !== provider.trim() || !isRecord(draft)) {
        throw new Error(`${domain} provider draft 格式无效`)
      }
      if (typeof draft.apiKey !== 'string') {
        throw new Error(`${domain}.${provider}.apiKey 格式无效`)
      }
      const existingRow = document.providerSettings.find(
        row => row.domain === domain && row.provider === provider
      )
      const existingCredential = document.credentials.find(
        row => row.domain === domain && row.provider === provider
      )
      const mutation: V2ProviderSettingMutation = {
        domain,
        provider,
        payload: serialize(draft, `${domain}.${provider}`),
        baseRevision: existingRow?.revision ?? 0,
        schemaVersion: 1,
        ...(existingRow?.credentialVersionId
          ? { credentialVersionId: existingRow.credentialVersionId }
          : {}),
      }
      const secret = draft.apiKey.trim()
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
  }

  const { provider: _vlmProvider, ...vlmDraft } = config.vlm
  const { provider: _llmProvider, useSameAsVlm: _useSameAsVlm, ...llmDraft } = config.llm
  const { provider: _embeddingProvider, ...embeddingDraft } = config.embedding
  const { provider: _rerankerProvider, ...rerankerDraft } = config.reranker
  const { provider: _imageGenProvider, ...imageGenDraft } = config.imageGen
  appendProviderMutations(
    INSIGHT_PROVIDER_DOMAINS.vlm,
    providerDrafts.vlm,
    vlmProvider,
    vlmDraft,
    serializeVlmDraft
  )
  appendProviderMutations(
    INSIGHT_PROVIDER_DOMAINS.llm,
    providerDrafts.llm,
    llmProvider,
    llmDraft,
    serializeLlmDraft
  )
  appendProviderMutations(
    INSIGHT_PROVIDER_DOMAINS.embedding,
    providerDrafts.embedding,
    embeddingProvider,
    embeddingDraft,
    serializeEmbeddingDraft
  )
  appendProviderMutations(
    INSIGHT_PROVIDER_DOMAINS.reranker,
    providerDrafts.reranker,
    rerankerProvider,
    rerankerDraft,
    serializeRetriedProviderDraft
  )
  appendProviderMutations(
    INSIGHT_PROVIDER_DOMAINS.imageGen,
    providerDrafts.imageGen,
    imageGenProvider,
    imageGenDraft,
    serializeRetriedProviderDraft
  )

  const prompts = config.prompts
  if (!isRecord(prompts) || !hasExactKeys(prompts, INSIGHT_PROMPT_TYPES)) {
    throw new Error('Insight 提示词设置字段不完整')
  }
  for (const type of INSIGHT_PROMPT_TYPES) {
    const content = prompts[type]
    const factory = currentPrompts.find(prompt => prompt.type === type && prompt.isFactoryDefault)
    if (!factory) throw new Error(`后端默认提示词不存在：${type}`)
    if (typeof content !== 'string') throw new Error(`提示词内容格式无效：${type}`)
    if (factory.content === content) continue
    promptEdits.push({
      id: factory.id,
      name: factory.name,
      content,
      baseRevision: factory.revision,
    })
  }
  const prepared = await prepareBrowserCredentialTransaction({
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
    promptEdits,
  } as V2SettingsTransaction)
  const saved = await saveV2SettingsTransaction(prepared.transaction)
  credentialSummaries = mergeCredentialSummaries(
    mergeCredentialSummaries(document.credentials, saved.credentials),
    prepared.summaries,
  )
  return withoutInsightApiKeys(snapshot)
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

type InsightConnectionTestConfig = {
  provider: string
  api_key: string
  model: string
  base_url?: string
}

export function testVlmConnection(
  config: InsightConnectionTestConfig
): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('vlm', 'insight_vlm', config)
}

export function testEmbeddingConnection(
  config: InsightConnectionTestConfig
): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('embedding', 'insight_embedding', config)
}

export function testRerankerConnection(
  config: InsightConnectionTestConfig
): Promise<V2ConnectionTestResult> {
  return diagnosticRequest('reranker', 'insight_reranker', config)
}

export function testLlmConnection(
  config: InsightConnectionTestConfig & { use_same_as_vlm: boolean }
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

function readInsightFactoryPrompts(prompts: V2Prompt[]): Record<PromptType, string> {
  const defaults = {} as Record<PromptType, string>
  for (const type of INSIGHT_PROMPT_TYPES) {
    const prompt = prompts.find(value => value.type === type && value.isFactoryDefault)
    if (!prompt) throw new Error(`后端默认提示词不存在：${type}`)
    defaults[type] = prompt.content
  }
  return defaults
}

export async function getDefaultPrompts(): Promise<Record<PromptType, string>> {
  return readInsightFactoryPrompts(await listV2Prompts())
}

export async function resetDefaultPrompt(type: PromptType): Promise<string> {
  const prompts = await listV2Prompts()
  const factory = prompts.find(prompt => prompt.type === type && prompt.isFactoryDefault)
  if (!factory) throw new Error(`默认提示词不存在：${type}`)
  const reset = await resetV2Prompt(factory)
  return reset.content
}

function savedPrompt(prompt: V2Prompt): SavedPromptItem {
  if (!isInsightPromptType(prompt.type)) {
    throw new Error(`不支持的 Insight 提示词类型：${prompt.type}`)
  }
  return {
    id: prompt.id,
    name: prompt.name,
    type: prompt.type,
    content: prompt.content,
  }
}

export async function getPromptsLibrary(): Promise<SavedPromptItem[]> {
  const prompts = await listV2Prompts()
  return prompts
    .filter(prompt => !prompt.isFactoryDefault && isInsightPromptType(prompt.type))
    .map(savedPrompt)
}

export async function savePromptToLibrary(prompt: SavedPromptInput): Promise<SavedPromptItem> {
  const saved = await createV2Prompt(prompt.type, prompt.name, prompt.content)
  return savedPrompt(saved)
}

export async function deletePromptFromLibrary(promptId: string): Promise<void> {
  await deleteV2Prompt(promptId)
}

export async function importPromptsLibrary(
  library: SavedPromptInput[]
): Promise<SavedPromptItem[]> {
  const current = await listV2Prompts()
  for (const prompt of library) {
    if (!isInsightPromptType(prompt.type)) {
      throw new Error(`不支持的 Insight 提示词类型：${prompt.type}`)
    }
    const existing = current.find(
      value => !value.isFactoryDefault && value.type === prompt.type && value.name === prompt.name
    )
    if (existing) {
      await updateV2Prompt({ ...existing, content: prompt.content })
    } else {
      await createV2Prompt(prompt.type, prompt.name, prompt.content)
    }
  }
  const saved = await listV2Prompts()
  return saved
    .filter(prompt => !prompt.isFactoryDefault && isInsightPromptType(prompt.type))
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
