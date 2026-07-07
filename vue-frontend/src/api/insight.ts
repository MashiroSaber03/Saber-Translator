import { apiClient } from './client'
import type { ApiResponse } from '@/types'
import type {
  InsightOverviewResponse,
  InsightStatusResponse,
  InsightTimelineResponse,
} from '@/types/insight'
import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'

export type { InsightOverviewResponse, InsightTimelineResponse }
export { fetchModels } from './config'

function insightPathSegment(value: string): string {
  return encodeURIComponent(value)
}

function insightEndpoint(suffix = ''): string {
  return `/api/manga-insight${suffix}`
}

function insightBookEndpoint(bookId: string, suffix = ''): string {
  return insightEndpoint(`/${insightPathSegment(bookId)}${suffix}`)
}

function insightQuery(params: Record<string, string | number | boolean | null | undefined>): string {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      query.set(key, String(value))
    }
  }
  const serialized = query.toString()
  return serialized ? `?${serialized}` : ''
}

export interface PageDialogueData {
  speaker_name?: string
  character?: string
  text?: string
  translated_text?: string
}

export interface PageAnalysisData {
  page_num?: number
  page_summary?: string
  scene?: string
  mood?: string
  analyzed?: boolean
  panels?: Array<{
    dialogues?: PageDialogueData[]
  }>
}

export interface PageDataResponse {
  success: boolean
  page?: PageAnalysisData
  analysis?: PageAnalysisData
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
    id: string
    title: string
    start_page: number
    end_page: number
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
  id: string
  type: 'text' | 'qa'
  content: string
  page_num?: number
  created_at: string
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
  vlm?: VlmConfig
  chat_llm?: LlmConfig
  embedding?: EmbeddingConfig
  reranker?: RerankerConfig
  image_gen?: ImageGenConfig
  analysis?: {
    batch?: BatchAnalysisConfig
  }
  prompts?: Record<string, string>
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
  error?: string
  message?: string
}

export type ReanalyzeResponse = StartAnalysisResponse

export interface ExportAnalysisResponse {
  success: boolean
  markdown?: string
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
  error?: string
  message?: string
}

export interface ChatResponse {
  success: boolean
  answer?: string
  mode?: string
  citations?: Array<{ page: number }>
  error?: string
}

export interface RebuildEmbeddingsResponse {
  success: boolean
  task_id?: string
  status?: string
  message?: string
  stats?: {
    pages_count?: number
    dialogues_count?: number
  }
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
    result_data?: {
      build_result?: Record<string, unknown>
    }
  } | null
  stats?: {
    available?: boolean
    pages_count?: number
    dialogues_count?: number
    scenes_count?: number
    events_count?: number
  }
  build_result?: Record<string, unknown>
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

export async function startAnalysis(
  bookId: string,
  options?: StartAnalysisOptions
): Promise<StartAnalysisResponse> {
  return apiClient.post<StartAnalysisResponse>(insightBookEndpoint(bookId, '/analyze/start'), options, {
    timeout: 0,
  })
}

export async function pauseAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightBookEndpoint(bookId, '/analyze/pause'), {
    task_id: taskId,
  })
}

export async function resumeAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightBookEndpoint(bookId, '/analyze/resume'), {
    task_id: taskId,
  })
}

export async function cancelAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightBookEndpoint(bookId, '/analyze/cancel'), {
    task_id: taskId,
  })
}

export async function getAnalysisStatus(bookId: string): Promise<InsightStatusResponse> {
  return apiClient.get<InsightStatusResponse>(insightBookEndpoint(bookId, '/analyze/status'))
}

export async function previewAnalysis(
  bookId: string,
  pages?: number[]
): Promise<PreviewAnalysisResponse> {
  return apiClient.post<PreviewAnalysisResponse>(
    insightBookEndpoint(bookId, '/preview'),
    pages ? { pages } : {},
    { timeout: 0 }
  )
}

export async function reanalyzePage(bookId: string, pageNum: number): Promise<ReanalyzeResponse> {
  return apiClient.post<ReanalyzeResponse>(insightBookEndpoint(bookId, `/reanalyze/page/${pageNum}`), {}, {
    timeout: 0,
  })
}

export async function reanalyzeChapter(bookId: string, chapterId: string): Promise<ReanalyzeResponse> {
  return apiClient.post<ReanalyzeResponse>(
    insightBookEndpoint(bookId, `/reanalyze/chapter/${insightPathSegment(chapterId)}`),
    {},
    { timeout: 0 }
  )
}

export async function getPageData(bookId: string, pageNum: number): Promise<PageDataResponse> {
  return apiClient.get<PageDataResponse>(insightBookEndpoint(bookId, `/pages/${pageNum}`))
}

export async function getAnalyzedPages(bookId: string): Promise<InsightPagesResponse> {
  return apiClient.get<InsightPagesResponse>(insightBookEndpoint(bookId, '/pages'))
}

export function getPageImageUrl(bookId: string, pageNum: number): string {
  return insightBookEndpoint(bookId, `/page-image/${pageNum}`)
}

export function getThumbnailUrl(bookId: string, pageNum: number): string {
  return insightBookEndpoint(bookId, `/thumbnail/${pageNum}`)
}

export async function getInsightChapters(bookId: string): Promise<InsightChapterListResponse> {
  return apiClient.get<InsightChapterListResponse>(insightBookEndpoint(bookId, '/chapters'))
}

export async function getOverviewBasic(bookId: string): Promise<InsightOverviewResponse> {
  return apiClient.get<InsightOverviewResponse>(insightBookEndpoint(bookId, '/overview'))
}

export async function getOverview(
  bookId: string,
  templateType?: string
): Promise<OverviewContentResponse> {
  const suffix = templateType ? `/overview/${insightPathSegment(templateType)}` : '/overview'
  return apiClient.get<OverviewContentResponse>(insightBookEndpoint(bookId, suffix))
}

export async function regenerateOverview(
  bookId: string,
  templateType: string,
  force = false
): Promise<OverviewContentResponse> {
  return apiClient.post<OverviewContentResponse>(
    insightBookEndpoint(bookId, '/overview/generate'),
    {
      template: templateType,
      force,
    },
    { timeout: 0 }
  )
}

export async function getGeneratedTemplates(bookId: string): Promise<GeneratedTemplatesResponse> {
  return apiClient.get<GeneratedTemplatesResponse>(insightBookEndpoint(bookId, '/overview/templates'))
}

export async function getTimeline(bookId: string): Promise<InsightTimelineResponse> {
  return apiClient.get<InsightTimelineResponse>(insightBookEndpoint(bookId, '/timeline'))
}

export async function regenerateTimeline(bookId: string): Promise<InsightTimelineResponse> {
  return apiClient.post<InsightTimelineResponse>(insightBookEndpoint(bookId, '/regenerate/timeline'), {}, {
    timeout: 0,
  })
}

export function getChatStreamUrl(bookId: string): string {
  return insightBookEndpoint(bookId, '/chat')
}

export async function sendChat(
  bookId: string,
  question: string,
  options?: {
    use_parent_child?: boolean
    use_reasoning?: boolean
    use_reranker?: boolean
    top_k?: number
    threshold?: number
    use_global_context?: boolean
  }
): Promise<ChatResponse> {
  return apiClient.post<ChatResponse>(
    insightBookEndpoint(bookId, '/chat'),
    {
      question,
      ...options,
    },
    { timeout: 0 }
  )
}

export async function rebuildEmbeddings(bookId: string): Promise<RebuildEmbeddingsResponse> {
  return apiClient.post<RebuildEmbeddingsResponse>(
    insightBookEndpoint(bookId, '/rebuild-embeddings'),
    {},
    { timeout: 0 }
  )
}

export async function getRebuildEmbeddingsStatus(
  bookId: string,
  taskId?: string
): Promise<RebuildEmbeddingsStatusResponse> {
  return apiClient.get<RebuildEmbeddingsStatusResponse>(
    insightBookEndpoint(bookId, `/rebuild-embeddings/status${insightQuery({ task_id: taskId })}`)
  )
}

export async function getNotes(bookId: string, type?: 'text' | 'qa'): Promise<NoteListResponse> {
  return apiClient.get<NoteListResponse>(insightBookEndpoint(bookId, '/notes'), {
    params: type ? { type } : undefined,
  })
}

export async function createNote(
  bookId: string,
  note: {
    type: 'text' | 'qa'
    content: string
    page_num?: number
  }
): Promise<NoteDetailResponse> {
  return apiClient.post<NoteDetailResponse>(insightBookEndpoint(bookId, '/notes'), note)
}

export async function updateNote(
  bookId: string,
  noteId: string,
  updates: { content?: string; page_num?: number }
): Promise<NoteDetailResponse> {
  return apiClient.put<NoteDetailResponse>(
    insightBookEndpoint(bookId, `/notes/${insightPathSegment(noteId)}`),
    updates
  )
}

export async function deleteNote(bookId: string, noteId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(insightBookEndpoint(bookId, `/notes/${insightPathSegment(noteId)}`))
}

export async function getGlobalConfig(): Promise<GlobalConfigResponse> {
  return apiClient.get<GlobalConfigResponse>(insightEndpoint('/config'))
}

export async function saveGlobalConfig(config: AnalysisConfig): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightEndpoint('/config'), config)
}

export async function testVlmConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(insightEndpoint('/config/test/vlm'), config)
}

export async function testEmbeddingConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(insightEndpoint('/config/test/embedding'), config)
}

export async function testRerankerConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(insightEndpoint('/config/test/reranker'), config)
}

export async function testLlmConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(insightEndpoint('/config/test/llm'), config)
}

export async function getDefaultPrompts(): Promise<DefaultPromptsResponse> {
  return apiClient.get<DefaultPromptsResponse>(insightEndpoint('/prompts/defaults'))
}

export async function getPromptsLibrary(): Promise<PromptsLibraryResponse> {
  return apiClient.get<PromptsLibraryResponse>(insightEndpoint('/prompts/library'))
}

export async function savePromptToLibrary(prompt: SavedPromptItem): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightEndpoint('/prompts/library'), prompt)
}

export async function deletePromptFromLibrary(promptId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(insightEndpoint(`/prompts/library/${insightPathSegment(promptId)}`))
}

export async function importPromptsLibrary(library: SavedPromptItem[]): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(insightEndpoint('/prompts/library/import'), { library })
}

export async function exportAnalysis(bookId: string): Promise<ExportAnalysisResponse> {
  return apiClient.get(insightBookEndpoint(bookId, '/export'))
}

export async function exportPageAnalysis(
  bookId: string,
  pageNum: number
): Promise<PageDataResponse> {
  return apiClient.get<PageDataResponse>(insightBookEndpoint(bookId, `/pages/${pageNum}`))
}
