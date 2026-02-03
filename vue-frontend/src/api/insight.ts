/**
 * 漫画分析 API
 * 包含分析控制、状态查询、页面数据、问答、笔记等功能
 */

import { apiClient } from './client'
import type {
  ApiResponse,
  InsightStatusResponse,
  InsightOverviewResponse,
  InsightTimelineResponse,
} from '@/types'

// 重新导出类型供组件使用
export type { InsightOverviewResponse, InsightTimelineResponse }

// ==================== 分析响应类型 ====================

/**
 * 页面数据响应
 */
export interface PageDataResponse {
  success: boolean
  page?: {
    page_num: number
    summary?: string
    dialogues?: Array<{
      character?: string
      text: string
      translated_text?: string
    }>
    analyzed: boolean
  }
  // 后端API实际返回的是analysis字段
  analysis?: {
    page_num?: number
    page_summary?: string
    scene?: string
    mood?: string
    panels?: Array<{
      dialogues?: Array<{
        speaker_name?: string
        character?: string
        text?: string
        translated_text?: string
      }>
    }>
  }
  error?: string
}

/**
 * 章节列表响应
 */
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

/**
 * 已生成模板列表响应
 */
export interface GeneratedTemplatesResponse {
  success: boolean
  templates?: Record<string, any>
  generated?: string[]
  generated_details?: Array<{ template_key: string; template_name?: string }>
  error?: string
}

/**
 * 笔记数据
 */
export interface NoteData {
  id: string
  type: 'text' | 'qa'
  content: string
  page_num?: number
  created_at: string
  updated_at: string
}

/**
 * 笔记列表响应
 */
export interface NoteListResponse {
  success: boolean
  notes?: NoteData[]
  error?: string
}

/**
 * 笔记详情响应
 */
export interface NoteDetailResponse {
  success: boolean
  note?: NoteData
  error?: string
}

/**
 * VLM 配置
 */
export interface VlmConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
  temperature?: number
  force_json?: boolean
  use_stream?: boolean
  image_max_size?: number
}

/**
 * LLM（对话模型）配置
 */
export interface LlmConfig {
  use_same_as_vlm: boolean
  provider?: string
  api_key?: string
  model?: string
  base_url?: string
  use_stream?: boolean
}

/**
 * Embedding 配置
 */
export interface EmbeddingConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
}

/**
 * Reranker 配置
 */
export interface RerankerConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  top_k?: number
}

/**
 * 批量分析配置
 */
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

/**
 * 分析配置
 */
export interface AnalysisConfig {
  vlm?: VlmConfig
  chat_llm?: LlmConfig
  embedding?: EmbeddingConfig
  reranker?: RerankerConfig
  analysis?: {
    batch?: BatchAnalysisConfig
  }
  prompts?: Record<string, string>
}

/**
 * 连接测试响应
 */
export interface ConnectionTestResponse {
  success: boolean
  error?: string
  message?: string
}

// ==================== 分析控制 API ====================

/**
 * 开始分析
 * @param bookId 书籍 ID
 * @param options 分析选项
 * 
 * 后端期望的 mode 为：
 * - 'full': 全书分析（强制重新分析所有页面）
 * - 'incremental': 增量分析（仅分析未分析的页面）
 * - 'chapters': 章节分析，需要配合 chapters 数组
 * - 'pages': 页面分析，需要配合 pages 数组
 */
export async function startAnalysis(
  bookId: string,
  options?: {
    mode?: 'full' | 'incremental' | 'chapters' | 'pages'
    chapters?: string[]   // 章节ID数组（chapters模式）
    pages?: number[]      // 页码数组（pages模式）
    force?: boolean       // 是否强制重新分析
  }
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/start`, options, {
    timeout: 0  // 移除超时限制，分析可能很耗时
  })
}

/**
 * 暂停分析
 * @param bookId 书籍 ID
 * @param taskId 任务 ID（可选，不传则后端取最新任务）
 */
export async function pauseAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/pause`, {
    task_id: taskId
  })
}

/**
 * 继续分析
 * @param bookId 书籍 ID
 * @param taskId 任务 ID（可选，不传则后端取最新任务）
 */
export async function resumeAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/resume`, {
    task_id: taskId
  })
}

/**
 * 取消分析
 * @param bookId 书籍 ID
 * @param taskId 任务 ID（可选，不传则后端取最新任务）
 */
export async function cancelAnalysis(bookId: string, taskId?: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/cancel`, {
    task_id: taskId
  })
}

/**
 * 获取分析状态
 * @param bookId 书籍 ID
 */
export async function getAnalysisStatus(bookId: string): Promise<InsightStatusResponse> {
  return apiClient.get<InsightStatusResponse>(`/api/manga-insight/${bookId}/analyze/status`)
}

/**
 * 重新分析单页
 * @param bookId 书籍 ID
 * @param pageNum 页码
 */
export async function reanalyzePage(bookId: string, pageNum: number): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/reanalyze/page/${pageNum}`, {}, {
    timeout: 0  // 移除超时限制，AI分析可能很耗时
  })
}

// ==================== 页面数据 API ====================

/**
 * 获取页面数据
 * @param bookId 书籍 ID
 * @param pageNum 页码
 */
export async function getPageData(bookId: string, pageNum: number): Promise<PageDataResponse> {
  return apiClient.get<PageDataResponse>(`/api/manga-insight/${bookId}/pages/${pageNum}`)
}

/**
 * 获取页面图片 URL
 * @param bookId 书籍 ID
 * @param pageNum 页码
 */
export function getPageImageUrl(bookId: string, pageNum: number): string {
  return `/api/manga-insight/${bookId}/page-image/${pageNum}`
}

/**
 * 获取缩略图 URL
 * @param bookId 书籍 ID
 * @param pageNum 页码
 */
export function getThumbnailUrl(bookId: string, pageNum: number): string {
  return `/api/manga-insight/${bookId}/thumbnail/${pageNum}`
}

/**
 * 获取章节列表
 * @param bookId 书籍 ID
 */
export async function getInsightChapters(bookId: string): Promise<InsightChapterListResponse> {
  return apiClient.get<InsightChapterListResponse>(`/api/manga-insight/${bookId}/chapters`)
}

// ==================== 概览和时间线 API ====================

/**
 * 获取概览（基础版，无模板）
 * @param bookId 书籍 ID
 */
export async function getOverviewBasic(
  bookId: string
): Promise<InsightOverviewResponse> {
  return apiClient.get<InsightOverviewResponse>(`/api/manga-insight/${bookId}/overview`)
}

/**
 * 获取模板概览（从缓存读取）
 * @param bookId 书籍 ID
 * @param templateType 模板类型
 */
export async function getOverview(
  bookId: string,
  templateType?: string
): Promise<any> {
  // 使用正确的API路由: /overview/{template_key}
  if (templateType) {
    return apiClient.get(`/api/manga-insight/${bookId}/overview/${templateType}`)
  }
  return apiClient.get<InsightOverviewResponse>(`/api/manga-insight/${bookId}/overview`)
}

/**
 * 生成/重新生成概览
 * @param bookId 书籍 ID
 * @param templateType 模板类型
 * @param force 是否强制重新生成
 */
export async function regenerateOverview(
  bookId: string,
  templateType: string,
  force: boolean = false
): Promise<any> {
  // 使用正确的API路由: POST /overview/generate
  return apiClient.post(`/api/manga-insight/${bookId}/overview/generate`, {
    template: templateType,
    force: force,
  }, {
    timeout: 0  // 移除超时限制，概览生成可能很耗时
  })
}

/**
 * 获取已生成的模板列表
 * @param bookId 书籍 ID
 */
export async function getGeneratedTemplates(bookId: string): Promise<GeneratedTemplatesResponse> {
  return apiClient.get<GeneratedTemplatesResponse>(
    `/api/manga-insight/${bookId}/overview/templates`
  )
}

/**
 * 获取时间线
 * @param bookId 书籍 ID
 */
export async function getTimeline(bookId: string): Promise<InsightTimelineResponse> {
  return apiClient.get<InsightTimelineResponse>(`/api/manga-insight/${bookId}/timeline`)
}

/**
 * 重新生成时间线
 * @param bookId 书籍 ID
 */
export async function regenerateTimeline(bookId: string): Promise<InsightTimelineResponse> {
  // 使用正确的API路由: POST /regenerate/timeline
  return apiClient.post<InsightTimelineResponse>(`/api/manga-insight/${bookId}/regenerate/timeline`, {}, {
    timeout: 0  // 移除超时限制，时间线生成可能很耗时
  })
}

// ==================== 问答 API ====================

/**
 * 问答响应类型
 */
export interface ChatResponse {
  success: boolean
  answer?: string
  mode?: string
  citations?: Array<{ page: number }>
  error?: string
}

/**
 * 发送问答请求（返回 EventSource URL，用于 SSE 流式响应）
 * @param bookId 书籍 ID
 */
export function getChatStreamUrl(bookId: string): string {
  return `/api/manga-insight/${bookId}/chat`
}

/**
 * 发送问答请求（非流式）
 * @param bookId 书籍 ID
 * @param question 问题
 * @param options 问答选项
 */
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
  return apiClient.post<ChatResponse>(`/api/manga-insight/${bookId}/chat`, {
    question,
    ...options,
  }, {
    timeout: 0  // 移除超时限制，AI问答可能很耗时
  })
}

/**
 * 重建向量索引响应类型
 */
export interface RebuildEmbeddingsResponse {
  success: boolean
  stats?: {
    pages_count?: number
    dialogues_count?: number
  }
  error?: string
}

/**
 * 重建向量索引
 * @param bookId 书籍 ID
 */
export async function rebuildEmbeddings(bookId: string): Promise<RebuildEmbeddingsResponse> {
  return apiClient.post<RebuildEmbeddingsResponse>(
    `/api/manga-insight/${bookId}/rebuild-embeddings`,
    {},
    {
      timeout: 0  // 移除超时限制，向量索引重建可能很耗时
    }
  )
}

// ==================== 笔记 API ====================

/**
 * 获取笔记列表
 * @param bookId 书籍 ID
 * @param type 笔记类型筛选
 */
export async function getNotes(bookId: string, type?: 'text' | 'qa'): Promise<NoteListResponse> {
  return apiClient.get<NoteListResponse>(`/api/manga-insight/${bookId}/notes`, {
    params: type ? { type } : undefined,
  })
}

/**
 * 创建笔记
 * @param bookId 书籍 ID
 * @param note 笔记数据
 */
export async function createNote(
  bookId: string,
  note: {
    type: 'text' | 'qa'
    content: string
    page_num?: number
  }
): Promise<NoteDetailResponse> {
  return apiClient.post<NoteDetailResponse>(`/api/manga-insight/${bookId}/notes`, note)
}

/**
 * 更新笔记
 * @param bookId 书籍 ID
 * @param noteId 笔记 ID
 * @param content 新内容
 */
export async function updateNote(
  bookId: string,
  noteId: string,
  content: string
): Promise<NoteDetailResponse> {
  return apiClient.put<NoteDetailResponse>(`/api/manga-insight/${bookId}/notes/${noteId}`, {
    content,
  })
}

/**
 * 删除笔记
 * @param bookId 书籍 ID
 * @param noteId 笔记 ID
 */
export async function deleteNote(bookId: string, noteId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/manga-insight/${bookId}/notes/${noteId}`)
}

// ==================== 配置 API ====================

/**
 * 获取分析配置
 * @param bookId 书籍 ID
 */
export async function getAnalysisConfig(
  bookId: string
): Promise<ApiResponse<{ config: AnalysisConfig }>> {
  return apiClient.get(`/api/manga-insight/${bookId}/config`)
}

/**
 * 保存分析配置
 * @param bookId 书籍 ID
 * @param config 配置数据
 */
export async function saveAnalysisConfig(
  bookId: string,
  config: AnalysisConfig
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/config`, config)
}

// ==================== 全局配置 API ====================

/**
 * 全局配置响应类型
 */
export interface GlobalConfigResponse {
  success: boolean
  config?: AnalysisConfig
  error?: string
}

/**
 * 获取全局分析配置（不依赖书籍）
 */
export async function getGlobalConfig(): Promise<GlobalConfigResponse> {
  return apiClient.get<GlobalConfigResponse>('/api/manga-insight/config')
}

/**
 * 保存全局分析配置（不依赖书籍）
 * @param config 配置数据
 */
export async function saveGlobalConfig(config: AnalysisConfig): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/manga-insight/config', config)
}

// ==================== 连接测试 API ====================

/**
 * 测试 VLM 连接
 * @param config VLM 配置
 */
export async function testVlmConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/manga-insight/config/test/vlm', config)
}

/**
 * 测试 Embedding 连接
 * @param config Embedding 配置
 */
export async function testEmbeddingConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/manga-insight/config/test/embedding', config)
}

/**
 * 测试 Reranker 连接
 * @param config Reranker 配置
 */
export async function testRerankerConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/manga-insight/config/test/reranker', config)
}

/**
 * 测试 LLM 连接
 * @param config LLM 配置
 */
export async function testLlmConnection(config: {
  provider: string
  api_key: string
  model: string
  base_url?: string
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/manga-insight/config/test/llm', config)
}

// ==================== 模型获取 API ====================

// 从 config.ts 重新导出 fetchModels，避免重复定义
export { fetchModels } from './config'

// ==================== 提示词管理 API ====================

/**
 * 提示词类型
 */
export type PromptType = 'batch_analysis' | 'segment_summary' | 'chapter_summary' | 'qa_response'

/**
 * 提示词元数据
 */
export interface PromptMetadata {
  label: string
  hint: string
}

/**
 * 提示词元数据映射
 */
export const PROMPT_METADATA: Record<PromptType, PromptMetadata> = {
  batch_analysis: {
    label: '📄 批量分析提示词',
    hint: '用于批量分析多个页面。支持变量：{page_count}, {start_page}, {end_page}',
  },
  segment_summary: {
    label: '📑 段落总结提示词',
    hint: '用于汇总多个批次的分析结果生成段落总结。',
  },
  chapter_summary: {
    label: '📖 章节总结提示词',
    hint: '用于生成章节级别的完整总结。',
  },
  qa_response: {
    label: '💬 问答响应提示词',
    hint: '用于回答用户关于漫画内容的问题。',
  },
}

/**
 * 保存的提示词项
 */
export interface SavedPromptItem {
  id: string
  name: string
  type: PromptType
  content: string
  created_at: string
}

/**
 * 提示词库响应
 */
export interface PromptsLibraryResponse {
  success: boolean
  library?: SavedPromptItem[]
  error?: string
}

/**
 * 默认提示词响应
 */
export interface DefaultPromptsResponse {
  success: boolean
  prompts?: Record<PromptType, string>
  error?: string
}

/**
 * 获取默认提示词（从后端）
 */
export async function getDefaultPrompts(): Promise<DefaultPromptsResponse> {
  return apiClient.get<DefaultPromptsResponse>('/api/manga-insight/prompts/defaults')
}

/**
 * 获取提示词库
 */
export async function getPromptsLibrary(): Promise<PromptsLibraryResponse> {
  return apiClient.get<PromptsLibraryResponse>('/api/manga-insight/prompts/library')
}

/**
 * 保存提示词到库
 * @param prompt 提示词数据
 */
export async function savePromptToLibrary(prompt: SavedPromptItem): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/manga-insight/prompts/library', prompt)
}

/**
 * 从库删除提示词
 * @param promptId 提示词 ID
 */
export async function deletePromptFromLibrary(promptId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/manga-insight/prompts/library/${promptId}`)
}

/**
 * 导入提示词库
 * @param library 提示词库数据
 */
export async function importPromptsLibrary(library: SavedPromptItem[]): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/manga-insight/prompts/library/import', { library })
}

/**
 * 导出分析数据
 * @param bookId 书籍 ID
 */
export async function exportAnalysis(
  bookId: string
): Promise<ApiResponse<{ markdown: string }>> {
  return apiClient.get(`/api/manga-insight/${bookId}/export`)
}

/**
 * 导出页面分析数据
 * @param bookId 书籍 ID
 * @param pageNum 页码
 */
export async function exportPageAnalysis(
  bookId: string,
  pageNum: number
): Promise<PageDataResponse> {
  return apiClient.get<PageDataResponse>(`/api/manga-insight/${bookId}/pages/${pageNum}`)
}
