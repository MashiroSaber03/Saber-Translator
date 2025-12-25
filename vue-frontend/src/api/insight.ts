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
 */
export async function startAnalysis(
  bookId: string,
  options?: {
    mode?: 'full' | 'chapter' | 'page'
    chapter_id?: string
    page_num?: number
    incremental?: boolean
  }
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/start`, options)
}

/**
 * 暂停分析
 * @param bookId 书籍 ID
 */
export async function pauseAnalysis(bookId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/pause`)
}

/**
 * 继续分析
 * @param bookId 书籍 ID
 */
export async function resumeAnalysis(bookId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/resume`)
}

/**
 * 取消分析
 * @param bookId 书籍 ID
 */
export async function cancelAnalysis(bookId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/analyze/cancel`)
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
  return apiClient.post<ApiResponse>(`/api/manga-insight/${bookId}/reanalyze/page/${pageNum}`)
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
  return apiClient.post<InsightTimelineResponse>(`/api/manga-insight/${bookId}/regenerate/timeline`, {})
}

// ==================== 问答 API ====================

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
 * @param context 上下文（可选）
 */
export async function sendChatMessage(
  bookId: string,
  question: string,
  context?: {
    page_num?: number
    chapter_id?: string
  }
): Promise<ApiResponse<{ answer: string }>> {
  return apiClient.post(`/api/manga-insight/${bookId}/chat`, {
    question,
    stream: false,
    ...context,
  })
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
 * 默认提示词
 */
export const DEFAULT_PROMPTS: Record<PromptType, string> = {
  batch_analysis: `【输出中文】分析以下 {page_count} 页漫画内容（第 {start_page} 页到第 {end_page} 页）。

请提取以下信息，以 JSON 格式输出：
{
    "pages": [
        {
            "page_num": <页码>,
            "summary": "<该页内容概述，2-3句话>",
            "dialogues": [
                {"character": "<角色名或'旁白'>", "text": "<对话内容>"}
            ],
            "scene": "<场景描述>",
            "mood": "<氛围/情绪>"
        }
    ],
    "batch_summary": "<这几页的整体概述，3-5句话>",
    "key_events": ["<关键事件1>", "<关键事件2>"]
}

要求：
1. 准确识别对话内容和说话者
2. 注意画面中的文字、音效
3. 描述重要的视觉元素和表情`,

  segment_summary: `【输出中文】基于以下批次分析结果，生成段落总结。

请生成段落总结，JSON 格式：
{
    "summary": "<段落整体概述，3-5句话>",
    "plot_points": ["<剧情要点1>", "<剧情要点2>"],
    "character_developments": ["<角色发展1>"],
    "themes": ["<主题1>"]
}

要求：
1. 综合多个批次的信息
2. 突出重要角色和关键事件
3. 注意剧情的因果关系`,

  chapter_summary: `【输出中文】基于以下内容，生成完整的章节总结。

请生成章节总结，JSON 格式：
{
    "summary": "<章节整体概述，5-8句话>",
    "main_plot": "<主要剧情线描述>",
    "key_events": ["<章节关键事件，按顺序>"],
    "themes": ["<本章主题>"],
    "atmosphere": "<整体氛围>"
}

要求：
1. 综合所有内容，形成完整的章节叙述
2. 理清人物关系和剧情脉络
3. 提炼章节主题和核心冲突`,

  qa_response: `【输出中文】根据分析结果回答用户问题，引用相关页面。
回答时请：
1. 基于提供的漫画内容回答
2. 引用具体页码作为依据
3. 如果问题超出已分析内容，请诚实说明`,
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
 * 提示词配置响应
 */
export interface PromptsConfigResponse {
  success: boolean
  prompts?: Record<string, string>
  error?: string
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
