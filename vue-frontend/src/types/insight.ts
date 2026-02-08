/**
 * Manga Insight 类型定义
 * 统一的类型定义单一来源
 */

// ==================== Store 状态类型 ====================

/**
 * 分析状态（Store 使用）
 */
export type AnalysisStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'error'

/**
 * 分析模式（Store 使用）
 * 注意：UI 组件使用 'full'|'chapter'|'page'，API 使用 'full'|'chapters'|'incremental'|'reanalyze'
 */
export type AnalysisMode = 'full' | 'chapter' | 'page' | 'chapters' | 'incremental' | 'reanalyze'

/**
 * 概览模板类型
 * UI 使用的实际模板值
 */
export type OverviewTemplateType =
  | 'no_spoiler'
  | 'story_summary'
  | 'recap'
  | 'character_guide'
  | 'world_setting'
  | 'highlights'
  | 'reading_notes'
  // 兼容旧版
  | 'novel' | 'character' | 'timeline' | 'world' | 'theme' | 'relationship' | 'custom'

/**
 * 页面数据（Store 使用，camelCase）
 */
export interface PageData {
  pageNum: number
  analyzed: boolean
  summary?: string
  events?: string[]
  characters?: string[]
}

/**
 * 章节信息（Store 使用，camelCase）
 */
export interface ChapterInfo {
  id: string
  title: string
  pageRange?: { start: number; end: number }
  // 组件使用的扁平字段
  startPage: number
  endPage: number
  analyzed: boolean
  summary?: string
}

/**
 * 概览数据（Store 使用）
 */
export interface OverviewData {
  type: OverviewTemplateType  // 使用 type 而非 template，与 store 代码一致
  template?: OverviewTemplateType  // 兼容旧代码
  content: string
  generatedAt: string
}

/**
 * 问答消息（Store 使用）
 */
export interface QAMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Array<{ page: number; content: string }>
  isLoading?: boolean  // 加载状态标识
  mode?: string  // 问答模式
  citations?: Array<{ page: number }>  // 引用页面
  saved?: boolean  // 是否已保存
}

/**
 * 批量配置（Store 使用，camelCase）
 */
export interface BatchConfig {
  pagesPerBatch: number
  contextBatchCount: number
  architecturePreset: string
  customLayers: Array<{ name: string; units: number; align: boolean }>
}

// ==================== Store 专用配置类型（camelCase）====================

/**
 * Store VLM 配置（camelCase）
 */
export interface StoreVlmConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  rpmLimit?: number
  maxRetries?: number
  maxImagesPerRequest?: number
  temperature?: number
  forceJson?: boolean
  useStream?: boolean
  imageMaxSize?: number
}

/**
 * Store LLM 配置（camelCase）
 */
export interface StoreLlmConfig {
  useSameAsVlm: boolean
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  rpmLimit?: number
  maxRetries?: number
  useStream?: boolean
}

/**
 * Store Embedding 配置（camelCase）
 */
export interface StoreEmbeddingConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  dimension?: number
  rpmLimit?: number
  maxRetries?: number
}

/**
 * Store Reranker 配置（camelCase）
 */
export interface StoreRerankerConfig {
  enabled?: boolean
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  topK?: number
  rpmLimit?: number
  maxRetries?: number
}

/**
 * Store ImageGen 配置（camelCase）
 */
export interface StoreImageGenConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  maxRetries?: number
}

/**
 * Store 分析进度（camelCase，与 API 的 AnalysisProgress 区分）
 */
export interface StoreAnalysisProgress {
  current: number
  total: number
  status: AnalysisStatus
  message?: string
}

/**
 * Store 完整配置（camelCase）
 */
export interface StoreInsightConfig {
  vlm: StoreVlmConfig
  llm: StoreLlmConfig
  embedding: StoreEmbeddingConfig
  reranker: StoreRerankerConfig
  imageGen: StoreImageGenConfig
  batch: BatchConfig
  prompts: Record<string, string>
}

// ==================== 配置类型 ====================

/**
 * VLM（视觉语言模型）配置
 */
export interface VlmConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
  max_retries?: number
  max_images_per_request?: number
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
  rpm_limit?: number
  max_retries?: number
  use_stream?: boolean
}

/**
 * Embedding（向量嵌入）配置
 */
export interface EmbeddingConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  dimension?: number
  rpm_limit?: number
  max_retries?: number
}

/**
 * Reranker（重排序）配置
 */
export interface RerankerConfig {
  enabled?: boolean
  provider: string
  api_key: string
  model: string
  base_url?: string
  top_k?: number
  rpm_limit?: number
  max_retries?: number
}

/**
 * ImageGen（生图）配置
 */
export interface ImageGenConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  max_retries?: number
}

/**
 * 层级配置
 */
export interface LayerConfig {
  name: string
  units_per_group: number
  align_to_chapter: boolean
}

/**
 * 批量分析配置
 */
export interface BatchAnalysisConfig {
  pages_per_batch: number
  context_batch_count: number
  architecture_preset: string
  custom_layers?: LayerConfig[]
}

/**
 * 分析设置
 */
export interface AnalysisSettings {
  depth?: string
  auto_analyze_new_chapters?: boolean
  save_intermediate_results?: boolean
  batch?: BatchAnalysisConfig
}

/**
 * 提示词配置
 */
export interface PromptsConfig {
  batch_analysis?: string
  segment_summary?: string
  chapter_summary?: string
  book_overview?: string
  group_summary?: string
  qa_response?: string
  question_decompose?: string
  analysis_system?: string
}

/**
 * 完整 Insight 配置
 */
export interface InsightConfig {
  vlm?: VlmConfig
  chat_llm?: LlmConfig
  embedding?: EmbeddingConfig
  reranker?: RerankerConfig
  image_gen?: ImageGenConfig
  analysis?: AnalysisSettings
  prompts?: PromptsConfig
  providerSettings?: Record<string, Record<string, Record<string, unknown>>>
}

// ==================== 分析数据类型 ====================

/**
 * 页面范围
 */
export interface PageRange {
  start: number
  end: number
}

/**
 * 页面分析数据
 */
export interface PageAnalysis {
  page_number?: number
  page_num?: number
  page_summary?: string
  summary?: string
  scene?: string
  mood?: string
  from_batch?: boolean
  batch_range?: PageRange
  analyzed_at?: string
  panels?: Array<{
    dialogues?: Array<{
      speaker_name?: string
      character?: string
      text?: string
      translated_text?: string
    }>
  }>
}

/**
 * 批次分析结果
 */
export interface BatchAnalysis {
  page_range: PageRange
  pages: PageAnalysis[]
  batch_summary: string
  key_events: string[]
  continuity_notes?: string
  analyzed_at?: string
  parse_error?: boolean
}

/**
 * 段落摘要
 */
export interface SegmentSummary {
  segment_id: string
  page_range: PageRange
  summary: string
  key_events?: string[]
  plot_progression?: string
  themes?: string[]
  batch_count?: number
  generated_at?: string
}

/**
 * 章节分析
 */
export interface ChapterAnalysis {
  chapter_id: string
  title: string
  page_range: PageRange
  summary: string
  main_plot?: string
  plot_events?: string[]
  themes?: string[]
  atmosphere?: string
  connections?: {
    previous?: string
    foreshadowing?: string
  }
  segment_count?: number
  batch_count?: number
  analysis_mode?: string
  analyzed_at?: string
}

/**
 * 全书概览
 */
export interface BookOverview {
  book_id: string
  title: string
  total_pages: number
  total_chapters: number
  summary: string
  section_summaries?: string[]
  summary_source?: string
  themes?: string[]
  generated_at?: string
}

// ==================== 任务类型 ====================

/**
 * 任务状态枚举
 */
export type TaskStatus = 'pending' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed'

/**
 * 任务类型枚举
 */
export type TaskType = 'full_book' | 'chapter' | 'incremental' | 'reanalyze'

/**
 * 分析进度
 */
export interface AnalysisProgress {
  current_phase: string
  current_page: number
  analyzed_pages: number
  total_pages: number
  percentage?: number
}

/**
 * 分析任务
 */
export interface AnalysisTask {
  task_id: string
  book_id: string
  task_type: TaskType
  status: TaskStatus
  progress: AnalysisProgress
  target_chapters?: string[]
  target_pages?: number[]
  is_incremental?: boolean
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  failed_pages?: number[]
}

// ==================== 时间线类型 ====================

/**
 * 时间线事件
 */
export interface TimelineEvent {
  id: string
  page_range: PageRange
  title: string
  description: string
  type?: string
  importance?: number
  characters?: string[]
  arc_id?: string
}

/**
 * 剧情弧
 */
export interface StoryArc {
  id: string
  name: string
  description?: string
  page_range: PageRange
  events: TimelineEvent[]
}

/**
 * 时间线数据
 */
export interface TimelineData {
  events: TimelineEvent[]
  arcs?: StoryArc[]
  characters?: Array<{
    name: string
    appearances: number[]
  }>
  mode?: string
  stats?: {
    total_events: number
    total_arcs?: number
    total_characters?: number
  }
  generated_at?: string
}

// ==================== 笔记类型 ====================

/**
 * 笔记类型
 */
export type NoteType = 'text' | 'qa'

/**
 * 笔记数据（统一使用 camelCase）
 * 在 API 边界使用 converters.ts 进行转换
 */
export interface NoteData {
  id: string
  type: NoteType
  content: string
  pageNum?: number
  createdAt?: string
  updatedAt?: string
  // 扩展字段
  title?: string
  tags?: string[]
  question?: string
  answer?: string
  citations?: Array<{ page: number; content: string }>
  comment?: string
}

// ==================== 问答类型 ====================

/**
 * 问答历史记录
 */
export interface QAHistory {
  id: string
  question: string
  answer: string
  sources?: Array<{
    page_num: number
    content: string
    score?: number
  }>
  created_at: string
}

// ==================== 概览模板类型 ====================

/**
 * 概览模板元信息
 */
export interface OverviewTemplateMeta {
  name: string
  icon: string
  description: string
}

/**
 * 已生成的模板数据
 */
export interface GeneratedTemplate {
  template_key: string
  template_name?: string
  content?: string
  generated_at?: string
}

// ==================== API 响应类型 ====================

/**
 * 分析状态响应
 */
export interface InsightStatusResponse {
  success: boolean
  status?: TaskStatus
  task?: AnalysisTask
  progress?: AnalysisProgress
  total_pages?: number
  analyzed_pages?: number
  has_overview?: boolean
  has_timeline?: boolean
  error?: string
}

/**
 * 概览响应
 */
export interface InsightOverviewResponse {
  success: boolean
  overview?: BookOverview
  content?: string
  template_key?: string
  generated_at?: string
  error?: string
}

/**
 * 时间线响应
 */
export interface InsightTimelineResponse {
  success: boolean
  timeline?: TimelineData
  error?: string
}

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
  analysis?: PageAnalysis
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
 * 笔记列表响应
 */
export interface NoteListResponse {
  success: boolean
  notes?: NoteData[]
  error?: string
}

/**
 * 连接测试响应
 */
export interface ConnectionTestResponse {
  success: boolean
  message?: string
  error?: string
}
