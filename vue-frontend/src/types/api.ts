/**
 * API 响应类型定义
 * 定义与后端 API 交互的数据结构
 */

import type { BubbleState, BubbleCoords } from './bubble'

/**
 * 通用 API 响应
 */
export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

/**
 * API 错误
 */
export interface ApiError {
  code: string
  message: string
  status: number
  details?: Record<string, unknown>
}

/**
 * 翻译 API 响应
 * 后端成功时返回 translated_image 等字段，失败时返回 error
 * 注意：后端成功响应不包含 success 字段
 */
export interface TranslateImageResponse {
  success?: boolean
  translated_image?: string
  clean_image?: string
  bubble_coords?: BubbleCoords[]
  bubble_states?: BubbleState[]
  original_texts?: string[]
  bubble_texts?: string[]
  textbox_texts?: string[]
  bubble_angles?: number[]
  /** 自动检测的排版方向数组（'v' 表示垂直，'h' 表示水平）- 后端基于文本行分析 */
  auto_directions?: ('v' | 'h')[]
  error?: string
}

/**
 * 重新渲染 API 响应
 */
export interface ReRenderResponse {
  success: boolean
  translated_image?: string
  rendered_image?: string  // 后端实际返回的字段名
  bubble_states?: Array<{ fontSize?: number;[key: string]: any }>
  error?: string
}

/**
 * 仅检测气泡框 API 响应
 */
export interface DetectBoxesResponse {
  success: boolean
  /** 气泡坐标数组 */
  bubble_coords?: BubbleCoords[]
  /** 气泡旋转角度数组 */
  bubble_angles?: number[]
  /** 自动检测的排版方向数组（'v' 表示垂直，'h' 表示水平） */
  auto_directions?: ('v' | 'h')[]
  error?: string
}

/**
 * 单气泡 OCR API 响应
 */
export interface OcrSingleBubbleResponse {
  success: boolean
  text?: string
  error?: string
}

/**
 * 单气泡修复 API 响应
 */
export interface InpaintSingleBubbleResponse {
  success: boolean
  inpainted_image?: string
  error?: string
}

/**
 * 高质量翻译 API 响应
 */
export interface HqTranslateResponse {
  success: boolean
  results?: Array<{
    index: number
    translations: string[]
  }>
  error?: string
}

/**
 * 会话数据
 */
export interface SessionData {
  name: string
  version: string
  savedAt: string
  imageCount: number
  ui_settings: Record<string, unknown>
  images: Array<{
    originalDataURL: string
    translatedDataURL?: string
    cleanImageData?: string
    bubbleStates?: BubbleState[]
    fileName: string
    [key: string]: unknown
  }>
  currentImageIndex: number
}

/**
 * 会话列表项
 */
export interface SessionListItem {
  name: string
  savedAt: string
  imageCount: number
  version: string
}

/**
 * 书籍数据
 */
export interface BookData {
  id: string
  title: string
  cover?: string
  description?: string
  tags?: string[]
  chapters?: ChapterData[]
  /** 章节数量（后端返回） */
  chapter_count?: number
  /** 总页数（后端返回） */
  total_pages?: number
  createdAt: string
  updatedAt: string
  /** 后端可能返回的snake_case格式 */
  created_at?: string
  updated_at?: string
}

/**
 * 章节数据
 */
export interface ChapterData {
  id: string
  title: string
  order: number
  imageCount: number
  /** 后端可能返回的 snake_case 格式 */
  image_count?: number
  hasSession: boolean
  /** 后端可能返回的 snake_case 格式 */
  has_session?: boolean
  /** 会话文件路径（书架模式） */
  session_path?: string
}

/**
 * 标签数据
 */
export interface TagData {
  id: string
  name: string
  color?: string
}

/**
 * 插件数据
 */
export interface PluginData {
  name: string
  displayName: string
  description: string
  version: string
  enabled: boolean
  defaultEnabled: boolean
  configSchema?: Record<string, unknown>
  config?: Record<string, unknown>
}

/**
 * 字体信息
 */
export interface FontInfo {
  file_name: string
  display_name: string
  path: string
  is_default: boolean
}

/**
 * 字体列表响应
 * 后端 API 返回格式: { fonts: [...], default_fonts: {...} }
 */
export interface FontListResponse {
  success?: boolean
  /** 字体列表（新格式：对象数组） */
  fonts?: FontInfo[] | string[]
  /** 默认字体映射 */
  default_fonts?: Record<string, string>
  error?: string
}

/**
 * 提示词列表响应
 * 后端 API 返回格式: { prompt_names: [...], default_prompt_content: "..." }
 */
export interface PromptListResponse {
  success?: boolean
  /** 提示词名称列表（后端实体字段） */
  prompt_names?: string[]
  /** 默认提示词内容 */
  default_prompt_content?: string
  error?: string
}

/**
 * 模型信息项
 */
export interface ModelInfoItem {
  id: string
  name: string
}



/**
 * 获取模型列表响应（/api/fetch_models）
 * 后端返回的 models 是对象数组 [{id, name}]
 */
export interface FetchModelsResponse {
  success: boolean
  models?: ModelInfoItem[]
  message?: string
  error?: string
}

/**
 * 服务器信息响应
 */
export interface ServerInfoResponse {
  success: boolean
  local_url?: string
  lan_url?: string
  lan_ip?: string
  port?: number
  error?: string
}

/**
 * 连接测试响应
 */
export interface ConnectionTestResponse {
  success: boolean
  message?: string
  /** 模型列表（Ollama/Sakura连接测试时返回） */
  models?: string[]
  error?: string
}

/**
 * 下载会话响应
 */
export interface DownloadSessionResponse {
  success: boolean
  session_id?: string
  error?: string
}

/**
 * 下载完成响应
 */
export interface DownloadFinalizeResponse {
  success: boolean
  file_id?: string
  error?: string
}

/**
 * PDF 解析开始响应
 */
export interface PdfParseStartResponse {
  success: boolean
  session_id?: string
  total_pages?: number
  error?: string
}

/**
 * PDF 解析批次响应
 * 注意：images 是对象数组，每个对象包含 page_index 和 data_url（复刻原版）
 */
export interface PdfParseBatchResponse {
  success: boolean
  images?: Array<{
    page_index: number
    data_url: string
  }>
  has_more?: boolean
  error?: string
}

/**
 * 漫画分析状态响应
 */
export interface InsightStatusResponse {
  success: boolean
  status?: 'idle' | 'running' | 'paused' | 'completed' | 'failed'
  analyzed?: boolean
  analyzed_pages_count?: number
  current_task?: {
    task_id: string
    status: 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
    progress?: {
      analyzed_pages: number
      total_pages: number
      percentage: number
    }
  }
  progress?: {
    current: number
    total: number
    message?: string
  }
  error?: string
}

/**
 * 漫画分析概览响应
 */
export interface InsightOverviewResponse {
  success: boolean
  overview?: {
    type: string
    content: string
  }
  error?: string
}

/**
 * 漫画分析时间线响应
 */
export interface InsightTimelineResponse {
  success: boolean
  timeline?: Array<{
    event: string
    page: number
    description: string
  }>
  error?: string
}
