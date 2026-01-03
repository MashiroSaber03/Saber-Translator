/**
 * 翻译 API
 * 包含图片翻译、重新渲染、高质量翻译、气泡检测等功能
 */

import { apiClient } from './client'
import type {
  ApiResponse,
  TranslateImageResponse,
  ReRenderResponse,
  DetectBoxesResponse,
  OcrSingleBubbleResponse,
  InpaintSingleBubbleResponse,
  HqTranslateResponse,
  BubbleState,
  BubbleCoords,
} from '@/types'

// ==================== 翻译请求参数类型 ====================

/**
 * 翻译图片请求参数
 * 字段命名与后端 TranslationRequest.from_dict() 保持一致
 */
export interface TranslateImageParams {
  // 图片数据
  image: string // Base64 图片数据（纯 Base64，不含 DataURL 前缀）

  // OCR 设置
  ocr_engine: string // OCR 引擎: 'manga_ocr' | 'baidu_ocr' | 'ai_vision'
  source_language?: string // 源语言

  // 百度 OCR 设置（后端字段名）
  baidu_api_key?: string
  baidu_secret_key?: string
  baidu_version?: string
  baidu_ocr_language?: string

  // AI 视觉 OCR 设置（后端字段名）
  ai_vision_provider?: string
  ai_vision_api_key?: string
  ai_vision_model_name?: string
  ai_vision_ocr_prompt?: string
  rpm_limit_ai_vision_ocr?: number
  custom_ai_vision_base_url?: string
  use_json_format_ai_vision_ocr?: boolean

  // 翻译服务设置（后端使用 model_provider, model_name, api_key）
  model_provider: string
  api_key?: string
  model_name?: string
  custom_base_url?: string
  target_language: string
  prompt_content?: string
  textbox_prompt_content?: string
  use_textbox_prompt?: boolean
  rpm_limit_translation?: number
  max_retries?: number
  use_json_format_translation?: boolean

  // 文字检测设置（后端使用 detector_type）
  detector_type?: string
  box_expand_ratio?: number
  box_expand_top?: number
  box_expand_bottom?: number
  box_expand_left?: number
  box_expand_right?: number

  // 精确文字掩膜设置（后端使用驼峰命名）
  usePreciseMask?: boolean
  maskDilateSize?: number
  maskBoxExpandRatio?: number

  // 文字样式设置（后端使用驼峰命名）
  fontSize?: number | 'auto'
  autoFontSize?: boolean
  fontFamily?: string
  textDirection?: string
  autoTextDirection?: boolean
  textColor?: string
  fillColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  
  // 智能颜色识别设置
  useAutoTextColor?: boolean  // 是否使用自动识别的文字颜色

  // 修复方式（后端使用 use_lama + lamaModel）
  use_inpainting?: boolean
  use_lama?: boolean
  lamaModel?: string // 'lama_mpe' | 'litelama'

  // 调试选项（后端使用驼峰命名 showDetectionDebug）
  showDetectionDebug?: boolean

  // 特殊模式
  remove_only?: boolean // 仅消除文字，不翻译
  skip_translation?: boolean // 跳过翻译（仅消除文字时使用）
  bubble_coords?: BubbleCoords[] // 已有气泡坐标（后端字段名）
  bubble_angles?: number[] // 气泡旋转角度
}

/**
 * 重新渲染请求参数（匹配后端API格式）
 */
export interface ReRenderParams {
  // 图片数据
  clean_image: string // 干净背景 Base64
  image?: string // 当前图片 Base64（可选）

  // 气泡数据（必需）
  bubble_texts: string[] // 文本数组
  bubble_coords: BubbleCoords[] // 坐标数组
  bubble_states?: Array<{
    translatedText?: string
    coords?: BubbleCoords
    fontSize?: number
    fontFamily?: string
    textDirection?: string
    textColor?: string
    rotationAngle?: number
    position?: { x: number; y: number }
    strokeEnabled?: boolean
    strokeColor?: string
    strokeWidth?: number
  }>

  // 全局样式设置
  fontSize?: number
  fontFamily?: string
  textDirection?: string
  textColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number

  // 控制选项
  use_individual_styles?: boolean
  use_inpainting?: boolean
  use_lama?: boolean
  is_font_style_change?: boolean
  autoFontSize?: boolean
}

/**
 * 单气泡重新渲染请求参数
 */
export interface ReRenderSingleBubbleParams {
  original_image: string
  clean_image: string
  bubble_state: BubbleState
  bubble_index: number
  current_translated_image: string
}

/**
 * 高质量翻译请求参数
 * 与原版 high_quality_translation.js 的 hqTranslateBatchApi 参数格式保持一致
 * 后端期望接收 messages 而不是 images
 */
export interface HqTranslateParams {
  // 服务设置
  provider: string
  api_key: string
  model_name: string  // 后端期望 model_name 而不是 model
  custom_base_url?: string

  // 消息数组（包含系统提示、用户消息+图片）
  messages: Array<{
    role: 'system' | 'user' | 'assistant'
    content: string | Array<{
      type: 'text' | 'image_url'
      text?: string
      image_url?: {
        url: string
      }
    }>
  }>

  // 高级选项
  low_reasoning?: boolean
  force_json_output?: boolean
  no_thinking_method?: 'gemini' | 'volcano'
  use_stream?: boolean
}

/**
 * 单文本翻译请求参数
 * 字段命名与后端 route_translate_single_text 保持一致
 */
export interface TranslateSingleTextParams {
  original_text: string
  model_provider: string
  api_key?: string
  model_name?: string
  custom_base_url?: string
  target_language: string
  prompt_content?: string
}

// ==================== 翻译 API ====================

/**
 * 翻译图片
 * @param params 翻译参数
 */
export async function translateImage(
  params: TranslateImageParams
): Promise<TranslateImageResponse> {
  return apiClient.post<TranslateImageResponse>('/api/translate_image', params)
}

/**
 * 重新渲染图片
 * @param params 渲染参数
 */
export async function reRenderImage(params: ReRenderParams): Promise<ReRenderResponse> {
  return apiClient.post<ReRenderResponse>('/api/re_render_image', params)
}

/**
 * 重新渲染单个气泡
 * @param params 渲染参数
 */
export async function reRenderSingleBubble(
  params: ReRenderSingleBubbleParams
): Promise<ReRenderResponse> {
  return apiClient.post<ReRenderResponse>('/api/re_render_single_bubble', params)
}

/**
 * 应用设置到所有图片
 * @param images 图片数据数组
 * @param settings 要应用的设置
 */
export async function applySettingsToAllImages(
  images: Array<{
    original_image: string
    clean_image: string
    bubble_states: BubbleState[]
  }>,
  settings: {
    font_size?: number
    font_family?: string
    text_direction?: string
    text_color?: string
    fill_color?: string
    stroke_enabled?: boolean
    stroke_color?: string
    stroke_width?: number
  }
): Promise<ApiResponse<{ translated_images: string[] }>> {
  return apiClient.post('/api/apply_settings_to_all_images', {
    images,
    settings,
  })
}

/**
 * 单文本翻译
 * @param params 翻译参数
 */
export async function translateSingleText(
  params: TranslateSingleTextParams
): Promise<ApiResponse<{ translated_text: string }>> {
  try {
    const result = await apiClient.post<{ translated_text: string }>('/api/translate_single_text', params)
    return {
      success: true,
      data: result
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '翻译失败'
    }
  }
}

/**
 * 高质量翻译（批量上下文翻译）
 * @param params 翻译参数
 */
export async function hqTranslateBatch(params: HqTranslateParams): Promise<HqTranslateResponse> {
  return apiClient.post<HqTranslateResponse>('/api/hq_translate_batch', params)
}

// ==================== 气泡检测 API ====================

/**
 * 仅检测气泡框（不进行翻译）
 * @param image Base64 图片数据
 * @param detectorType 文字检测器类型（后端字段名 detector_type）
 * @param boxExpandParams 文本框扩展参数
 */
export async function detectBoxes(
  image: string,
  detectorType?: string,
  boxExpandParams?: {
    box_expand_ratio?: number
    box_expand_top?: number
    box_expand_bottom?: number
    box_expand_left?: number
    box_expand_right?: number
  }
): Promise<DetectBoxesResponse> {
  return apiClient.post<DetectBoxesResponse>('/api/detect_boxes', {
    image,
    detector_type: detectorType,
    ...boxExpandParams,
  })
}

/**
 * 单气泡 OCR 重新识别
 * @param imageData Base64 图片数据
 * @param bubbleCoords 气泡坐标
 * @param ocrEngine OCR 引擎
 * @param ocrParams OCR 参数
 */
export async function ocrSingleBubble(
  imageData: string,
  bubbleCoords: BubbleCoords,
  ocrEngine: string,
  ocrParams?: {
    source_language?: string
    // 百度 OCR 参数
    baidu_ocr_api_key?: string
    baidu_ocr_secret_key?: string
    baidu_version?: string
    baidu_source_language?: string
    // AI 视觉 OCR 参数
    ai_vision_provider?: string
    ai_vision_api_key?: string
    ai_vision_model_name?: string
    ai_vision_ocr_prompt?: string
    custom_ai_vision_base_url?: string
  }
): Promise<OcrSingleBubbleResponse> {
  return apiClient.post<OcrSingleBubbleResponse>('/api/ocr_single_bubble', {
    image_data: imageData,
    bubble_coords: bubbleCoords,
    ocr_engine: ocrEngine,
    ...ocrParams,
  })
}

/**
 * 单气泡背景修复选项
 */
export interface InpaintSingleBubbleOptions {
  /** 气泡旋转角度（度） */
  bubbleAngle?: number
  /** 修复方法 */
  method?: 'lama'
  /** LAMA 模型类型 */
  lamaModel?: 'lama_mpe' | 'litelama'
  /** 笔刷精确掩膜 Base64（可选，用于笔刷修复模式） */
  maskData?: string
}

/**
 * 单气泡背景修复（LAMA 修复）
 * @param image Base64 图片数据
 * @param bubbleCoords 气泡坐标 [x1, y1, x2, y2]
 * @param options 修复选项（包含角度、模型类型等）
 */
export async function inpaintSingleBubble(
  image: string,
  bubbleCoords: BubbleCoords,
  options?: InpaintSingleBubbleOptions
): Promise<InpaintSingleBubbleResponse> {
  return apiClient.post<InpaintSingleBubbleResponse>('/api/inpaint_single_bubble', {
    image_data: image,
    bubble_coords: bubbleCoords,
    bubble_angle: options?.bubbleAngle ?? 0,
    method: options?.method ?? 'lama',
    lama_model: options?.lamaModel ?? 'lama_mpe',
    ...(options?.maskData && { mask_data: options.maskData }),
  })
}
