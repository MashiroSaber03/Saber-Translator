/**
 * 并行翻译 API
 * 
 * 为并行流水线提供独立的步骤API调用
 */

import { apiClient } from './client'

// ==================== 检测 API ====================

export interface ParallelDetectParams {
  image: string
  detector_type?: string
  box_expand_ratio?: number
  box_expand_top?: number
  box_expand_bottom?: number
  box_expand_left?: number
  box_expand_right?: number
}

export interface ParallelDetectResponse {
  success: boolean
  bubble_coords?: number[][]
  bubble_angles?: number[]
  bubble_polygons?: number[][][]
  auto_directions?: string[]
  raw_mask?: string
  textlines_per_bubble?: any[]
  error?: string
}

export async function parallelDetect(params: ParallelDetectParams): Promise<ParallelDetectResponse> {
  return apiClient.post<ParallelDetectResponse>('/api/parallel/detect', params)
}

// ==================== OCR API ====================

export interface ParallelOcrParams {
  image: string
  bubble_coords: number[][]
  source_language?: string
  ocr_engine?: string
  baidu_api_key?: string
  baidu_secret_key?: string
  baidu_version?: string
  baidu_ocr_language?: string
  ai_vision_provider?: string
  ai_vision_api_key?: string
  ai_vision_model_name?: string
  ai_vision_ocr_prompt?: string
  custom_ai_vision_base_url?: string
  ai_vision_min_image_size?: number
  textlines_per_bubble?: any[]
}

export interface ParallelOcrResponse {
  success: boolean
  original_texts?: string[]
  textlines_per_bubble?: any[]
  error?: string
}

export async function parallelOcr(params: ParallelOcrParams): Promise<ParallelOcrResponse> {
  return apiClient.post<ParallelOcrResponse>('/api/parallel/ocr', params)
}

// ==================== 颜色提取 API ====================

export interface ParallelColorParams {
  image: string
  bubble_coords: number[][]
  textlines_per_bubble?: any[]
}

export interface ParallelColorResponse {
  success: boolean
  colors?: Array<{
    textColor: string
    bgColor: string
    autoFgColor?: [number, number, number] | null
    autoBgColor?: [number, number, number] | null
  }>
  error?: string
}

export async function parallelColor(params: ParallelColorParams): Promise<ParallelColorResponse> {
  return apiClient.post<ParallelColorResponse>('/api/parallel/color', params)
}

// ==================== 翻译 API ====================

export interface ParallelTranslateParams {
  original_texts: string[]
  target_language: string
  source_language?: string
  model_provider: string
  model_name?: string
  api_key?: string
  custom_base_url?: string
  prompt_content?: string
  textbox_prompt_content?: string
  use_textbox_prompt?: boolean
  rpm_limit?: number
  max_retries?: number
  use_json_format?: boolean
}

export interface ParallelTranslateResponse {
  success: boolean
  translated_texts?: string[]
  textbox_texts?: string[]
  error?: string
}

export async function parallelTranslate(params: ParallelTranslateParams): Promise<ParallelTranslateResponse> {
  return apiClient.post<ParallelTranslateResponse>('/api/parallel/translate', params)
}

// ==================== 修复 API ====================

export interface ParallelInpaintParams {
  image: string
  bubble_coords: number[][]
  bubble_polygons?: number[][][]
  raw_mask?: string       // 文字检测掩膜
  user_mask?: string      // 用户笔刷掩膜（新增）
  method?: string
  lama_model?: string
  fill_color?: string
  mask_dilate_size?: number
  mask_box_expand_ratio?: number
}

export interface ParallelInpaintResponse {
  success: boolean
  clean_image?: string
  error?: string
}

export async function parallelInpaint(params: ParallelInpaintParams): Promise<ParallelInpaintResponse> {
  return apiClient.post<ParallelInpaintResponse>('/api/parallel/inpaint', params)
}

// ==================== 渲染 API ====================

export interface ParallelRenderParams {
  clean_image: string
  bubble_states: any[]
  fontSize?: number
  fontFamily?: string
  textDirection?: string
  textColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  autoFontSize?: boolean
  use_individual_styles?: boolean
}

export interface ParallelRenderResponse {
  success: boolean
  final_image?: string
  bubble_states?: any[]
  error?: string
}

export async function parallelRender(params: ParallelRenderParams): Promise<ParallelRenderResponse> {
  return apiClient.post<ParallelRenderResponse>('/api/parallel/render', params)
}
