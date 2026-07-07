import { apiClient } from './client'
import type { BubbleState, BubbleTextline } from '@/types/bubble'
import type { OcrResult } from '@/types/ocr'
import type {
  GlossarySettings,
  NonTranslateSettings,
  TranslationWarning,
} from '@/types/translationConstraints'
import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'

export interface ParallelDetectParams {
  image: string
  translation_mode?: string
  translation_scope?: string
  detector_type?: string
  min_text_block_area_percent?: number
  enable_aux_yolo_detection?: boolean
  aux_yolo_conf_threshold?: number
  aux_yolo_overlap_threshold?: number
  enable_saber_yolo_refine?: boolean
  saber_yolo_refine_overlap_threshold?: number
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
  textlines_per_bubble?: BubbleTextline[][]
  error?: string
}

export interface ParallelOcrParams {
  image: string
  bubble_coords: number[][]
  translation_mode?: string
  translation_scope?: string
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
  ai_vision_prompt_mode?: 'normal' | 'json' | 'paddleocr_vl'
  custom_ai_vision_base_url?: string
  openai_options?: OpenAICompatibleOptionsWire
  ai_vision_min_image_size?: number
  enable_hybrid_ocr?: boolean
  secondary_ocr_engine?: string
  hybrid_ocr_threshold?: number
  textlines_per_bubble?: BubbleTextline[][]
}

export interface ParallelOcrResponse {
  success: boolean
  original_texts?: string[]
  ocr_results?: OcrResult[]
  textlines_per_bubble?: BubbleTextline[][]
  error?: string
}

export interface ParallelColorParams {
  image: string
  bubble_coords: number[][]
  translation_mode?: string
  translation_scope?: string
  textlines_per_bubble?: BubbleTextline[][]
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

export interface ParallelTranslateParams {
  original_texts: string[]
  translation_mode?: string
  translation_scope?: string
  target_language: string
  source_language?: string
  model_provider: string
  model_name?: string
  api_key?: string
  custom_base_url?: string
  prompt_content?: string
  textbox_prompt_content?: string
  use_textbox_prompt?: boolean
  glossary_settings?: GlossarySettings
  non_translate_settings?: NonTranslateSettings
  openai_options?: OpenAICompatibleOptionsWire
}

export interface ParallelTranslateResponse {
  success: boolean
  translated_texts?: string[]
  textbox_texts?: string[]
  warnings?: TranslationWarning[]
  error?: string
}

export interface ParallelInpaintParams {
  image: string
  bubble_coords: number[][]
  translation_mode?: string
  translation_scope?: string
  bubble_polygons?: number[][][]
  raw_mask?: string
  user_mask?: string
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

export interface ParallelRenderParams {
  clean_image: string
  bubble_states: BubbleState[]
  translation_mode?: string
  translation_scope?: string
  fontSize?: number
  fontFamily?: string
  textDirection?: string
  textColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  lineSpacing?: number
  textAlign?: 'start' | 'center' | 'end'
  autoFontSize?: boolean
  use_individual_styles?: boolean
}

export interface ParallelRenderResponse {
  success: boolean
  final_image?: string
  bubble_states?: BubbleState[]
  error?: string
}

function parallelEndpoint(step: string): string {
  return `/api/parallel/${step}`
}

export async function parallelDetect(
  params: ParallelDetectParams,
): Promise<ParallelDetectResponse> {
  return apiClient.post<ParallelDetectResponse>(parallelEndpoint('detect'), params)
}

export async function parallelOcr(params: ParallelOcrParams): Promise<ParallelOcrResponse> {
  return apiClient.post<ParallelOcrResponse>(parallelEndpoint('ocr'), params)
}

export async function parallelColor(params: ParallelColorParams): Promise<ParallelColorResponse> {
  return apiClient.post<ParallelColorResponse>(parallelEndpoint('color'), params)
}

export async function parallelTranslate(
  params: ParallelTranslateParams,
): Promise<ParallelTranslateResponse> {
  return apiClient.post<ParallelTranslateResponse>(parallelEndpoint('translate'), params)
}

export async function parallelInpaint(
  params: ParallelInpaintParams,
): Promise<ParallelInpaintResponse> {
  return apiClient.post<ParallelInpaintResponse>(parallelEndpoint('inpaint'), params)
}

export async function parallelRender(
  params: ParallelRenderParams,
): Promise<ParallelRenderResponse> {
  return apiClient.post<ParallelRenderResponse>(parallelEndpoint('render'), params)
}
