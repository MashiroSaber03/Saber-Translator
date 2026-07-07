import { apiClient } from './client'
import type {
  ApiResponse,
  BubbleCoords,
  BubbleTextline,
  GlossaryExtractionResponse,
  HqTranslateResponse,
  InpaintSingleBubbleResponse,
  OcrSingleBubbleResponse,
  ReRenderResponse,
} from '@/types'
import type {
  GlossarySettings,
  NonTranslateSettings,
  TranslationWarning,
} from '@/types/translationConstraints'
import type { OpenAICompatibleOptionsWire } from '@/utils/openaiOptions'

const TRANSLATE_ENDPOINTS = {
  reRenderImage: '/api/re_render_image',
  singleText: '/api/translate_single_text',
  hqBatch: '/api/hq_translate_batch',
  glossaryExtract: '/api/translation/glossary/extract',
  ocrSingleBubble: '/api/ocr_single_bubble',
  inpaintSingleBubble: '/api/inpaint_single_bubble',
} as const

export interface ReRenderParams {
  clean_image: string
  image?: string
  translation_mode?: string
  translation_scope?: string
  bubble_texts: string[]
  bubble_coords: BubbleCoords[]
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
    lineSpacing?: number
    textAlign?: 'start' | 'center' | 'end'
  }>
  fontSize?: number
  fontFamily?: string
  textDirection?: string
  textColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  lineSpacing?: number
  textAlign?: 'start' | 'center' | 'end'
  use_individual_styles?: boolean
  use_inpainting?: boolean
  use_lama?: boolean
  is_font_style_change?: boolean
  autoFontSize?: boolean
}

export interface HqTranslateParams {
  provider: string
  api_key: string
  model_name: string
  custom_base_url?: string
  translation_mode?: string
  translation_scope?: string
  jsonData: HqTranslateJsonData[]
  imageBase64Array: string[]
  target_language?: string
  prompt?: string
  systemPrompt?: string
  isProofreading?: boolean
  enableDebugLogs?: boolean
  glossary_settings?: GlossarySettings
  non_translate_settings?: NonTranslateSettings
  openai_options?: OpenAICompatibleOptionsWire
}

export interface HqTranslateJsonData {
  imageIndex: number
  bubbles: Array<{
    bubbleIndex: number
    original: string
    translated: string
    textDirection: string
  }>
}

export interface TranslateSingleTextParams {
  original_text: string
  translation_mode?: string
  translation_scope?: string
  model_provider: string
  api_key?: string
  model_name?: string
  custom_base_url?: string
  target_language: string
  prompt_content?: string
  glossary_settings?: GlossarySettings
  non_translate_settings?: NonTranslateSettings
  openai_options?: OpenAICompatibleOptionsWire
}

export interface ExtractGlossaryEntriesParams {
  original_texts: string[]
  target_language: string
  model_provider: string
  api_key?: string
  model_name?: string
  custom_base_url?: string
  prompt?: string
  existing_entries?: GlossarySettings['entries']
  openai_options?: OpenAICompatibleOptionsWire
}

export interface OcrSingleBubbleOptions {
  source_language?: string
  baidu_ocr_api_key?: string
  baidu_ocr_secret_key?: string
  baidu_version?: string
  baidu_source_language?: string
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
  bubble_textlines?: BubbleTextline[]
  text_detector?: string
  enable_aux_yolo_detection?: boolean
  aux_yolo_conf_threshold?: number
  aux_yolo_overlap_threshold?: number
  enable_saber_yolo_refine?: boolean
  saber_yolo_refine_overlap_threshold?: number
}

export interface InpaintSingleBubbleOptions {
  bubbleAngle?: number
  method?: 'lama'
  lamaModel?: 'lama_mpe' | 'litelama'
  maskData?: string
}

export async function reRenderImage(params: ReRenderParams): Promise<ReRenderResponse> {
  return apiClient.post<ReRenderResponse>(TRANSLATE_ENDPOINTS.reRenderImage, params)
}

export async function translateSingleText(
  params: TranslateSingleTextParams,
): Promise<ApiResponse<{ translated_text: string; warnings?: TranslationWarning[] }>> {
  try {
    const result = await apiClient.post<{
      translated_text: string
      warnings?: TranslationWarning[]
    }>(TRANSLATE_ENDPOINTS.singleText, params)
    return {
      success: true,
      data: result,
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '翻译失败',
    }
  }
}

export async function hqTranslateBatch(params: HqTranslateParams): Promise<HqTranslateResponse> {
  return apiClient.post<HqTranslateResponse>(TRANSLATE_ENDPOINTS.hqBatch, params)
}

export async function extractGlossaryEntries(
  params: ExtractGlossaryEntriesParams,
): Promise<GlossaryExtractionResponse> {
  return apiClient.post<GlossaryExtractionResponse>(TRANSLATE_ENDPOINTS.glossaryExtract, params)
}

export async function ocrSingleBubble(
  imageData: string,
  bubbleCoords: BubbleCoords,
  ocrEngine: string,
  ocrParams?: OcrSingleBubbleOptions,
): Promise<OcrSingleBubbleResponse> {
  return apiClient.post<OcrSingleBubbleResponse>(TRANSLATE_ENDPOINTS.ocrSingleBubble, {
    image_data: imageData,
    bubble_coords: bubbleCoords,
    ocr_engine: ocrEngine,
    ...ocrParams,
  })
}

export async function inpaintSingleBubble(
  image: string,
  bubbleCoords: BubbleCoords,
  options?: InpaintSingleBubbleOptions,
): Promise<InpaintSingleBubbleResponse> {
  return apiClient.post<InpaintSingleBubbleResponse>(TRANSLATE_ENDPOINTS.inpaintSingleBubble, {
    image_data: image,
    bubble_coords: bubbleCoords,
    bubble_angle: options?.bubbleAngle ?? 0,
    method: options?.method ?? 'lama',
    lama_model: options?.lamaModel ?? 'lama_mpe',
    ...(options?.maskData && { mask_data: options.maskData }),
  })
}
