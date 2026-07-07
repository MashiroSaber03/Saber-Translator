import type { BubbleState, BubbleTextline } from './bubble'
import type { OcrResult } from './ocr'
import type { TranslationWarning } from './translationConstraints'

export interface ReRenderResponse {
  success: boolean
  translated_image?: string
  rendered_image?: string
  bubble_states?: BubbleState[]
  error?: string
}

export interface OcrSingleBubbleResponse {
  success: boolean
  text?: string
  ocr_result?: OcrResult
  textlines?: BubbleTextline[]
  error?: string
}

export interface InpaintSingleBubbleResponse {
  success: boolean
  inpainted_image?: string
  error?: string
}

export interface HqTranslateResponse {
  success: boolean
  results?: Array<{
    imageIndex: number
    bubbles: Array<{
      bubbleIndex?: number
      original?: string
      translated: string
      textDirection?: string
    }>
  }>
  content?: string
  warning?: string
  warnings?: TranslationWarning[]
  error?: string
}

export interface GlossaryExtractionResponse {
  success: boolean
  new_entries?: Array<{
    source: string
    target: string
    note: string
    matchMode: 'text' | 'regex'
  }>
  candidate_count?: number
  duplicate_count?: number
  error?: string
}
