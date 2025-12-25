/**
 * 设置类型定义
 * 定义翻译设置、OCR设置、高质量翻译设置等
 */

import type { TextDirection, InpaintMethod } from './bubble'

/**
 * OCR 引擎类型
 */
export type OcrEngine = 'manga_ocr' | 'paddle_ocr' | 'baidu_ocr' | 'ai_vision'

/**
 * 文本检测器类型
 */
export type TextDetector = 'ctd' | 'yolo' | 'yolov5' | 'default'

/**
 * 翻译服务商类型
 */
export type TranslationProvider =
  | 'siliconflow'
  | 'deepseek'
  | 'volcano'
  | 'caiyun'
  | 'baidu'
  | 'baidu_translate'
  | 'youdao'
  | 'youdao_translate'
  | 'gemini'
  | 'ollama'
  | 'sakura'
  | 'custom_openai'

/**
 * 高质量翻译服务商类型
 */
export type HqTranslationProvider =
  | 'siliconflow'
  | 'deepseek'
  | 'volcano'
  | 'gemini'
  | 'custom_openai'

/**
 * 取消思考方法类型
 */
export type NoThinkingMethod = 'gemini' | 'volcano'

/**
 * PDF 处理方式
 */
export type PdfProcessingMethod = 'frontend' | 'backend'

/**
 * 主题类型
 */
export type Theme = 'light' | 'dark'

/**
 * 百度 OCR 设置
 */
export interface BaiduOcrSettings {
  apiKey: string
  secretKey: string
  version: string
  sourceLanguage: string
}

/**
 * AI 视觉 OCR 设置
 */
export interface AiVisionOcrSettings {
  provider: string
  apiKey: string
  modelName: string
  prompt: string
  rpmLimit: number
  customBaseUrl: string
  isJsonMode: boolean
}

/**
 * 翻译服务设置
 */
export interface TranslationServiceSettings {
  provider: TranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  rpmLimit: number
  maxRetries: number
  isJsonMode: boolean
}

/**
 * 高质量翻译设置
 */
export interface HqTranslationSettings {
  provider: HqTranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  batchSize: number
  sessionReset: number
  rpmLimit: number
  maxRetries: number
  lowReasoning: boolean
  noThinkingMethod: NoThinkingMethod
  forceJsonOutput: boolean
  useStream: boolean
  prompt: string
}

/**
 * AI 校对轮次配置
 */
export interface ProofreadingRound {
  name: string
  provider: HqTranslationProvider
  apiKey: string
  modelName: string
  customBaseUrl: string
  batchSize: number
  sessionReset: number
  rpmLimit: number
  lowReasoning: boolean
  noThinkingMethod: NoThinkingMethod
  forceJsonOutput: boolean
  prompt: string
  /** UI状态：是否显示API Key（不持久化） */
  showApiKey?: boolean
}

/**
 * AI 校对设置
 */
export interface ProofreadingSettings {
  enabled: boolean
  rounds: ProofreadingRound[]
  maxRetries: number
}

/**
 * 文本框扩展参数
 */
export interface BoxExpandSettings {
  ratio: number
  top: number
  bottom: number
  left: number
  right: number
}

/**
 * 精确文字掩膜设置
 */
export interface PreciseMaskSettings {
  enabled: boolean
  dilateSize: number
  boxExpandRatio: number
}

/**
 * 文字样式设置
 */
export interface TextStyleSettings {
  fontSize: number
  autoFontSize: boolean
  fontFamily: string
  layoutDirection: TextDirection
  textColor: string
  fillColor: string
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  inpaintMethod: InpaintMethod
}

/**
 * 完整的翻译设置
 */
export interface TranslationSettings {
  // 文字样式设置
  textStyle: TextStyleSettings

  // OCR 设置
  ocrEngine: OcrEngine
  sourceLanguage: string
  textDetector: TextDetector
  baiduOcr: BaiduOcrSettings
  aiVisionOcr: AiVisionOcrSettings

  // 翻译服务设置
  translation: TranslationServiceSettings
  targetLanguage: string
  translatePrompt: string
  useTextboxPrompt: boolean
  textboxPrompt: string

  // 高质量翻译设置
  hqTranslation: HqTranslationSettings

  // AI 校对设置
  proofreading: ProofreadingSettings

  // 文本框扩展参数
  boxExpand: BoxExpandSettings

  // 精确文字掩膜设置
  preciseMask: PreciseMaskSettings

  // PDF 处理方式
  pdfProcessingMethod: PdfProcessingMethod

  // 调试选项
  showDetectionDebug: boolean
}

/**
 * 设置更新参数
 */
export type TranslationSettingsUpdates = Partial<TranslationSettings>
