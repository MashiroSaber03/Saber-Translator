/**
 * 设置类型定义
 * 定义翻译设置、OCR设置、高质量翻译设置等
 */

import type { TextDirection, InpaintMethod } from './bubble'

/**
 * OCR 引擎类型
 */
export type OcrEngine = 'manga_ocr' | 'paddle_ocr' | 'paddleocr_vl' | 'baidu_ocr' | 'ai_vision' | '48px_ocr'

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
  | 'baidu_translate'
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
 * 百度 OCR 设置
 */
export interface BaiduOcrSettings {
  apiKey: string
  secretKey: string
  version: string
  sourceLanguage: string
}

/**
 * PaddleOCR-VL 设置
 */
export interface PaddleOcrVlSettings {
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
  /** 最小图片尺寸 (VLM 模型通常要求 >= 28px) */
  minImageSize: number
}

/**
 * 翻译模式类型
 * - batch: 整页批量翻译 (默认)，一次发送全部气泡文本，对模型要求较高
 * - single: 逐气泡翻译，每个气泡单独翻译，对模型要求较低，适合小模型
 */
export type TranslationMode = 'batch' | 'single'

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
  /** 翻译模式：batch=整页批量，single=逐气泡 */
  translationMode: TranslationMode
  /** 批量翻译 - 普通模式提示词 */
  batchNormalPrompt: string
  /** 批量翻译 - JSON模式提示词 */
  batchJsonPrompt: string
  /** 逐气泡翻译 - 普通模式提示词 */
  singleNormalPrompt: string
  /** 逐气泡翻译 - JSON模式提示词 */
  singleJsonPrompt: string
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
  maxRetries?: number  // 最大重试次数（可选，默认使用全局设置）
  lowReasoning: boolean
  noThinkingMethod: NoThinkingMethod
  forceJsonOutput: boolean
  /** 是否使用流式调用 */
  useStream: boolean
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
  // 智能颜色识别设置
  useAutoTextColor: boolean  // 默认使用自动识别的文字颜色
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
  paddleOcrVl: PaddleOcrVlSettings
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

  // 并行翻译设置
  parallel: ParallelSettings

  // 书架模式自动保存（翻译一张保存一张）
  autoSaveInBookshelfMode: boolean

  // 消除文字模式同时执行OCR（获取带原文的干净背景图）
  removeTextWithOcr: boolean

  // 详细日志（全局开关，影响所有翻译模式）
  enableVerboseLogs: boolean
}

/**
 * 并行翻译设置
 */
export interface ParallelSettings {
  enabled: boolean
  deepLearningLockSize: number
}

/**
 * 设置更新参数
 */
export type TranslationSettingsUpdates = Partial<TranslationSettings>
