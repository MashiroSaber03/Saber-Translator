import type { OpenAICompatibleOptions } from './openaiSettings'
import type { HybridOcrEngine } from './settingsProviders'

export interface BaiduOcrSettings {
  apiKey: string
  secretKey: string
  version: string
  sourceLanguage: string
}

export type PaddleOcrVlLanguage =
  | 'japanese'
  | 'chinese'
  | 'chinese_cht'
  | 'korean'
  | 'english'
  | 'french'
  | 'german'
  | 'spanish'
  | 'italian'
  | 'portuguese'
  | 'dutch'
  | 'polish'
  | 'thai'
  | 'vietnamese'
  | 'indonesian'
  | 'malay'
  | 'russian'
  | 'arabic'
  | 'hindi'
  | 'turkish'
  | 'greek'
  | 'hebrew'

export interface PaddleOcrVlSettings {
  sourceLanguage: PaddleOcrVlLanguage
}

export interface AiVisionOcrSettings {
  provider: string
  apiKey: string
  modelName: string
  prompt: string
  promptMode: 'normal' | 'json' | 'paddleocr_vl'
  customBaseUrl: string
  openaiOptions: OpenAICompatibleOptions
  minImageSize: number
}

export interface HybridOcrSettings {
  enabled: boolean
  secondaryEngine: HybridOcrEngine
  confidenceThreshold: number
}
