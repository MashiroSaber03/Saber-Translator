import type { OpenAICompatibleOptions } from './openaiSettings'
import type { HybridOcrEngine } from './settingsProviders'

export interface BaiduOcrSettings {
  apiKey: string
  secretKey: string
  version: string
  sourceLanguage: string
}

export interface PaddleOcrVlSettings {
  sourceLanguage: string
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
