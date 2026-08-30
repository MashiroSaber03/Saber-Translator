import type { OpenAICompatibleOptions } from '@/types/settings'

export interface TranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  openaiOptions?: OpenAICompatibleOptions
  translationMode?: 'batch' | 'single'
}

export interface HqTranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  batchSize?: number
  openaiOptions?: OpenAICompatibleOptions
  prompt?: string
}

export interface PluginAgentProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  openaiOptions?: OpenAICompatibleOptions
}

export interface AiVisionOcrProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  prompt?: string
  promptMode?: 'normal' | 'json' | 'paddleocr_vl'
  openaiOptions?: OpenAICompatibleOptions
  minImageSize?: number
}

export interface ProviderConfigsCache {
  translation: Record<string, TranslationProviderConfig>
  hqTranslation: Record<string, HqTranslationProviderConfig>
  pluginAgent: Record<string, PluginAgentProviderConfig>
  browserDomAgent: Record<string, PluginAgentProviderConfig>
  aiVisionOcr: Record<string, AiVisionOcrProviderConfig>
}
