import type { OpenAICompatibleOptions } from '@/types/settings'

export type ProviderOpenAICompatibleOptions = {
  request?: Partial<OpenAICompatibleOptions['request']>
  execution?: Partial<OpenAICompatibleOptions['execution']>
}

export interface TranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  openaiOptions?: ProviderOpenAICompatibleOptions
  translationMode?: 'batch' | 'single'
}

export interface HqTranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  batchSize?: number
  openaiOptions?: ProviderOpenAICompatibleOptions
  prompt?: string
}

export interface PluginAgentProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  openaiOptions?: ProviderOpenAICompatibleOptions
}

export interface AiVisionOcrProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  prompt?: string
  promptMode?: 'normal' | 'json' | 'paddleocr_vl'
  openaiOptions?: ProviderOpenAICompatibleOptions
  minImageSize?: number
}

export interface ProviderConfigsCache {
  translation: Record<string, TranslationProviderConfig>
  hqTranslation: Record<string, HqTranslationProviderConfig>
  pluginAgent: Record<string, PluginAgentProviderConfig>
  aiVisionOcr: Record<string, AiVisionOcrProviderConfig>
}
