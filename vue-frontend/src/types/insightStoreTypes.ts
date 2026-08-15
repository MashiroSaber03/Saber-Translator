import type { OpenAICompatibleOptions } from './settings'
import type { components } from '@/api/generated/v2'

export type AnalysisStatus = 'idle' | components['schemas']['JobStatus']

export type OverviewTemplateType =
  | 'no_spoiler'
  | 'story_summary'
  | 'recap'
  | 'character_guide'
  | 'world_setting'
  | 'highlights'
  | 'reading_notes'

export type QAMode = 'precise' | 'global'

export interface ChapterInfo {
  id: string
  title: string
  pageRange?: { start: number; end: number }
  startPage: number
  endPage: number
  analyzed: boolean
  analyzedCount?: number
  summary?: string
}

export interface QAMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  isLoading?: boolean
  mode?: QAMode
  citations?: Array<{ page: number }>
  saved?: boolean
}

export interface BatchConfig {
  pagesPerBatch: number
  contextBatchCount: number
  architecturePreset: string
  customLayers: Array<{ name: string; units: number; align: boolean }>
}

export type StoreOpenAICompatibleOptions = OpenAICompatibleOptions

export interface StoreVlmConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  openaiOptions: StoreOpenAICompatibleOptions
  imageMaxSize: number
}

export interface StoreLlmConfig {
  useSameAsVlm: boolean
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  openaiOptions: StoreOpenAICompatibleOptions
}

export interface StoreEmbeddingConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  rpmLimit: number
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

export interface StoreRerankerConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

export interface StoreImageGenConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

export interface StoreAnalysisProgress {
  current: number
  total: number
  status: AnalysisStatus
  message?: string
}

export interface StoreInsightConfig {
  vlm: StoreVlmConfig
  llm: StoreLlmConfig
  embedding: StoreEmbeddingConfig
  reranker: StoreRerankerConfig
  imageGen: StoreImageGenConfig
  batch: BatchConfig
  prompts: Record<string, string>
}

export type InsightVlmProviderDraft = Omit<StoreVlmConfig, 'provider'>
export type InsightLlmProviderDraft = Omit<StoreLlmConfig, 'provider' | 'useSameAsVlm'>
export type InsightEmbeddingProviderDraft = Omit<StoreEmbeddingConfig, 'provider'>
export type InsightRerankerProviderDraft = Omit<StoreRerankerConfig, 'provider'>
export type InsightImageGenProviderDraft = Omit<StoreImageGenConfig, 'provider'>

export interface InsightProviderDrafts {
  vlm: Record<string, InsightVlmProviderDraft>
  llm: Record<string, InsightLlmProviderDraft>
  embedding: Record<string, InsightEmbeddingProviderDraft>
  reranker: Record<string, InsightRerankerProviderDraft>
  imageGen: Record<string, InsightImageGenProviderDraft>
}

export interface InsightSettingsSnapshot {
  config: StoreInsightConfig
  providerDrafts: InsightProviderDrafts
}
