import type { OpenAICompatibleOptions } from './settings'

export type AnalysisStatus =
  | 'idle'
  | 'queued'
  | 'running'
  | 'pausing'
  | 'paused'
  | 'cancelling'
  | 'interrupted'
  | 'completed'
  | 'completed_with_errors'
  | 'failed'
  | 'cancelled'
  | 'error'

export type AnalysisMode = 'full' | 'chapter' | 'page' | 'chapters' | 'incremental' | 'reanalyze'

export type OverviewTemplateType =
  | 'no_spoiler'
  | 'story_summary'
  | 'recap'
  | 'character_guide'
  | 'world_setting'
  | 'highlights'
  | 'reading_notes'

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
  timestamp: string
  sources?: Array<{ page: number; content: string }>
  isLoading?: boolean
  mode?: string
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
  baseUrl?: string
  openaiOptions: StoreOpenAICompatibleOptions
  imageMaxSize?: number
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
  baseUrl?: string
  rpmLimit?: number
  transportRetries?: number
  businessRetries?: number
  timeoutSeconds?: number
}

export interface StoreRerankerConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  topK?: number
  transportRetries?: number
  businessRetries?: number
  timeoutSeconds?: number
}

export interface StoreImageGenConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl?: string
  transportRetries?: number
  businessRetries?: number
  timeoutSeconds?: number
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
