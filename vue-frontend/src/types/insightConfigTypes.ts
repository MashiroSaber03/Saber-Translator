export interface InsightOpenAIOptions {
  request: {
    force_json_output: boolean
    temperature?: number
    extra_body?: Record<string, unknown>
  }
  execution: {
    use_stream: boolean
    rpm_limit: number
    transport_retries: number
    business_retries: number
  }
}

export interface VlmConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  openai_options?: InsightOpenAIOptions
  image_max_size?: number
}

export interface LlmConfig {
  use_same_as_vlm: boolean
  provider?: string
  api_key?: string
  model?: string
  base_url?: string
  openai_options?: InsightOpenAIOptions
}

export interface EmbeddingConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  rpm_limit?: number
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface RerankerConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  top_k?: number
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface ImageGenConfig {
  provider: string
  api_key: string
  model: string
  base_url?: string
  transport_retries?: number
  business_retries?: number
  timeout_seconds?: number
}

export interface LayerConfig {
  name: string
  units_per_group: number
  align_to_chapter: boolean
}

export interface BatchAnalysisConfig {
  pages_per_batch: number
  context_batch_count: number
  architecture_preset: string
  custom_layers?: LayerConfig[]
}

export interface AnalysisSettings {
  depth?: string
  auto_analyze_new_chapters?: boolean
  save_intermediate_results?: boolean
  batch?: BatchAnalysisConfig
}

export interface PromptsConfig {
  batch_analysis?: string
  segment_summary?: string
  chapter_summary?: string
  book_overview?: string
  group_summary?: string
  qa_response?: string
  question_decompose?: string
  analysis_system?: string
}

export interface InsightConfig {
  vlm?: VlmConfig
  chat_llm?: LlmConfig
  embedding?: EmbeddingConfig
  reranker?: RerankerConfig
  image_gen?: ImageGenConfig
  analysis?: AnalysisSettings
  prompts?: PromptsConfig
  provider_settings?: Record<string, Record<string, Record<string, unknown>>>
}
