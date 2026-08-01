export type WebImportAgentProvider = string

export type WebImportEngine = 'auto' | 'gallery-dl' | 'ai-agent'

export type WebImportImageFormat = 'jpeg' | 'png' | 'webp' | 'original'

export interface WebImportCompressionSettings {
  enabled: boolean
  quality: number
  maxWidth: number
  maxHeight: number
}

export interface WebImportFormatConvertSettings {
  enabled: boolean
  targetFormat: WebImportImageFormat
}

export interface ImagePreprocessSettings {
  enabled: boolean
  autoRotate: boolean
  compression: WebImportCompressionSettings
  formatConvert: WebImportFormatConvertSettings
}

export interface WebImportFirecrawlSettings {
  apiKey: string
}

export interface WebImportAgentSettings {
  provider: WebImportAgentProvider
  apiKey: string
  customBaseUrl: string
  modelName: string
  useStream: boolean
  forceJsonOutput: boolean
  maxRetries: number
  timeout: number
}

export interface WebImportExtractionSettings {
  prompt: string
  maxIterations: number
}

export interface WebImportDownloadSettings {
  concurrency: number
  timeout: number
  retries: number
  delay: number
  useReferer: boolean
}

export interface WebImportAdvancedSettings {
  customCookie: string
  customHeaders: string
  bypassProxy: boolean
}

export interface WebImportUiSettings {
  showAgentLogs: boolean
  autoImport: boolean
}

export interface WebImportSettings {
  firecrawl: WebImportFirecrawlSettings
  agent: WebImportAgentSettings
  extraction: WebImportExtractionSettings
  download: WebImportDownloadSettings
  imagePreprocess: ImagePreprocessSettings
  advanced: WebImportAdvancedSettings
  ui: WebImportUiSettings
}

export interface WebImportAgentProviderConfig {
  apiKey: string
  modelName: string
  customBaseUrl: string
}

export interface WebImportProviderConfigs {
  agent: Record<string, WebImportAgentProviderConfig>
}

export interface WebImportSettingsPayload {
  webImportSettingsSchemaVersion: number
  settings: WebImportSettings
  providerConfigs: WebImportProviderConfigs
}

export interface ComicPage {
  pageNumber: number
  imageUrl: string
}

export interface ExtractResult {
  success: boolean
  comicTitle: string
  chapterTitle: string
  pages: ComicPage[]
  totalPages: number
  sourceUrl: string
  referer?: string
  engine?: WebImportEngine
  error?: string
}

export interface AgentLog {
  timestamp: string
  type: 'info' | 'tool_call' | 'tool_result' | 'thinking' | 'error'
  message: string
}

export interface WebImportDownloadProgress {
  current: number
  total: number
}

export interface WebImportState {
  status: 'idle' | 'extracting' | 'extracted' | 'downloading' | 'completed' | 'error'
  url: string
  engine: WebImportEngine
  currentEngine: WebImportEngine | null
  referer: string
  logs: AgentLog[]
  extractResult: ExtractResult | null
  selectedPages: Set<number>
  downloadProgress: WebImportDownloadProgress
  error: string | null
}
