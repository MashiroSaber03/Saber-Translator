export type WebImportAgentProvider = string

export type WebImportEngine = 'auto' | 'gallery-dl' | 'ai-agent'

export type WebImportResolvedEngine = Exclude<WebImportEngine, 'auto'>

export type WebImportStatus =
  | 'idle'
  | 'extracting'
  | 'extracted'
  | 'downloading'
  | 'completed'
  | 'error'

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
  pages: ComicPage[]
  totalPages: number
  engine: WebImportResolvedEngine
}

export interface AgentLog {
  timestamp: string
  type: 'info' | 'tool_call' | 'tool_result' | 'thinking' | 'error'
  message: string
}
