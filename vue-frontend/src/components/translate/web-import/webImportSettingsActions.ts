import type { WebImportSettings } from '@/types/webImport'

export interface WebImportSettingsActions {
  setAgentApiKey: (value: string) => void
  setAgentBaseUrl: (value: string) => void
  setAgentForceJsonOutput: (value: boolean) => void
  setAgentMaxRetries: (value: number) => void
  setAgentModelName: (value: string) => void
  setAgentProvider: (value: string) => void
  setAgentTimeout: (value: number) => void
  setAgentUseStream: (value: boolean) => void
  setAutoImport: (value: boolean) => void
  setBypassProxy: (value: boolean) => void
  setCustomCookie: (value: string) => void
  setCustomHeaders: (value: string) => void
  setDownloadConcurrency: (value: number) => void
  setDownloadDelay: (value: number) => void
  setDownloadRetries: (value: number) => void
  setDownloadTimeout: (value: number) => void
  setDownloadUseReferer: (value: boolean) => void
  setExtractionMaxIterations: (value: number) => void
  setExtractionPrompt: (value: string) => void
  setFirecrawlApiKey: (value: string) => void
  setImageAutoRotate: (value: boolean) => void
  setImageCompressionEnabled: (value: boolean) => void
  setImageCompressionQuality: (value: number) => void
  setImageFormatConvertEnabled: (value: boolean) => void
  setImageMaxHeight: (value: number) => void
  setImageMaxWidth: (value: number) => void
  setImagePreprocessEnabled: (value: boolean) => void
  setImageTargetFormat: (
    value: WebImportSettings['imagePreprocess']['formatConvert']['targetFormat']
  ) => void
  setShowAgentLogs: (value: boolean) => void
}
