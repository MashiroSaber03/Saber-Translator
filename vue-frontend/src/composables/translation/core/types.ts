export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

export type ExecutionScope = 'current' | 'all' | 'failed' | 'selection'

export interface PageSelection {
  // User-facing page numbers are 1-based.
  pages: number[]
}

export interface TranslationProgress {
  current: number
  total: number
  completed: number
  failed: number
  isInProgress: boolean
  label?: string
  percentage?: number
}

export interface ProgressReporter {
  init(total: number, label?: string): void
  update(current: number, label?: string): void
  setPercentage(percentage: number, label?: string): void
  incrementCompleted(): void
  incrementFailed(): void
  finish(): void
  getProgress(): TranslationProgress
}

export interface BatchOptions {
  batchSize?: number
  maxRetries?: number
  rpmLimit?: number
}

export interface PipelineConfig {
  mode: TranslationMode
  scope: ExecutionScope
  pageSelection?: PageSelection
  batchOptions?: BatchOptions
}

export interface PipelineResult {
  success: boolean
  completed: number
  failed: number
  errors?: string[]
  autoGlossaryStats?: {
    added: number
    duplicates: number
    failedPages: number
  }
}

export interface SavedTextStyles {
  fontFamily: string
  fontSize: number
  autoFontSize: boolean
  textDirection: string
  autoTextDirection: boolean
  layoutDirection: 'auto' | 'vertical' | 'horizontal'
  fillColor: string
  textColor: string
  rotationAngle: number
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  useAutoTextColor: boolean
  inpaintMethod: 'solid' | 'lama_mpe' | 'litelama'
  lineSpacing: number
  textAlign: 'start' | 'center' | 'end'
}
