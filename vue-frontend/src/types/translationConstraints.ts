/**
 * 术语表 / 禁翻表相关类型
 */

export type TranslationConstraintMatchMode = 'text' | 'regex'

export interface GlossaryEntry {
  source: string
  target: string
  note: string
  matchMode: TranslationConstraintMatchMode
}

export interface GlossarySettings {
  enabled: boolean
  autoExtractEnabled: boolean
  entries: GlossaryEntry[]
}

export interface NonTranslateEntry {
  pattern: string
  note: string
  matchMode: TranslationConstraintMatchMode
}

export interface NonTranslateSettings {
  enabled: boolean
  entries: NonTranslateEntry[]
}

export interface TranslationWarning {
  imageIndex?: number
  bubbleIndex?: number
  source: string
  expectedTarget: string
  actualTranslation: string
}

export interface GlossaryExtractionStats {
  added: number
  duplicates: number
  failedPages: number
}
