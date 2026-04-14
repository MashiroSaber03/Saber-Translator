/**
 * 角色卡工坊类型定义
 */

export interface CharacterCandidate {
  name: string
  aliases: string[]
  first_appearance: number
  description: string
  arc: string
  dialogue_count: number
  has_dialogues: boolean
  sample_pages: number[]
}

export interface CharacterBookEntry {
  uid: number
  key: string[]
  keysecondary: string[]
  comment: string
  content: string
  constant: boolean
  selective: boolean
  insertion_order: number
  enabled: boolean
  position: string
  extensions: Record<string, unknown>
}

export interface CharacterBookSchema {
  name: string
  description: string
  scan_depth: number
  token_budget: number
  recursive_scanning: boolean
  extensions: Record<string, unknown>
  entries: CharacterBookEntry[]
}

export interface RegexProfile {
  id: string
  name: string
  enabled: boolean
  scope: string
  source: string
  pattern: string
  replacement: string
  flags: string
  depth_min: number
  depth_max: number
  order: number
  notes: string
}

export interface MvuVariable {
  name: string
  type: string
  scope: string
  default: unknown
  value: unknown
  validator: Record<string, unknown>
  description: string
}

export interface HelperUiManifest {
  layout: string
  theme: string
  panels: Array<Record<string, unknown>>
  widgets: Array<Record<string, unknown>>
  actions: Array<Record<string, unknown>>
  events: Array<Record<string, unknown>>
  bindings: Array<Record<string, unknown>>
}

export interface SaberTavernExtension {
  regex_profiles: RegexProfile[]
  mvu: {
    version: string
    variables: MvuVariable[]
  }
  ui_manifest: HelperUiManifest
  import_manifest: Record<string, unknown>
  source?: Record<string, unknown>
}

export interface CharacterCardV2Data {
  name: string
  description: string
  personality: string
  scenario: string
  first_mes: string
  mes_example: string
  creator_notes: string
  system_prompt: string
  post_history_instructions: string
  alternate_greetings: string[]
  tags: string[]
  creator: string
  character_version: string
  character_book: CharacterBookSchema
  extensions: {
    saber_tavern?: SaberTavernExtension
    [key: string]: unknown
  }
}

export interface CharacterCardV2 {
  spec: string
  spec_version: string
  data: CharacterCardV2Data
}

export interface CharacterCardDraftItem {
  character: string
  card: CharacterCardV2
  source_stats: Record<string, unknown>
}

export interface CharacterCardDraftPayload {
  book_id: string
  style: string
  generated_at?: string
  saved_at?: string
  cards: CharacterCardDraftItem[]
  missing_characters?: string[]
}

export interface CharacterCandidatesResponse {
  success: boolean
  book_id?: string
  candidates?: CharacterCandidate[]
  count?: number
  generated_at?: string
  error?: string
}

export interface CharacterCardDraftResponse {
  success: boolean
  draft?: CharacterCardDraftPayload | null
  has_data?: boolean
  error?: string
  message?: string
}

export interface CharacterCardCompileResponse {
  success: boolean
  valid?: boolean
  errors?: string[]
  warnings?: string[]
  compiled_cards?: Record<string, CharacterCardV2>
  compatibility_reports?: Record<string, CharacterCardCompatibilityReport>
  compiled_count?: number
  compiled_at?: string
  error?: string
  message?: string
}

export interface CharacterCardCompatibilityReport {
  compatible: boolean
  core_ready: boolean
  helper_ready: boolean
  checks: Record<string, boolean>
  errors: string[]
  warnings: string[]
}

export interface CharacterCardCompatResponse {
  success: boolean
  book_id?: string
  character?: string
  report?: CharacterCardCompatibilityReport
  checked_at?: string
  error?: string
}
