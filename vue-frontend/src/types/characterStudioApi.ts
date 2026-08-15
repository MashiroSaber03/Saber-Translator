import type {
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioGreetingOption,
} from './characterStudioChat'

export interface CharacterStudioCandidate {
  id: string
  name: string
  aliases: string[]
  first_appearance_page: number | null
  key_moment_count: number
  related_page_count: number
  related_page_numbers: number[]
}

export interface CharacterStudioSummary {
  id: string
  title: string
  origin: 'analysis' | 'manual' | 'imported'
  source_character?: string | null
  updated_at: string
  tags: string[]
  is_favorite: boolean
  has_avatar: boolean
}

export interface CharacterStudioChatState {
  doc_id: string
  index_revision: number
  active_session: CharacterStudioChatSession | null
  archived_sessions: CharacterStudioChatSessionSummary[]
  available_greetings: CharacterStudioGreetingOption[]
}

export interface ExportDiagnostic {
  valid: boolean
  errors: string[]
  warnings: string[]
  checks: {
    document: boolean
    v3_export: boolean
    v2_export: boolean
  }
}

export interface CharacterStudioIndex {
  book_id: string
  documents: CharacterStudioSummary[]
  candidates: CharacterStudioCandidate[]
  count: number
  has_timeline: boolean
}
