import type {
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioGreetingOption,
} from './characterStudioChat'

export interface CharacterStudioCandidate {
  name: string
  aliases: string[]
  first_appearance: number
  dialogue_count: number
  has_dialogues: boolean
  sample_pages: number[]
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
  sample_pages: number[]
}

export interface CharacterStudioChatState {
  doc_id: string
  active_session?: CharacterStudioChatSession
  archived_sessions: CharacterStudioChatSessionSummary[]
  available_greetings: CharacterStudioGreetingOption[]
}

export interface ExportDiagnostic {
  valid: boolean
  errors: string[]
  warnings: string[]
  checks: Record<string, boolean>
}

export interface CharacterStudioIndex {
  book_id: string
  documents: CharacterStudioSummary[]
  candidates: CharacterStudioCandidate[]
  count: number
  has_timeline: boolean
}
