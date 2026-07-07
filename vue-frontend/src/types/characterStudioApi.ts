import type {
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioGreetingOption,
} from './characterStudioChat'
import type { CharacterStudioDocument } from './characterStudioDocument'

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

export interface CharacterStudioChatStateResponse {
  success: boolean
  doc_id?: string
  active_session?: CharacterStudioChatSession
  archived_sessions?: CharacterStudioChatSessionSummary[]
  available_greetings?: CharacterStudioGreetingOption[]
  session?: CharacterStudioChatSession
  prompt_preview?: string
  error?: string
  message?: string
}

export interface ExportDiagnostic {
  valid: boolean
  errors: string[]
  warnings: string[]
  checks: Record<string, boolean>
}

export interface CharacterStudioReviewReport {
  summary: string
  issues: string[]
  suggestions: string[]
  generated_at?: string
}

export interface CardAgentResponse {
  content: string
  context: string
}

export interface CharacterStudioIndexResponse {
  success: boolean
  book_id?: string
  documents?: CharacterStudioSummary[]
  candidates?: CharacterStudioCandidate[]
  count?: number
  has_timeline?: boolean
  error?: string
  message?: string
}

export interface CharacterStudioDocumentResponse {
  success: boolean
  document?: CharacterStudioDocument
  error?: string
  message?: string
}
