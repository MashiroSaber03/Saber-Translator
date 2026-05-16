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

export interface LorebookEntryNode {
  id: string
  comment: string
  keys: string[]
  secondary_keys?: string[]
  content: string
  enabled: boolean
  constant: boolean
  selective: boolean
  priority: number
  position: string
  depth: number
  probability?: number
  prevent_recursion?: boolean
  use_regex?: boolean
  match_persona_description?: boolean
  match_character_description?: boolean
  match_character_personality?: boolean
  match_character_depth_prompt?: boolean
  match_scenario?: boolean
  children: LorebookEntryNode[]
}

export interface RegexScript {
  id: string
  scriptName: string
  findRegex: string
  replaceString: string
  placement: number[]
  markdownOnly: boolean
  promptOnly: boolean
  runOnEdit: boolean
  disabled: boolean
}

export interface StateTask {
  id: string
  name: string
  triggerTiming: string
  interval: number
  commands: string
  disabled: boolean
}

export interface CharacterStudioDocument {
  id: string
  bookId: string
  origin: {
    type: 'analysis' | 'manual' | 'imported'
    source_character?: string | null
    source_pages: number[]
  }
  status: {
    is_favorite: boolean
    frozen_sections: string[]
    last_validated_at?: string | null
  }
  meta: {
    title: string
    tags: string[]
    created_at: string
    updated_at: string
  }
  avatar: {
    mode: string
    asset_path?: string | null
    source_page?: number | null
  }
  identity: {
    name: string
    aliases: string[]
    description: string
    personality: string
    scenario: string
  }
  coreMessages: {
    first_message: string
    message_example: string
    alternate_greetings: string[]
    system_prompt: string
    post_history_instructions: string
    creator_notes: string
    character_version: string
  }
  lorebook: {
    name: string
    entries: LorebookEntryNode[]
  }
  regexScripts: RegexScript[]
  stateTasks: StateTask[]
  chatPreset: {
    opening_mode: string
  }
  grounding: {
    timeline_mode: string
    sample_pages: number[]
    relationships: Array<Record<string, unknown>>
    key_moments: Array<Record<string, unknown>>
  }
  exportArtifacts: Record<string, unknown>
}

export interface CharacterStudioChatAttachment {
  attachment_id: string
  filename: string
  mime_type: string
  asset_path: string
  created_at: string
}

export interface CharacterStudioChatMessage {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  attachments: CharacterStudioChatAttachment[]
  runtime_log: Array<Record<string, unknown>>
  variables_snapshot: Record<string, unknown>
  generation_meta: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface CharacterStudioChatSummaryBlock {
  summary_id: string
  content: string
  created_at: string
  covered_message_ids: string[]
}

export interface CharacterStudioChatSessionSummary {
  session_id: string
  title: string
  message_count: number
  updated_at: string
  archived_at?: string | null
  last_message_excerpt?: string
}

export interface CharacterStudioChatSession {
  session_id: string
  doc_id: string
  title: string
  created_at: string
  updated_at: string
  archived_at?: string | null
  greeting_source?: Record<string, unknown>
  summary_blocks: CharacterStudioChatSummaryBlock[]
  messages: CharacterStudioChatMessage[]
  variables: Record<string, unknown>
  _runtime?: Record<string, unknown>
  last_prompt_preview: string
}

export interface CharacterStudioGreetingOption {
  greeting_id: string
  label: string
  content: string
  source: Record<string, unknown>
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

export interface CharacterStudioEditorPendingState {
  generatingSection: string | null
  validating: boolean
  importingWorldbook: boolean
  deleting: boolean
  saving: boolean
  downloadingFormat: string | null
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
