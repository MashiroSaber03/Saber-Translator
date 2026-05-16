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
  previewState: {
    variables: Record<string, unknown>
    messages: PreviewMessage[]
  }
  grounding: {
    timeline_mode: string
    sample_pages: number[]
    relationships: Array<Record<string, unknown>>
    key_moments: Array<Record<string, unknown>>
  }
  exportArtifacts: Record<string, unknown>
}

export interface PreviewMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface PreviewSessionState {
  doc_id: string
  messages: PreviewMessage[]
  variables: Record<string, unknown>
  log: Array<Record<string, unknown>>
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
  preview_session?: PreviewSessionState
  error?: string
  message?: string
}
