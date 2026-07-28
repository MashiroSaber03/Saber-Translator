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
  revision?: number
  avatarUrl?: string | null
  createdAt?: string | null
  updatedAt?: string | null
}
