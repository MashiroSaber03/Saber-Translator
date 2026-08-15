import type { ExportDiagnostic } from './characterStudioApi'

export type CharacterStudioSection =
  | 'identity'
  | 'greetings'
  | 'lorebook'
  | 'regex'
  | 'state-tasks'

export type CharacterStudioGenerationSection =
  | CharacterStudioSection
  | 'translate'
  | 'full'
  | 'review'

export const CHARACTER_STUDIO_LOREBOOK_REQUIRED_FIELDS = [
  'id',
  'comment',
  'keys',
  'content',
  'enabled',
  'constant',
  'selective',
  'priority',
  'position',
  'depth',
  'children',
] as const

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
    source_character: string | null
  }
  status: {
    is_favorite: boolean
    frozen_sections: CharacterStudioSection[]
    last_diagnostics: ExportDiagnostic | null
    last_validated_at: string | null
  }
  meta: {
    title: string
    tags: string[]
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
  exportArtifacts: Record<string, unknown>
  revision: number
  avatarUrl: string | null
  createdAt: string
  updatedAt: string
}
