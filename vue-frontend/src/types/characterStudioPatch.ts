import type { LorebookEntryNode, RegexScript, StateTask } from './characterStudioDocument'

export interface CharacterStudioPatchUpdateOp<TChanges extends Record<string, unknown>> {
  id: string
  changes: TChanges
}

export interface CharacterStudioPatchDeleteOp {
  id: string
}

export type CharacterStudioWorldbookAddPayload = Partial<Omit<LorebookEntryNode, 'id' | 'children'>> & {
  children?: LorebookEntryNode[]
}

export type CharacterStudioWorldbookChanges = Partial<Pick<
  LorebookEntryNode,
  | 'comment'
  | 'keys'
  | 'secondary_keys'
  | 'content'
  | 'enabled'
  | 'constant'
  | 'selective'
  | 'priority'
  | 'position'
  | 'depth'
  | 'probability'
  | 'prevent_recursion'
  | 'use_regex'
  | 'match_persona_description'
  | 'match_character_description'
  | 'match_character_personality'
  | 'match_character_depth_prompt'
  | 'match_scenario'
>>

export type CharacterStudioRegexAddPayload = Partial<Omit<RegexScript, 'id'>>

export type CharacterStudioRegexChanges = Partial<Pick<
  RegexScript,
  'scriptName' | 'findRegex' | 'replaceString' | 'placement' | 'markdownOnly' | 'promptOnly' | 'runOnEdit' | 'disabled'
>>

export type CharacterStudioTaskAddPayload = Partial<Omit<StateTask, 'id'>>

export type CharacterStudioTaskChanges = Partial<Pick<
  StateTask,
  'name' | 'triggerTiming' | 'interval' | 'commands' | 'disabled'
>>

export interface CharacterStudioAgentPatchV2 {
  set?: Record<string, unknown>
  greeting_add?: string | string[]
  worldbook_add?: CharacterStudioWorldbookAddPayload | CharacterStudioWorldbookAddPayload[]
  worldbook_update?:
    | CharacterStudioPatchUpdateOp<CharacterStudioWorldbookChanges>
    | Array<CharacterStudioPatchUpdateOp<CharacterStudioWorldbookChanges>>
  worldbook_delete?: CharacterStudioPatchDeleteOp | CharacterStudioPatchDeleteOp[]
  regex_add?: CharacterStudioRegexAddPayload | CharacterStudioRegexAddPayload[]
  regex_update?:
    | CharacterStudioPatchUpdateOp<CharacterStudioRegexChanges>
    | Array<CharacterStudioPatchUpdateOp<CharacterStudioRegexChanges>>
  regex_delete?: CharacterStudioPatchDeleteOp | CharacterStudioPatchDeleteOp[]
  task_add?: CharacterStudioTaskAddPayload | CharacterStudioTaskAddPayload[]
  task_update?:
    | CharacterStudioPatchUpdateOp<CharacterStudioTaskChanges>
    | Array<CharacterStudioPatchUpdateOp<CharacterStudioTaskChanges>>
  task_delete?: CharacterStudioPatchDeleteOp | CharacterStudioPatchDeleteOp[]
}
