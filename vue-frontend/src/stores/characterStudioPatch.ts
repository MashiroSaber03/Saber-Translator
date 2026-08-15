import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioDocument,
  CharacterStudioSection,
  CharacterStudioPatchDeleteOp,
  CharacterStudioPatchUpdateOp,
  CharacterStudioRegexAddPayload,
  CharacterStudioRegexChanges,
  CharacterStudioTaskAddPayload,
  CharacterStudioTaskChanges,
  CharacterStudioWorldbookAddPayload,
  CharacterStudioWorldbookChanges,
  LorebookEntryNode,
  RegexScript,
  StateTask,
} from '@/types/characterStudio'
import {
  CHARACTER_STUDIO_LOREBOOK_REQUIRED_FIELDS,
} from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'

type NormalizedPatch = {
  set: Record<string, unknown>
  greetingAdds: string[]
  worldbookAdds: CharacterStudioWorldbookAddPayload[]
  worldbookUpdates: Array<CharacterStudioPatchUpdateOp<CharacterStudioWorldbookChanges>>
  worldbookDeletes: CharacterStudioPatchDeleteOp[]
  regexAdds: CharacterStudioRegexAddPayload[]
  regexUpdates: Array<CharacterStudioPatchUpdateOp<CharacterStudioRegexChanges>>
  regexDeletes: CharacterStudioPatchDeleteOp[]
  taskAdds: CharacterStudioTaskAddPayload[]
  taskUpdates: Array<CharacterStudioPatchUpdateOp<CharacterStudioTaskChanges>>
  taskDeletes: CharacterStudioPatchDeleteOp[]
}

const WORLD_BOOK_CHANGE_KEYS = new Set<keyof CharacterStudioWorldbookChanges>([
  'comment',
  'keys',
  'secondary_keys',
  'content',
  'enabled',
  'constant',
  'selective',
  'priority',
  'position',
  'depth',
  'probability',
  'prevent_recursion',
  'use_regex',
  'match_persona_description',
  'match_character_description',
  'match_character_personality',
  'match_character_depth_prompt',
  'match_scenario',
])

const REGEX_CHANGE_KEYS = new Set<keyof CharacterStudioRegexChanges>([
  'scriptName',
  'findRegex',
  'replaceString',
  'placement',
  'markdownOnly',
  'promptOnly',
  'runOnEdit',
  'disabled',
])

const TASK_CHANGE_KEYS = new Set<keyof CharacterStudioTaskChanges>([
  'name',
  'triggerTiming',
  'interval',
  'commands',
  'disabled',
])

const WORLD_BOOK_ADD_KEYS = new Set([
  ...WORLD_BOOK_CHANGE_KEYS,
  'children',
])
const LOREBOOK_NODE_KEYS = new Set([...WORLD_BOOK_ADD_KEYS, 'id'])

const ALLOWED_PATCH_KEYS = new Set([
  'set',
  'greeting_add',
  'worldbook_add',
  'worldbook_update',
  'worldbook_delete',
  'regex_add',
  'regex_update',
  'regex_delete',
  'task_add',
  'task_update',
  'task_delete',
])

const VALID_TRIGGER_TIMINGS = new Set(['initialization', 'message_received', 'message_sent'])
const VALID_REGEX_PLACEMENTS = new Set([1, 2])
const VALID_LOREBOOK_POSITIONS = new Set(['before_char', 'at_depth', 'after_char'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function toArray<T>(value: T | T[] | undefined | null): T[] {
  if (value === undefined || value === null) return []
  return Array.isArray(value) ? value : [value]
}

function ensureRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${label} 必须为对象`)
  }
  return value
}

function ensureAllowedKeys(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  label: string,
) {
  const unknown = Object.keys(value).find(key => !allowed.has(key))
  if (unknown) throw new Error(`${label} 不支持字段: ${unknown}`)
}

function ensureString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} 缺少有效字符串`)
  }
  return value.trim()
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label} 必须为字符串`)
  return value
}

function normalizeStringList(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} 必须为字符串数组`)
  }
  return value.map((item, index) => {
    if (typeof item !== 'string') {
      throw new Error(`${label}[${index}] 必须为字符串`)
    }
    return item.trim()
  })
}

function normalizePlacement(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) throw new Error(`${label} 必须为整数数组`)
  const normalized = value.map((item, index) => {
    if (!Number.isInteger(item)) {
      throw new Error(`${label}[${index}] 必须为整数`)
    }
    if (!VALID_REGEX_PLACEMENTS.has(item as number)) {
      throw new Error(`${label}[${index}] 只能使用 1 或 2`)
    }
    return item as number
  })
  if (normalized.length === 0) {
    throw new Error(`${label} 不能为空`)
  }
  return [...new Set(normalized)]
}

function normalizeBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} 必须为布尔值`)
  }
  return value
}

function normalizeInteger(
  value: unknown,
  label: string,
  { minimum, maximum }: { minimum?: number; maximum?: number } = {},
): number {
  if (!Number.isInteger(value)) throw new Error(`${label} 必须为整数`)
  const result = value as number
  if (minimum !== undefined && result < minimum) {
    throw new Error(`${label} 不能小于 ${minimum}`)
  }
  if (maximum !== undefined && result > maximum) {
    throw new Error(`${label} 不能大于 ${maximum}`)
  }
  return result
}

function createGeneratedId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
}

function rootSectionToFrozenKey(section: string): CharacterStudioSection | null {
  if (section === 'identity' || section === 'meta') return 'identity'
  if (section === 'coreMessages') return 'greetings'
  if (section === 'lorebook') return 'lorebook'
  if (section === 'regexScripts') return 'regex'
  if (section === 'stateTasks') return 'state-tasks'
  return null
}

function normalizeLorebookPosition(value: unknown, label: string): string {
  const next = stringValue(value, label).trim()
  if (!VALID_LOREBOOK_POSITIONS.has(next)) {
    throw new Error(`${label} 只能使用 before_char、at_depth、after_char`)
  }
  return next
}

function normalizeLorebookNode(value: unknown, label: string): LorebookEntryNode {
  const node = ensureRecord(value, label)
  ensureAllowedKeys(node, LOREBOOK_NODE_KEYS, label)
  for (const field of CHARACTER_STUDIO_LOREBOOK_REQUIRED_FIELDS) {
    if (node[field] === undefined) throw new Error(`${label}.${field} 缺失`)
  }
  if (!Array.isArray(node.children)) throw new Error(`${label}.children 必须为数组`)
  const result: LorebookEntryNode = {
    id: ensureString(node.id, `${label}.id`),
    comment: stringValue(node.comment, `${label}.comment`),
    keys: normalizeStringList(node.keys, `${label}.keys`),
    content: stringValue(node.content, `${label}.content`),
    enabled: normalizeBoolean(node.enabled, `${label}.enabled`),
    constant: normalizeBoolean(node.constant, `${label}.constant`),
    selective: normalizeBoolean(node.selective, `${label}.selective`),
    priority: normalizeInteger(node.priority, `${label}.priority`),
    position: normalizeLorebookPosition(node.position, `${label}.position`),
    depth: normalizeInteger(node.depth, `${label}.depth`, { minimum: 0 }),
    children: node.children.map((child, index) =>
      normalizeLorebookNode(child, `${label}.children[${index}]`)
    ),
  }
  if (node.secondary_keys !== undefined) {
    result.secondary_keys = normalizeStringList(node.secondary_keys, `${label}.secondary_keys`)
  }
  if (node.probability !== undefined) {
    result.probability = normalizeInteger(node.probability, `${label}.probability`, {
      minimum: 0,
      maximum: 100,
    })
  }
  for (const field of [
    'prevent_recursion',
    'use_regex',
    'match_persona_description',
    'match_character_description',
    'match_character_personality',
    'match_character_depth_prompt',
    'match_scenario',
  ] as const) {
    if (node[field] !== undefined) {
      result[field] = normalizeBoolean(node[field], `${label}.${field}`)
    }
  }
  return result
}

function normalizeWorldbookAddPayload(value: unknown): CharacterStudioWorldbookAddPayload {
  const payload = ensureRecord(value, 'worldbook_add')
  ensureAllowedKeys(payload, WORLD_BOOK_ADD_KEYS, 'worldbook_add')
  const normalized: CharacterStudioWorldbookAddPayload = {}
  if (payload.comment !== undefined) normalized.comment = stringValue(payload.comment, 'worldbook_add.comment')
  if (payload.keys !== undefined) normalized.keys = normalizeStringList(payload.keys, 'worldbook_add.keys')
  if (payload.secondary_keys !== undefined) normalized.secondary_keys = normalizeStringList(payload.secondary_keys, 'worldbook_add.secondary_keys')
  if (payload.content !== undefined) normalized.content = stringValue(payload.content, 'worldbook_add.content')
  if (payload.enabled !== undefined) normalized.enabled = normalizeBoolean(payload.enabled, 'worldbook_add.enabled')
  if (payload.constant !== undefined) normalized.constant = normalizeBoolean(payload.constant, 'worldbook_add.constant')
  if (payload.selective !== undefined) normalized.selective = normalizeBoolean(payload.selective, 'worldbook_add.selective')
  if (payload.priority !== undefined) normalized.priority = normalizeInteger(payload.priority, 'worldbook_add.priority')
  if (payload.position !== undefined) normalized.position = normalizeLorebookPosition(payload.position, 'worldbook_add.position')
  if (payload.depth !== undefined) normalized.depth = normalizeInteger(payload.depth, 'worldbook_add.depth', { minimum: 0 })
  if (payload.probability !== undefined) normalized.probability = normalizeInteger(payload.probability, 'worldbook_add.probability', { minimum: 0, maximum: 100 })
  if (payload.prevent_recursion !== undefined) normalized.prevent_recursion = normalizeBoolean(payload.prevent_recursion, 'worldbook_add.prevent_recursion')
  if (payload.use_regex !== undefined) normalized.use_regex = normalizeBoolean(payload.use_regex, 'worldbook_add.use_regex')
  if (payload.match_persona_description !== undefined) normalized.match_persona_description = normalizeBoolean(payload.match_persona_description, 'worldbook_add.match_persona_description')
  if (payload.match_character_description !== undefined) normalized.match_character_description = normalizeBoolean(payload.match_character_description, 'worldbook_add.match_character_description')
  if (payload.match_character_personality !== undefined) normalized.match_character_personality = normalizeBoolean(payload.match_character_personality, 'worldbook_add.match_character_personality')
  if (payload.match_character_depth_prompt !== undefined) normalized.match_character_depth_prompt = normalizeBoolean(payload.match_character_depth_prompt, 'worldbook_add.match_character_depth_prompt')
  if (payload.match_scenario !== undefined) normalized.match_scenario = normalizeBoolean(payload.match_scenario, 'worldbook_add.match_scenario')
  if (payload.children !== undefined) {
    if (!Array.isArray(payload.children)) {
      throw new Error('worldbook_add.children 必须为数组')
    }
    normalized.children = payload.children.map((child, index) =>
      normalizeLorebookNode(child, `worldbook_add.children[${index}]`)
    )
  }
  return normalized
}

function normalizeWorldbookChanges(value: unknown): CharacterStudioWorldbookChanges {
  const changes = ensureRecord(value, 'worldbook_update.changes')
  const normalized: CharacterStudioWorldbookChanges = {}
  const target = normalized as unknown as Record<string, unknown>
  for (const [key, raw] of Object.entries(changes)) {
    if (!WORLD_BOOK_CHANGE_KEYS.has(key as keyof CharacterStudioWorldbookChanges)) {
      throw new Error(`worldbook_update 不支持字段: ${key}`)
    }
    if (key === 'comment' || key === 'content') {
      target[key] = stringValue(raw, `worldbook_update.${key}`)
    } else if (key === 'position') {
      normalized.position = normalizeLorebookPosition(raw, 'worldbook_update.position')
    } else if (key === 'keys' || key === 'secondary_keys') {
      target[key] = normalizeStringList(raw, `worldbook_update.${key}`)
    } else if (
      key === 'enabled' ||
      key === 'constant' ||
      key === 'selective' ||
      key === 'prevent_recursion' ||
      key === 'use_regex' ||
      key === 'match_persona_description' ||
      key === 'match_character_description' ||
      key === 'match_character_personality' ||
      key === 'match_character_depth_prompt' ||
      key === 'match_scenario'
    ) {
      target[key] = normalizeBoolean(raw, `worldbook_update.${key}`)
    } else {
      target[key] = normalizeInteger(
        raw,
        `worldbook_update.${key}`,
        key === 'depth'
          ? { minimum: 0 }
          : key === 'probability'
            ? { minimum: 0, maximum: 100 }
            : {},
      )
    }
  }
  return normalized
}

function normalizeRegexAddPayload(value: unknown): CharacterStudioRegexAddPayload {
  const payload = ensureRecord(value, 'regex_add')
  ensureAllowedKeys(payload, REGEX_CHANGE_KEYS, 'regex_add')
  const normalized: CharacterStudioRegexAddPayload = {}
  if (payload.scriptName !== undefined) normalized.scriptName = stringValue(payload.scriptName, 'regex_add.scriptName')
  if (payload.findRegex !== undefined) normalized.findRegex = stringValue(payload.findRegex, 'regex_add.findRegex')
  if (payload.replaceString !== undefined) normalized.replaceString = stringValue(payload.replaceString, 'regex_add.replaceString')
  if (payload.placement !== undefined) normalized.placement = normalizePlacement(payload.placement, 'regex_add.placement')
  if (payload.markdownOnly !== undefined) normalized.markdownOnly = normalizeBoolean(payload.markdownOnly, 'regex_add.markdownOnly')
  if (payload.promptOnly !== undefined) normalized.promptOnly = normalizeBoolean(payload.promptOnly, 'regex_add.promptOnly')
  if (payload.runOnEdit !== undefined) normalized.runOnEdit = normalizeBoolean(payload.runOnEdit, 'regex_add.runOnEdit')
  if (payload.disabled !== undefined) normalized.disabled = normalizeBoolean(payload.disabled, 'regex_add.disabled')
  return normalized
}

function normalizeRegexChanges(value: unknown): CharacterStudioRegexChanges {
  const changes = ensureRecord(value, 'regex_update.changes')
  const normalized: CharacterStudioRegexChanges = {}
  const target = normalized as unknown as Record<string, unknown>
  for (const [key, raw] of Object.entries(changes)) {
    if (!REGEX_CHANGE_KEYS.has(key as keyof CharacterStudioRegexChanges)) {
      throw new Error(`regex_update 不支持字段: ${key}`)
    }
    if (key === 'scriptName' || key === 'findRegex' || key === 'replaceString') {
      target[key] = stringValue(raw, `regex_update.${key}`)
    } else if (key === 'placement') {
      normalized.placement = normalizePlacement(raw, 'regex_update.placement')
    } else {
      target[key] = normalizeBoolean(raw, `regex_update.${key}`)
    }
  }
  return normalized
}

function normalizeTaskAddPayload(value: unknown): CharacterStudioTaskAddPayload {
  const payload = ensureRecord(value, 'task_add')
  ensureAllowedKeys(payload, TASK_CHANGE_KEYS, 'task_add')
  const normalized: CharacterStudioTaskAddPayload = {}
  if (payload.name !== undefined) normalized.name = stringValue(payload.name, 'task_add.name')
  if (payload.triggerTiming !== undefined) {
    const triggerTiming = stringValue(payload.triggerTiming, 'task_add.triggerTiming')
    if (!VALID_TRIGGER_TIMINGS.has(triggerTiming)) {
      throw new Error(`task_add.triggerTiming 不支持值: ${triggerTiming}`)
    }
    normalized.triggerTiming = triggerTiming
  }
  if (payload.interval !== undefined) normalized.interval = normalizeInteger(payload.interval, 'task_add.interval', { minimum: 0 })
  if (payload.commands !== undefined) normalized.commands = stringValue(payload.commands, 'task_add.commands')
  if (payload.disabled !== undefined) normalized.disabled = normalizeBoolean(payload.disabled, 'task_add.disabled')
  return normalized
}

function normalizeTaskChanges(value: unknown): CharacterStudioTaskChanges {
  const changes = ensureRecord(value, 'task_update.changes')
  const normalized: CharacterStudioTaskChanges = {}
  const target = normalized as unknown as Record<string, unknown>
  for (const [key, raw] of Object.entries(changes)) {
    if (!TASK_CHANGE_KEYS.has(key as keyof CharacterStudioTaskChanges)) {
      throw new Error(`task_update 不支持字段: ${key}`)
    }
    if (key === 'name' || key === 'commands') {
      target[key] = stringValue(raw, `task_update.${key}`)
    } else if (key === 'triggerTiming') {
      const triggerTiming = stringValue(raw, 'task_update.triggerTiming')
      if (!VALID_TRIGGER_TIMINGS.has(triggerTiming)) {
        throw new Error(`task_update.triggerTiming 不支持值: ${triggerTiming}`)
      }
      normalized.triggerTiming = triggerTiming
    } else if (key === 'interval') {
      normalized.interval = normalizeInteger(raw, 'task_update.interval', { minimum: 0 })
    } else {
      target[key] = normalizeBoolean(raw, `task_update.${key}`)
    }
  }
  return normalized
}

function normalizeUpdateOps<TChanges extends Record<string, unknown>>(
  value: unknown,
  label: string,
  normalizeChanges: (changes: unknown) => TChanges,
): Array<CharacterStudioPatchUpdateOp<TChanges>> {
  return toArray(value).map((item, index) => {
    const record = ensureRecord(item, `${label}[${index}]`)
    ensureAllowedKeys(record, new Set(['id', 'changes']), `${label}[${index}]`)
    return {
      id: ensureString(record.id, `${label}[${index}].id`),
      changes: normalizeChanges(record.changes),
    }
  })
}

function normalizeDeleteOps(value: unknown, label: string): CharacterStudioPatchDeleteOp[] {
  return toArray(value).map((item, index) => {
    const record = ensureRecord(item, `${label}[${index}]`)
    ensureAllowedKeys(record, new Set(['id']), `${label}[${index}]`)
    return {
      id: ensureString(record.id, `${label}[${index}].id`),
    }
  })
}

function normalizeGreetingAdds(value: unknown): string[] {
  return toArray(value).map((item, index) => {
    if (typeof item !== 'string') {
      throw new Error(`greeting_add[${index}] 必须为字符串`)
    }
    return item
  })
}

function normalizePatch(value: unknown): NormalizedPatch {
  const patch = ensureRecord(value, 'patch')
  for (const key of Object.keys(patch)) {
    if (!ALLOWED_PATCH_KEYS.has(key)) {
      throw new Error(`不支持的 patch 顶层字段: ${key}`)
    }
  }
  const set = patch.set === undefined ? {} : ensureRecord(patch.set, 'set')
  return {
    set,
    greetingAdds: patch.greeting_add === undefined ? [] : normalizeGreetingAdds(patch.greeting_add),
    worldbookAdds: toArray(patch.worldbook_add).map(normalizeWorldbookAddPayload),
    worldbookUpdates: normalizeUpdateOps(patch.worldbook_update, 'worldbook_update', normalizeWorldbookChanges),
    worldbookDeletes: normalizeDeleteOps(patch.worldbook_delete, 'worldbook_delete'),
    regexAdds: toArray(patch.regex_add).map(normalizeRegexAddPayload),
    regexUpdates: normalizeUpdateOps(patch.regex_update, 'regex_update', normalizeRegexChanges),
    regexDeletes: normalizeDeleteOps(patch.regex_delete, 'regex_delete'),
    taskAdds: toArray(patch.task_add).map(normalizeTaskAddPayload),
    taskUpdates: normalizeUpdateOps(patch.task_update, 'task_update', normalizeTaskChanges),
    taskDeletes: normalizeDeleteOps(patch.task_delete, 'task_delete'),
  }
}

function buildLorebookEntry(payload: CharacterStudioWorldbookAddPayload): LorebookEntryNode {
  return {
    id: createGeneratedId('entry'),
    comment: payload.comment ?? '新条目',
    keys: payload.keys ?? [],
    secondary_keys: payload.secondary_keys ?? [],
    content: payload.content ?? '',
    enabled: payload.enabled ?? true,
    constant: payload.constant ?? false,
    selective: payload.selective ?? true,
    priority: payload.priority ?? 100,
    position: payload.position ?? 'before_char',
    depth: payload.depth ?? 4,
    probability: payload.probability ?? 100,
    prevent_recursion: payload.prevent_recursion ?? true,
    use_regex: payload.use_regex ?? false,
    match_persona_description: payload.match_persona_description ?? true,
    match_character_description: payload.match_character_description ?? true,
    match_character_personality: payload.match_character_personality ?? true,
    match_character_depth_prompt: payload.match_character_depth_prompt ?? true,
    match_scenario: payload.match_scenario ?? true,
    children: deepClone(payload.children ?? []),
  }
}

function buildRegexScript(payload: CharacterStudioRegexAddPayload): RegexScript {
  return {
    id: createGeneratedId('regex'),
    scriptName: payload.scriptName ?? '新脚本',
    findRegex: payload.findRegex ?? '',
    replaceString: payload.replaceString ?? '',
    placement: payload.placement ?? [2],
    markdownOnly: payload.markdownOnly ?? false,
    promptOnly: payload.promptOnly ?? false,
    runOnEdit: payload.runOnEdit ?? true,
    disabled: payload.disabled ?? false,
  }
}

function buildStateTask(payload: CharacterStudioTaskAddPayload): StateTask {
  return {
    id: createGeneratedId('task'),
    name: payload.name ?? '新任务',
    triggerTiming: payload.triggerTiming ?? 'initialization',
    interval: payload.interval ?? 0,
    commands: payload.commands ?? '',
    disabled: payload.disabled ?? false,
  }
}

function updateLorebookEntryById(
  entries: LorebookEntryNode[],
  id: string,
  changes: CharacterStudioWorldbookChanges,
): { entries: LorebookEntryNode[]; found: boolean } {
  let found = false
  const nextEntries = entries.map(entry => {
    if (entry.id === id) {
      found = true
      return {
        ...entry,
        ...deepClone(changes),
      }
    }
    const nested = updateLorebookEntryById(entry.children, id, changes)
    if (nested.found) {
      found = true
      return {
        ...entry,
        children: nested.entries,
      }
    }
    return entry
  })
  return { entries: nextEntries, found }
}

function deleteLorebookEntryById(
  entries: LorebookEntryNode[],
  id: string,
): { entries: LorebookEntryNode[]; found: boolean } {
  let found = false
  const nextEntries: LorebookEntryNode[] = []
  for (const entry of entries) {
    if (entry.id === id) {
      found = true
      continue
    }
    const nested = deleteLorebookEntryById(entry.children, id)
    if (nested.found) {
      found = true
      nextEntries.push({
        ...entry,
        children: nested.entries,
      })
      continue
    }
    nextEntries.push(entry)
  }
  return { entries: nextEntries, found }
}

function updateArrayItemById<T extends { id: string }>(
  items: T[],
  id: string,
  changes: Record<string, unknown>,
  label: string,
): T[] {
  const index = items.findIndex(item => item.id === id)
  if (index < 0) {
    throw new Error(`未找到可更新的 ${label} 条目: ${id}`)
  }
  const nextItems = [...items]
  nextItems[index] = {
    ...nextItems[index]!,
    ...deepClone(changes),
  }
  return nextItems
}

function deleteArrayItemById<T extends { id: string }>(
  items: T[],
  id: string,
  label: string,
): T[] {
  const index = items.findIndex(item => item.id === id)
  if (index < 0) {
    throw new Error(`未找到可删除的 ${label} 条目: ${id}`)
  }
  return items.filter(item => item.id !== id)
}

function applySetField(
  document: CharacterStudioDocument,
  path: string,
  value: unknown,
) {
  switch (path) {
    case 'identity.name': {
      const name = ensureString(value, path)
      document.identity.name = name
      document.meta.title = name
      return
    }
    case 'identity.aliases':
      document.identity.aliases = normalizeStringList(value, path)
      return
    case 'identity.description':
      document.identity.description = stringValue(value, path)
      return
    case 'identity.personality':
      document.identity.personality = stringValue(value, path)
      return
    case 'identity.scenario':
      document.identity.scenario = stringValue(value, path)
      return
    case 'meta.tags':
      document.meta.tags = normalizeStringList(value, path)
      return
    case 'coreMessages.first_message':
      document.coreMessages.first_message = stringValue(value, path)
      return
    case 'coreMessages.message_example':
      document.coreMessages.message_example = stringValue(value, path)
      return
    case 'coreMessages.alternate_greetings':
      document.coreMessages.alternate_greetings = normalizeStringList(value, path)
      return
    case 'coreMessages.system_prompt':
      document.coreMessages.system_prompt = stringValue(value, path)
      return
    case 'coreMessages.post_history_instructions':
      document.coreMessages.post_history_instructions = stringValue(value, path)
      return
    case 'coreMessages.creator_notes':
      document.coreMessages.creator_notes = stringValue(value, path)
      return
    case 'coreMessages.character_version':
      document.coreMessages.character_version = stringValue(value, path)
      return
    case 'lorebook.name':
      document.lorebook.name = stringValue(value, path)
      return
    default:
      if (
        path === 'regexScripts'
        || path.startsWith('regexScripts.')
        || path === 'stateTasks'
        || path.startsWith('stateTasks.')
        || path === 'lorebook.entries'
        || path.startsWith('lorebook.entries.')
      ) {
        throw new Error(`set 不允许直接修改集合字段，请改用专用 patch 操作: ${path}`)
      }
      throw new Error(`set 不支持字段路径: ${path}`)
  }
}

export function applyCharacterStudioAgentPatch(
  document: CharacterStudioDocument,
  patch: CharacterStudioAgentPatchV2,
): CharacterStudioDocument {
  const nextDocument = deepClone(document)
  const frozenSections = new Set(nextDocument.status.frozen_sections)
  const normalizedPatch = normalizePatch(patch)

  for (const [path, value] of Object.entries(normalizedPatch.set)) {
    const rootSection = path.split('.')[0] || ''
    const frozenKey = rootSectionToFrozenKey(rootSection)
    if (frozenKey && frozenSections.has(frozenKey)) {
      continue
    }
    applySetField(nextDocument, path, value)
  }

  if (!frozenSections.has('greetings')) {
    nextDocument.coreMessages.alternate_greetings.push(...normalizedPatch.greetingAdds)
  }

  if (!frozenSections.has('lorebook')) {
    for (const item of normalizedPatch.worldbookAdds) {
      nextDocument.lorebook.entries.push(buildLorebookEntry(item))
    }
    for (const item of normalizedPatch.worldbookUpdates) {
      const result = updateLorebookEntryById(nextDocument.lorebook.entries, item.id, item.changes)
      if (!result.found) {
        throw new Error(`未找到可更新的 worldbook 条目: ${item.id}`)
      }
      nextDocument.lorebook.entries = result.entries
    }
    for (const item of normalizedPatch.worldbookDeletes) {
      const result = deleteLorebookEntryById(nextDocument.lorebook.entries, item.id)
      if (!result.found) {
        throw new Error(`未找到可删除的 worldbook 条目: ${item.id}`)
      }
      nextDocument.lorebook.entries = result.entries
    }
  }

  if (!frozenSections.has('regex')) {
    for (const item of normalizedPatch.regexAdds) {
      nextDocument.regexScripts.push(buildRegexScript(item))
    }
    for (const item of normalizedPatch.regexUpdates) {
      nextDocument.regexScripts = updateArrayItemById(nextDocument.regexScripts, item.id, item.changes, 'regex')
    }
    for (const item of normalizedPatch.regexDeletes) {
      nextDocument.regexScripts = deleteArrayItemById(nextDocument.regexScripts, item.id, 'regex')
    }
  }

  if (!frozenSections.has('state-tasks')) {
    for (const item of normalizedPatch.taskAdds) {
      nextDocument.stateTasks.push(buildStateTask(item))
    }
    for (const item of normalizedPatch.taskUpdates) {
      nextDocument.stateTasks = updateArrayItemById(nextDocument.stateTasks, item.id, item.changes, 'task')
    }
    for (const item of normalizedPatch.taskDeletes) {
      nextDocument.stateTasks = deleteArrayItemById(nextDocument.stateTasks, item.id, 'task')
    }
  }

  return nextDocument
}
