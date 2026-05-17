import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioDocument,
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

function cloneDocument(document: CharacterStudioDocument): CharacterStudioDocument {
  return JSON.parse(JSON.stringify(document)) as CharacterStudioDocument
}

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

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

function ensureString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} 缺少有效字符串`)
  }
  return value.trim()
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
  const source = Array.isArray(value) ? value : [value]
  const normalized = source.map((item, index) => {
    const next = Number(item)
    if (!Number.isFinite(next)) {
      throw new Error(`${label}[${index}] 必须为数字`)
    }
    if (!VALID_REGEX_PLACEMENTS.has(next)) {
      throw new Error(`${label}[${index}] 只能使用 1 或 2`)
    }
    return next
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

function normalizeNumber(value: unknown, label: string): number {
  const next = Number(value)
  if (!Number.isFinite(next)) {
    throw new Error(`${label} 必须为数字`)
  }
  return next
}

function createGeneratedId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
}

function setByPath(target: Record<string, unknown>, path: string, value: unknown) {
  const keys = path.split('.')
  let current: Record<string, unknown> = target
  for (let index = 0; index < keys.length - 1; index += 1) {
    const key = keys[index]!
    const next = current[key]
    if (!isRecord(next)) {
      current[key] = {}
    }
    current = current[key] as Record<string, unknown>
  }
  current[keys[keys.length - 1]!] = value
}

function rootSectionToFrozenKey(section: string): string | null {
  if (section === 'identity') return 'identity'
  if (section === 'coreMessages') return 'greetings'
  if (section === 'lorebook') return 'lorebook'
  if (section === 'regexScripts') return 'regex'
  if (section === 'stateTasks') return 'state-tasks'
  return null
}

function normalizeLorebookPosition(value: unknown, label: string): string {
  const next = String(value || '').trim()
  if (!VALID_LOREBOOK_POSITIONS.has(next)) {
    throw new Error(`${label} 只能使用 before_char、at_depth、after_char`)
  }
  return next
}

function normalizeWorldbookAddPayload(value: unknown): CharacterStudioWorldbookAddPayload {
  const payload = ensureRecord(value, 'worldbook_add')
  const normalized: CharacterStudioWorldbookAddPayload = {}
  if (payload.comment !== undefined) normalized.comment = String(payload.comment)
  if (payload.keys !== undefined) normalized.keys = normalizeStringList(payload.keys, 'worldbook_add.keys')
  if (payload.secondary_keys !== undefined) normalized.secondary_keys = normalizeStringList(payload.secondary_keys, 'worldbook_add.secondary_keys')
  if (payload.content !== undefined) normalized.content = String(payload.content)
  if (payload.enabled !== undefined) normalized.enabled = normalizeBoolean(payload.enabled, 'worldbook_add.enabled')
  if (payload.constant !== undefined) normalized.constant = normalizeBoolean(payload.constant, 'worldbook_add.constant')
  if (payload.selective !== undefined) normalized.selective = normalizeBoolean(payload.selective, 'worldbook_add.selective')
  if (payload.priority !== undefined) normalized.priority = normalizeNumber(payload.priority, 'worldbook_add.priority')
  if (payload.position !== undefined) normalized.position = normalizeLorebookPosition(payload.position, 'worldbook_add.position')
  if (payload.depth !== undefined) normalized.depth = normalizeNumber(payload.depth, 'worldbook_add.depth')
  if (payload.probability !== undefined) normalized.probability = normalizeNumber(payload.probability, 'worldbook_add.probability')
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
    normalized.children = cloneValue(payload.children as LorebookEntryNode[])
  }
  return normalized
}

function normalizeWorldbookChanges(value: unknown): CharacterStudioWorldbookChanges {
  const changes = ensureRecord(value, 'worldbook_update.changes')
  const normalized: CharacterStudioWorldbookChanges = {}
  for (const [key, raw] of Object.entries(changes)) {
    if (!WORLD_BOOK_CHANGE_KEYS.has(key as keyof CharacterStudioWorldbookChanges)) {
      throw new Error(`worldbook_update 不支持字段: ${key}`)
    }
    if (key === 'comment' || key === 'content') {
      normalized[key] = String(raw)
    } else if (key === 'position') {
      normalized.position = normalizeLorebookPosition(raw, 'worldbook_update.position')
    } else if (key === 'keys' || key === 'secondary_keys') {
      normalized[key] = normalizeStringList(raw, `worldbook_update.${key}`)
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
      normalized[key] = normalizeBoolean(raw, `worldbook_update.${key}`)
    } else {
      normalized[key] = normalizeNumber(raw, `worldbook_update.${key}`)
    }
  }
  return normalized
}

function normalizeRegexAddPayload(value: unknown): CharacterStudioRegexAddPayload {
  const payload = ensureRecord(value, 'regex_add')
  const normalized: CharacterStudioRegexAddPayload = {}
  if (payload.scriptName !== undefined) normalized.scriptName = String(payload.scriptName)
  if (payload.findRegex !== undefined) normalized.findRegex = String(payload.findRegex)
  if (payload.replaceString !== undefined) normalized.replaceString = String(payload.replaceString)
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
  for (const [key, raw] of Object.entries(changes)) {
    if (!REGEX_CHANGE_KEYS.has(key as keyof CharacterStudioRegexChanges)) {
      throw new Error(`regex_update 不支持字段: ${key}`)
    }
    if (key === 'scriptName' || key === 'findRegex' || key === 'replaceString') {
      normalized[key] = String(raw)
    } else if (key === 'placement') {
      normalized.placement = normalizePlacement(raw, 'regex_update.placement')
    } else {
      normalized[key] = normalizeBoolean(raw, `regex_update.${key}`)
    }
  }
  return normalized
}

function normalizeTaskAddPayload(value: unknown): CharacterStudioTaskAddPayload {
  const payload = ensureRecord(value, 'task_add')
  const normalized: CharacterStudioTaskAddPayload = {}
  if (payload.name !== undefined) normalized.name = String(payload.name)
  if (payload.triggerTiming !== undefined) {
    const triggerTiming = String(payload.triggerTiming)
    if (!VALID_TRIGGER_TIMINGS.has(triggerTiming)) {
      throw new Error(`task_add.triggerTiming 不支持值: ${triggerTiming}`)
    }
    normalized.triggerTiming = triggerTiming
  }
  if (payload.interval !== undefined) normalized.interval = normalizeNumber(payload.interval, 'task_add.interval')
  if (payload.commands !== undefined) normalized.commands = String(payload.commands)
  if (payload.disabled !== undefined) normalized.disabled = normalizeBoolean(payload.disabled, 'task_add.disabled')
  return normalized
}

function normalizeTaskChanges(value: unknown): CharacterStudioTaskChanges {
  const changes = ensureRecord(value, 'task_update.changes')
  const normalized: CharacterStudioTaskChanges = {}
  for (const [key, raw] of Object.entries(changes)) {
    if (!TASK_CHANGE_KEYS.has(key as keyof CharacterStudioTaskChanges)) {
      throw new Error(`task_update 不支持字段: ${key}`)
    }
    if (key === 'name' || key === 'commands') {
      normalized[key] = String(raw)
    } else if (key === 'triggerTiming') {
      const triggerTiming = String(raw)
      if (!VALID_TRIGGER_TIMINGS.has(triggerTiming)) {
        throw new Error(`task_update.triggerTiming 不支持值: ${triggerTiming}`)
      }
      normalized.triggerTiming = triggerTiming
    } else if (key === 'interval') {
      normalized.interval = normalizeNumber(raw, 'task_update.interval')
    } else {
      normalized[key] = normalizeBoolean(raw, `task_update.${key}`)
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
    return {
      id: ensureString(record.id, `${label}[${index}].id`),
      changes: normalizeChanges(record.changes),
    }
  })
}

function normalizeDeleteOps(value: unknown, label: string): CharacterStudioPatchDeleteOp[] {
  return toArray(value).map((item, index) => {
    const record = ensureRecord(item, `${label}[${index}]`)
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

function normalizePatch(patch: CharacterStudioAgentPatchV2): NormalizedPatch {
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
    comment: payload.comment || '新条目',
    keys: payload.keys || [],
    secondary_keys: payload.secondary_keys || [],
    content: payload.content || '',
    enabled: payload.enabled ?? true,
    constant: payload.constant ?? false,
    selective: payload.selective ?? true,
    priority: payload.priority ?? 100,
    position: payload.position || 'before_char',
    depth: payload.depth ?? 4,
    probability: payload.probability ?? 100,
    prevent_recursion: payload.prevent_recursion ?? true,
    use_regex: payload.use_regex ?? false,
    match_persona_description: payload.match_persona_description ?? true,
    match_character_description: payload.match_character_description ?? true,
    match_character_personality: payload.match_character_personality ?? true,
    match_character_depth_prompt: payload.match_character_depth_prompt ?? true,
    match_scenario: payload.match_scenario ?? true,
    children: cloneValue(payload.children || []),
  }
}

function buildRegexScript(payload: CharacterStudioRegexAddPayload): RegexScript {
  return {
    id: createGeneratedId('regex'),
    scriptName: payload.scriptName || '新脚本',
    findRegex: payload.findRegex || '',
    replaceString: payload.replaceString || '',
    placement: payload.placement || [2],
    markdownOnly: payload.markdownOnly ?? false,
    promptOnly: payload.promptOnly ?? false,
    runOnEdit: payload.runOnEdit ?? true,
    disabled: payload.disabled ?? false,
  }
}

function buildStateTask(payload: CharacterStudioTaskAddPayload): StateTask {
  return {
    id: createGeneratedId('task'),
    name: payload.name || '新任务',
    triggerTiming: payload.triggerTiming || 'initialization',
    interval: payload.interval ?? 0,
    commands: payload.commands || '',
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
        ...cloneValue(changes),
      }
    }
    const nested = updateLorebookEntryById(entry.children || [], id, changes)
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
    const nested = deleteLorebookEntryById(entry.children || [], id)
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
    ...cloneValue(changes),
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

export function applyCharacterStudioAgentPatch(
  document: CharacterStudioDocument,
  patch: CharacterStudioAgentPatchV2,
): CharacterStudioDocument {
  const nextDocument = cloneDocument(document)
  const frozenSections = new Set(nextDocument.status.frozen_sections || [])
  const normalizedPatch = normalizePatch(patch)

  for (const [path, value] of Object.entries(normalizedPatch.set)) {
    if (
      path === 'regexScripts' ||
      path.startsWith('regexScripts.') ||
      path === 'stateTasks' ||
      path.startsWith('stateTasks.') ||
      path === 'lorebook.entries' ||
      path.startsWith('lorebook.entries.')
    ) {
      throw new Error(`set 不允许直接修改集合字段，请改用专用 patch 操作: ${path}`)
    }
    const rootSection = path.split('.')[0] || ''
    const frozenKey = rootSectionToFrozenKey(rootSection)
    if (frozenKey && frozenSections.has(frozenKey)) {
      continue
    }
    setByPath(nextDocument as unknown as Record<string, unknown>, path, value)
  }

  const patchedName = String(nextDocument.identity.name || '').trim()
  if (patchedName) {
    nextDocument.meta.title = patchedName
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
