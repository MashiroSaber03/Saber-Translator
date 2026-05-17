import type {
  CharacterStudioAgentPatchV2,
  CharacterStudioDocument,
  CharacterStudioPatchDeleteOp,
  CharacterStudioPatchUpdateOp,
  CharacterStudioRegexChanges,
  CharacterStudioTaskChanges,
  CharacterStudioWorldbookChanges,
  LorebookEntryNode,
  RegexScript,
  StateTask,
} from '@/types/characterStudio'

export interface CharacterStudioPatchSummarySection {
  key: string
  title: string
  items: string[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function toArray<T>(value: T | T[] | undefined | null): T[] {
  if (value === undefined || value === null) return []
  return Array.isArray(value) ? value : [value]
}

function findLorebookEntry(entries: LorebookEntryNode[], id: string): LorebookEntryNode | null {
  for (const entry of entries) {
    if (entry.id === id) return entry
    const nested = findLorebookEntry(entry.children || [], id)
    if (nested) return nested
  }
  return null
}

function findRegexScript(scripts: RegexScript[], id: string): RegexScript | null {
  return scripts.find(item => item.id === id) || null
}

function findStateTask(tasks: StateTask[], id: string): StateTask | null {
  return tasks.find(item => item.id === id) || null
}

function summarizeSet(patch: CharacterStudioAgentPatchV2): string[] {
  if (!isRecord(patch.set)) return []
  return Object.entries(patch.set).map(([path, value]) => {
    const rendered = typeof value === 'string' ? value : JSON.stringify(value)
    return `${path} → ${rendered}`
  })
}

function summarizeGreetings(patch: CharacterStudioAgentPatchV2): string[] {
  return toArray(patch.greeting_add)
    .filter((item): item is string => typeof item === 'string')
    .map(item => `追加备用问候语：${item}`)
}

function summarizeWorldbookUpdate(
  op: CharacterStudioPatchUpdateOp<CharacterStudioWorldbookChanges>,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findLorebookEntry(document.lorebook.entries, op.id) : null
  const label = target?.comment || op.id
  const changedKeys = Object.keys(op.changes)
  return `更新「${label}」：${changedKeys.join('、')}`
}

function summarizeWorldbookDelete(
  op: CharacterStudioPatchDeleteOp,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findLorebookEntry(document.lorebook.entries, op.id) : null
  return `删除「${target?.comment || op.id}」`
}

function summarizeRegexUpdate(
  op: CharacterStudioPatchUpdateOp<CharacterStudioRegexChanges>,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findRegexScript(document.regexScripts, op.id) : null
  const label = target?.scriptName || op.id
  const changedKeys = Object.keys(op.changes)
  return `更新「${label}」：${changedKeys.join('、')}`
}

function summarizeRegexDelete(
  op: CharacterStudioPatchDeleteOp,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findRegexScript(document.regexScripts, op.id) : null
  return `删除「${target?.scriptName || op.id}」`
}

function summarizeTaskUpdate(
  op: CharacterStudioPatchUpdateOp<CharacterStudioTaskChanges>,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findStateTask(document.stateTasks, op.id) : null
  const label = target?.name || op.id
  const changedKeys = Object.keys(op.changes)
  return `更新「${label}」：${changedKeys.join('、')}`
}

function summarizeTaskDelete(
  op: CharacterStudioPatchDeleteOp,
  document: CharacterStudioDocument | null,
): string {
  const target = document ? findStateTask(document.stateTasks, op.id) : null
  return `删除「${target?.name || op.id}」`
}

export function buildCharacterStudioPatchSummary(
  patch: CharacterStudioAgentPatchV2 | null,
  document: CharacterStudioDocument | null,
): CharacterStudioPatchSummarySection[] {
  if (!patch) return []

  const sections: CharacterStudioPatchSummarySection[] = []

  const fieldItems = summarizeSet(patch)
  if (fieldItems.length > 0) {
    sections.push({
      key: 'set',
      title: '字段更新',
      items: fieldItems,
    })
  }

  const greetingItems = summarizeGreetings(patch)
  if (greetingItems.length > 0) {
    sections.push({
      key: 'greetings',
      title: '问候语',
      items: greetingItems,
    })
  }

  const worldbookItems = [
    ...toArray(patch.worldbook_add).map(item => {
      const payload = isRecord(item) ? item : {}
      const label = String(payload.comment || '新世界书条目')
      return `新增「${label}」`
    }),
    ...toArray(patch.worldbook_update).filter((item): item is CharacterStudioPatchUpdateOp<CharacterStudioWorldbookChanges> => isRecord(item) && typeof item.id === 'string' && isRecord(item.changes)).map(item => summarizeWorldbookUpdate(item, document)),
    ...toArray(patch.worldbook_delete).filter((item): item is CharacterStudioPatchDeleteOp => isRecord(item) && typeof item.id === 'string').map(item => summarizeWorldbookDelete(item, document)),
  ]
  if (worldbookItems.length > 0) {
    sections.push({
      key: 'worldbook',
      title: '世界书',
      items: worldbookItems,
    })
  }

  const regexItems = [
    ...toArray(patch.regex_add).map(item => {
      const payload = isRecord(item) ? item : {}
      return `新增「${String(payload.scriptName || '新正则脚本')}」`
    }),
    ...toArray(patch.regex_update).filter((item): item is CharacterStudioPatchUpdateOp<CharacterStudioRegexChanges> => isRecord(item) && typeof item.id === 'string' && isRecord(item.changes)).map(item => summarizeRegexUpdate(item, document)),
    ...toArray(patch.regex_delete).filter((item): item is CharacterStudioPatchDeleteOp => isRecord(item) && typeof item.id === 'string').map(item => summarizeRegexDelete(item, document)),
  ]
  if (regexItems.length > 0) {
    sections.push({
      key: 'regex',
      title: '正则',
      items: regexItems,
    })
  }

  const taskItems = [
    ...toArray(patch.task_add).map(item => {
      const payload = isRecord(item) ? item : {}
      return `新增「${String(payload.name || '新状态任务')}」`
    }),
    ...toArray(patch.task_update).filter((item): item is CharacterStudioPatchUpdateOp<CharacterStudioTaskChanges> => isRecord(item) && typeof item.id === 'string' && isRecord(item.changes)).map(item => summarizeTaskUpdate(item, document)),
    ...toArray(patch.task_delete).filter((item): item is CharacterStudioPatchDeleteOp => isRecord(item) && typeof item.id === 'string').map(item => summarizeTaskDelete(item, document)),
  ]
  if (taskItems.length > 0) {
    sections.push({
      key: 'tasks',
      title: '状态任务',
      items: taskItems,
    })
  }

  return sections
}
