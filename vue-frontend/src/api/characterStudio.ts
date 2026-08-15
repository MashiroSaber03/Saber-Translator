import { downloadBlob, readApiErrorMessage } from './download'
import { readSseStream } from './sse'
import { waitForOperation } from '@/api/v2/operations'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import {
  activateV2StudioSession,
  abortV2StudioSession,
  createV2StudioDocument,
  createV2StudioSession,
  deleteV2StudioDocument,
  deleteV2StudioMessage,
  deleteV2StudioSession,
  editV2StudioMessage,
  generateV2StudioDocument,
  getV2StudioCandidates,
  getV2StudioChatState,
  getV2StudioDocument,
  getV2StudioIndex,
  getV2StudioPromptPreview,
  getV2StudioSession,
  importV2StudioDocument,
  importV2StudioSession,
  importV2StudioWorldbook,
  regenerateV2StudioMessage,
  sendV2StudioMessage,
  summarizeV2StudioSession,
  updateV2StudioDocument,
  uploadV2StudioAsset,
  validateV2StudioDocument,
  v2StudioAgentUrl,
  v2StudioDocumentExportUrl,
  v2StudioSessionExportUrl,
} from '@/api/v2/studio'
import type {
  CharacterStudioChatAttachment,
  CharacterStudioChatMessage,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
  CharacterStudioChatState,
  CharacterStudioGenerationSection,
  CharacterStudioIndex,
  CharacterStudioSection,
  ExportDiagnostic,
} from '@/types/characterStudio'
import {
  CHARACTER_STUDIO_LOREBOOK_REQUIRED_FIELDS,
} from '@/types/characterStudio'
import { characterStudioDocumentContent } from '@/utils/characterStudioDocumentContent'

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label}必须是对象`)
  }
  return value as Record<string, unknown>
}

function exactObject(
  value: unknown,
  allowedKeys: readonly string[],
  requiredKeys: readonly string[],
  label: string,
): Record<string, unknown> {
  const result = objectValue(value, label)
  const keys = Object.keys(result)
  if (
    keys.some(key => !allowedKeys.includes(key))
    || requiredKeys.some(key => !Object.prototype.hasOwnProperty.call(result, key))
  ) {
    throw new Error(`${label}字段无效`)
  }
  return result
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label}必须是字符串`)
  return value
}

function nonEmptyString(value: unknown, label: string): string {
  const result = stringValue(value, label)
  if (!result) throw new Error(`${label}不能为空`)
  return result
}

function nullableString(value: unknown, label: string): string | null {
  return value === null ? null : stringValue(value, label)
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label}必须是布尔值`)
  return value
}

function arrayValue(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label}必须是数组`)
  return value
}

function stringArray(value: unknown, label: string): string[] {
  return arrayValue(value, label).map((item, index) => stringValue(item, `${label}[${index}]`))
}

const STUDIO_SECTIONS = new Set<CharacterStudioSection>([
  'identity',
  'greetings',
  'lorebook',
  'regex',
  'state-tasks',
])

function studioSections(value: unknown, label: string): CharacterStudioSection[] {
  return stringArray(value, label).map((section) => {
    if (!STUDIO_SECTIONS.has(section as CharacterStudioSection)) {
      throw new Error(`${label}包含无效区段`)
    }
    return section as CharacterStudioSection
  })
}

function integerValue(value: unknown, label: string, minimum?: number): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${label}必须是整数`)
  }
  if (minimum !== undefined && (value as number) < minimum) {
    throw new Error(`${label}必须是大于等于 ${minimum} 的整数`)
  }
  return value as number
}

function dateValue(value: unknown, label: string): string {
  const result = nonEmptyString(value, label)
  if (Number.isNaN(Date.parse(result))) throw new Error(`${label}必须是有效时间`)
  return result
}

function nullableDate(value: unknown, label: string): string | null {
  return value === null ? null : dateValue(value, label)
}

function mapDiagnostics(value: unknown, label: string): ExportDiagnostic {
  const diagnostics = exactObject(
    value,
    ['valid', 'errors', 'warnings', 'checks'],
    ['valid', 'errors', 'warnings', 'checks'],
    label,
  )
  const checks = exactObject(
    diagnostics.checks,
    ['document', 'v3_export', 'v2_export'],
    ['document', 'v3_export', 'v2_export'],
    `${label}.checks`,
  )
  return {
    valid: booleanValue(diagnostics.valid, `${label}.valid`),
    errors: stringArray(diagnostics.errors, `${label}.errors`),
    warnings: stringArray(diagnostics.warnings, `${label}.warnings`),
    checks: {
      document: booleanValue(checks.document, `${label}.checks.document`),
      v3_export: booleanValue(checks.v3_export, `${label}.checks.v3_export`),
      v2_export: booleanValue(checks.v2_export, `${label}.checks.v2_export`),
    },
  }
}

function optionalBoolean(
  value: unknown,
  label: string,
): boolean | undefined {
  return value === undefined ? undefined : booleanValue(value, label)
}

function optionalInteger(
  value: unknown,
  label: string,
  minimum = 0,
): number | undefined {
  return value === undefined ? undefined : integerValue(value, label, minimum)
}

function optionalStringArray(value: unknown, label: string): string[] | undefined {
  return value === undefined ? undefined : stringArray(value, label)
}

function mapLorebookEntry(
  value: unknown,
  label: string,
): CharacterStudioDocument['lorebook']['entries'][number] {
  const allowed = [
    'id', 'comment', 'keys', 'secondary_keys', 'content', 'enabled', 'constant',
    'selective', 'priority', 'position', 'depth', 'probability', 'prevent_recursion',
    'use_regex', 'match_persona_description', 'match_character_description',
    'match_character_personality', 'match_character_depth_prompt', 'match_scenario', 'children',
  ] as const
  const entry = exactObject(
    value,
    allowed,
    CHARACTER_STUDIO_LOREBOOK_REQUIRED_FIELDS,
    label,
  )
  return {
    id: nonEmptyString(entry.id, `${label}.id`),
    comment: stringValue(entry.comment, `${label}.comment`),
    keys: stringArray(entry.keys, `${label}.keys`),
    ...(entry.secondary_keys === undefined
      ? {}
      : { secondary_keys: optionalStringArray(entry.secondary_keys, `${label}.secondary_keys`) }),
    content: stringValue(entry.content, `${label}.content`),
    enabled: booleanValue(entry.enabled, `${label}.enabled`),
    constant: booleanValue(entry.constant, `${label}.constant`),
    selective: booleanValue(entry.selective, `${label}.selective`),
    priority: integerValue(entry.priority, `${label}.priority`),
    position: nonEmptyString(entry.position, `${label}.position`),
    depth: integerValue(entry.depth, `${label}.depth`, 0),
    ...(entry.probability === undefined
      ? {}
      : { probability: optionalInteger(entry.probability, `${label}.probability`) }),
    ...(entry.prevent_recursion === undefined
      ? {}
      : { prevent_recursion: optionalBoolean(entry.prevent_recursion, `${label}.prevent_recursion`) }),
    ...(entry.use_regex === undefined
      ? {}
      : { use_regex: optionalBoolean(entry.use_regex, `${label}.use_regex`) }),
    ...(entry.match_persona_description === undefined
      ? {}
      : { match_persona_description: optionalBoolean(entry.match_persona_description, `${label}.match_persona_description`) }),
    ...(entry.match_character_description === undefined
      ? {}
      : { match_character_description: optionalBoolean(entry.match_character_description, `${label}.match_character_description`) }),
    ...(entry.match_character_personality === undefined
      ? {}
      : { match_character_personality: optionalBoolean(entry.match_character_personality, `${label}.match_character_personality`) }),
    ...(entry.match_character_depth_prompt === undefined
      ? {}
      : { match_character_depth_prompt: optionalBoolean(entry.match_character_depth_prompt, `${label}.match_character_depth_prompt`) }),
    ...(entry.match_scenario === undefined
      ? {}
      : { match_scenario: optionalBoolean(entry.match_scenario, `${label}.match_scenario`) }),
    children: arrayValue(entry.children, `${label}.children`).map((child, index) =>
      mapLorebookEntry(child, `${label}.children[${index}]`)
    ),
  }
}

function mapRegexScript(
  value: unknown,
  label: string,
): CharacterStudioDocument['regexScripts'][number] {
  const script = exactObject(
    value,
    ['id', 'scriptName', 'findRegex', 'replaceString', 'placement', 'markdownOnly', 'promptOnly', 'runOnEdit', 'disabled'],
    ['id', 'scriptName', 'findRegex', 'replaceString', 'placement', 'markdownOnly', 'promptOnly', 'runOnEdit', 'disabled'],
    label,
  )
  return {
    id: nonEmptyString(script.id, `${label}.id`),
    scriptName: stringValue(script.scriptName, `${label}.scriptName`),
    findRegex: stringValue(script.findRegex, `${label}.findRegex`),
    replaceString: stringValue(script.replaceString, `${label}.replaceString`),
    placement: arrayValue(script.placement, `${label}.placement`).map((item, index) =>
      integerValue(item, `${label}.placement[${index}]`)
    ),
    markdownOnly: booleanValue(script.markdownOnly, `${label}.markdownOnly`),
    promptOnly: booleanValue(script.promptOnly, `${label}.promptOnly`),
    runOnEdit: booleanValue(script.runOnEdit, `${label}.runOnEdit`),
    disabled: booleanValue(script.disabled, `${label}.disabled`),
  }
}

function mapStateTask(
  value: unknown,
  label: string,
): CharacterStudioDocument['stateTasks'][number] {
  const task = exactObject(
    value,
    ['id', 'name', 'triggerTiming', 'interval', 'commands', 'disabled'],
    ['id', 'name', 'triggerTiming', 'interval', 'commands', 'disabled'],
    label,
  )
  const triggerTiming = stringValue(task.triggerTiming, `${label}.triggerTiming`)
  return {
    id: nonEmptyString(task.id, `${label}.id`),
    name: stringValue(task.name, `${label}.name`),
    triggerTiming,
    interval: integerValue(task.interval, `${label}.interval`),
    commands: stringValue(task.commands, `${label}.commands`),
    disabled: booleanValue(task.disabled, `${label}.disabled`),
  }
}

function mapDocument(value: unknown): CharacterStudioDocument {
  const raw = exactObject(
    value,
    ['id', 'bookId', 'title', 'revision', 'avatarAssetId', 'avatarUrl', 'origin', 'status', 'meta', 'identity', 'coreMessages', 'lorebook', 'regexScripts', 'stateTasks', 'exportArtifacts', 'createdAt', 'updatedAt'],
    ['id', 'bookId', 'title', 'revision', 'avatarAssetId', 'avatarUrl', 'origin', 'status', 'meta', 'identity', 'coreMessages', 'lorebook', 'regexScripts', 'stateTasks', 'exportArtifacts', 'createdAt', 'updatedAt'],
    '角色文档',
  )
  const title = nonEmptyString(raw.title, '角色文档.title')
  const origin = exactObject(raw.origin, ['type', 'source_character'], ['type', 'source_character'], '角色文档.origin')
  const originType = stringValue(origin.type, '角色文档.origin.type')
  if (!['analysis', 'manual', 'imported'].includes(originType)) {
    throw new Error('角色文档.origin.type无效')
  }
  const status = exactObject(
    raw.status,
    ['is_favorite', 'frozen_sections', 'last_diagnostics', 'last_validated_at'],
    ['is_favorite', 'frozen_sections', 'last_diagnostics', 'last_validated_at'],
    '角色文档.status',
  )
  const meta = exactObject(raw.meta, ['title', 'tags'], ['title', 'tags'], '角色文档.meta')
  const identity = exactObject(
    raw.identity,
    ['name', 'aliases', 'description', 'personality', 'scenario'],
    ['name', 'aliases', 'description', 'personality', 'scenario'],
    '角色文档.identity',
  )
  const metaTitle = nonEmptyString(meta.title, '角色文档.meta.title')
  const identityName = nonEmptyString(identity.name, '角色文档.identity.name')
  if (title !== metaTitle || title !== identityName) {
    throw new Error('角色文档标题字段不一致')
  }
  const core = exactObject(
    raw.coreMessages,
    ['first_message', 'message_example', 'alternate_greetings', 'system_prompt', 'post_history_instructions', 'creator_notes', 'character_version'],
    ['first_message', 'message_example', 'alternate_greetings', 'system_prompt', 'post_history_instructions', 'creator_notes', 'character_version'],
    '角色文档.coreMessages',
  )
  const lorebook = exactObject(raw.lorebook, ['name', 'entries'], ['name', 'entries'], '角色文档.lorebook')
  nullableString(raw.avatarAssetId, '角色文档.avatarAssetId')
  const createdAt = dateValue(raw.createdAt, '角色文档.createdAt')
  const updatedAt = dateValue(raw.updatedAt, '角色文档.updatedAt')
  const diagnostics = status.last_diagnostics === null
    ? null
    : mapDiagnostics(status.last_diagnostics, '角色文档.status.last_diagnostics')
  return {
    id: nonEmptyString(raw.id, '角色文档.id'),
    bookId: nonEmptyString(raw.bookId, '角色文档.bookId'),
    origin: {
      type: originType as CharacterStudioDocument['origin']['type'],
      source_character: nullableString(origin.source_character, '角色文档.origin.source_character'),
    },
    status: {
      is_favorite: booleanValue(status.is_favorite, '角色文档.status.is_favorite'),
      frozen_sections: studioSections(status.frozen_sections, '角色文档.status.frozen_sections'),
      last_diagnostics: diagnostics,
      last_validated_at: nullableDate(status.last_validated_at, '角色文档.status.last_validated_at'),
    },
    meta: {
      title,
      tags: stringArray(meta.tags, '角色文档.meta.tags'),
    },
    identity: {
      name: title,
      aliases: stringArray(identity.aliases, '角色文档.identity.aliases'),
      description: stringValue(identity.description, '角色文档.identity.description'),
      personality: stringValue(identity.personality, '角色文档.identity.personality'),
      scenario: stringValue(identity.scenario, '角色文档.identity.scenario'),
    },
    coreMessages: {
      first_message: stringValue(core.first_message, '角色文档.coreMessages.first_message'),
      message_example: stringValue(core.message_example, '角色文档.coreMessages.message_example'),
      alternate_greetings: stringArray(core.alternate_greetings, '角色文档.coreMessages.alternate_greetings'),
      system_prompt: stringValue(core.system_prompt, '角色文档.coreMessages.system_prompt'),
      post_history_instructions: stringValue(core.post_history_instructions, '角色文档.coreMessages.post_history_instructions'),
      creator_notes: stringValue(core.creator_notes, '角色文档.coreMessages.creator_notes'),
      character_version: stringValue(core.character_version, '角色文档.coreMessages.character_version'),
    },
    lorebook: {
      name: stringValue(lorebook.name, '角色文档.lorebook.name'),
      entries: arrayValue(lorebook.entries, '角色文档.lorebook.entries').map((entry, index) =>
        mapLorebookEntry(entry, `角色文档.lorebook.entries[${index}]`)
      ),
    },
    regexScripts: arrayValue(raw.regexScripts, '角色文档.regexScripts').map((script, index) =>
      mapRegexScript(script, `角色文档.regexScripts[${index}]`)
    ),
    stateTasks: arrayValue(raw.stateTasks, '角色文档.stateTasks').map((task, index) =>
      mapStateTask(task, `角色文档.stateTasks[${index}]`)
    ),
    exportArtifacts: objectValue(raw.exportArtifacts, '角色文档.exportArtifacts'),
    revision: integerValue(raw.revision, '角色文档.revision', 1),
    avatarUrl: nullableString(raw.avatarUrl, '角色文档.avatarUrl'),
    createdAt,
    updatedAt,
  }
}

function mapAttachment(value: unknown, label: string): CharacterStudioChatAttachment {
  const attachment = exactObject(
    value,
    ['assetId', 'assetUrl', 'mimeType', 'byteSize', 'width', 'height', 'available'],
    ['assetId', 'assetUrl', 'mimeType', 'byteSize', 'width', 'height', 'available'],
    label,
  )
  const assetId = nonEmptyString(attachment.assetId, `${label}.assetId`)
  integerValue(attachment.byteSize, `${label}.byteSize`, 1)
  if (attachment.width !== null) integerValue(attachment.width, `${label}.width`, 1)
  if (attachment.height !== null) integerValue(attachment.height, `${label}.height`, 1)
  booleanValue(attachment.available, `${label}.available`)
  return {
    attachment_id: assetId,
    filename: assetId,
    mime_type: nonEmptyString(attachment.mimeType, `${label}.mimeType`),
    asset_path: nonEmptyString(attachment.assetUrl, `${label}.assetUrl`),
  }
}

function mapMessage(value: unknown, label: string): CharacterStudioChatMessage {
  const message = exactObject(
    value,
    ['messageId', 'ordinal', 'role', 'content', 'attachments', 'runtimeLog', 'variablesSnapshot', 'generationMeta', 'createdAt', 'updatedAt'],
    ['messageId', 'ordinal', 'role', 'content', 'attachments', 'runtimeLog', 'variablesSnapshot', 'generationMeta', 'createdAt', 'updatedAt'],
    label,
  )
  integerValue(message.ordinal, `${label}.ordinal`, 1)
  const role = stringValue(message.role, `${label}.role`)
  if (!['system', 'user', 'assistant'].includes(role)) throw new Error(`${label}.role无效`)
  const runtimeLog = arrayValue(message.runtimeLog, `${label}.runtimeLog`).map((item, index) =>
    objectValue(item, `${label}.runtimeLog[${index}]`)
  )
  return {
    message_id: nonEmptyString(message.messageId, `${label}.messageId`),
    role: role as CharacterStudioChatMessage['role'],
    content: stringValue(message.content, `${label}.content`),
    attachments: arrayValue(message.attachments, `${label}.attachments`).map((item, index) =>
      mapAttachment(item, `${label}.attachments[${index}]`)
    ),
    runtime_log: runtimeLog,
    variables_snapshot: objectValue(message.variablesSnapshot, `${label}.variablesSnapshot`),
    generation_meta: objectValue(message.generationMeta, `${label}.generationMeta`),
    created_at: dateValue(message.createdAt, `${label}.createdAt`),
    updated_at: dateValue(message.updatedAt, `${label}.updatedAt`),
  }
}

function mapSummaryBlocks(value: unknown, label: string): Array<{ summary: string }> {
  return arrayValue(value, label).map((item, index) => {
    const block = exactObject(item, ['summary'], ['summary'], `${label}[${index}]`)
    return { summary: nonEmptyString(block.summary, `${label}[${index}].summary`) }
  })
}

function mapSession(value: unknown): CharacterStudioChatSession {
  const raw = exactObject(
    value,
    ['sessionId', 'documentId', 'indexRevision', 'title', 'revision', 'generation', 'greetingSource', 'variables', 'summaryBlocks', 'summaryThroughMessageId', 'summaryGeneration', 'runtimeState', 'archived', 'archivedAt', 'messages', 'createdAt', 'updatedAt'],
    ['sessionId', 'documentId', 'indexRevision', 'title', 'revision', 'generation', 'greetingSource', 'variables', 'summaryBlocks', 'summaryThroughMessageId', 'summaryGeneration', 'runtimeState', 'archived', 'archivedAt', 'messages', 'createdAt', 'updatedAt'],
    '角色聊天会话',
  )
  const archived = booleanValue(raw.archived, '角色聊天会话.archived')
  const archivedAt = nullableDate(raw.archivedAt, '角色聊天会话.archivedAt')
  if (archived !== (archivedAt !== null)) throw new Error('角色聊天会话归档状态不一致')
  nullableString(raw.summaryThroughMessageId, '角色聊天会话.summaryThroughMessageId')
  integerValue(raw.summaryGeneration, '角色聊天会话.summaryGeneration', 0)
  objectValue(raw.runtimeState, '角色聊天会话.runtimeState')
  return {
    session_id: nonEmptyString(raw.sessionId, '角色聊天会话.sessionId'),
    doc_id: nonEmptyString(raw.documentId, '角色聊天会话.documentId'),
    index_revision: integerValue(raw.indexRevision, '角色聊天会话.indexRevision', 1),
    title: stringValue(raw.title, '角色聊天会话.title'),
    created_at: dateValue(raw.createdAt, '角色聊天会话.createdAt'),
    updated_at: dateValue(raw.updatedAt, '角色聊天会话.updatedAt'),
    archived_at: archivedAt,
    greeting_source: objectValue(raw.greetingSource, '角色聊天会话.greetingSource'),
    summary_blocks: mapSummaryBlocks(raw.summaryBlocks, '角色聊天会话.summaryBlocks'),
    messages: arrayValue(raw.messages, '角色聊天会话.messages').map((message, index) =>
      mapMessage(message, `角色聊天会话.messages[${index}]`)
    ),
    variables: objectValue(raw.variables, '角色聊天会话.variables'),
    revision: integerValue(raw.revision, '角色聊天会话.revision', 1),
    generation: integerValue(raw.generation, '角色聊天会话.generation', 1),
  }
}

function mapSessionSummary(value: unknown, label: string): CharacterStudioChatSessionSummary {
  const raw = exactObject(
    value,
    ['sessionId', 'title', 'revision', 'generation', 'archived', 'archivedAt', 'messageCount', 'lastMessageExcerpt', 'updatedAt'],
    ['sessionId', 'title', 'revision', 'generation', 'archived', 'archivedAt', 'messageCount', 'lastMessageExcerpt', 'updatedAt'],
    label,
  )
  const archived = booleanValue(raw.archived, `${label}.archived`)
  const archivedAt = nullableDate(raw.archivedAt, `${label}.archivedAt`)
  if (archived !== (archivedAt !== null)) throw new Error(`${label}归档状态不一致`)
  return {
    session_id: nonEmptyString(raw.sessionId, `${label}.sessionId`),
    title: stringValue(raw.title, `${label}.title`),
    revision: integerValue(raw.revision, `${label}.revision`, 1),
    generation: integerValue(raw.generation, `${label}.generation`, 1),
    message_count: integerValue(raw.messageCount, `${label}.messageCount`, 0),
    updated_at: dateValue(raw.updatedAt, `${label}.updatedAt`),
    archived_at: archivedAt,
    last_message_excerpt: stringValue(raw.lastMessageExcerpt, `${label}.lastMessageExcerpt`),
  }
}

function mapChatState(value: unknown): CharacterStudioChatState {
  const raw = exactObject(
    value,
    ['documentId', 'indexRevision', 'sessions', 'activeSession', 'availableGreetings'],
    ['documentId', 'indexRevision', 'sessions', 'activeSession', 'availableGreetings'],
    '角色聊天状态',
  )
  const summaries = arrayValue(raw.sessions, '角色聊天状态.sessions').map((item, index) =>
    mapSessionSummary(item, `角色聊天状态.sessions[${index}]`)
  )
  return {
    doc_id: nonEmptyString(raw.documentId, '角色聊天状态.documentId'),
    index_revision: integerValue(raw.indexRevision, '角色聊天状态.indexRevision', 1),
    active_session: raw.activeSession === null ? null : mapSession(raw.activeSession),
    archived_sessions: summaries.filter(item => item.archived_at !== null),
    available_greetings: arrayValue(raw.availableGreetings, '角色聊天状态.availableGreetings').map((item, index) => {
      const label = `角色聊天状态.availableGreetings[${index}]`
      const greeting = exactObject(item, ['greetingId', 'label', 'content', 'source'], ['greetingId', 'label', 'content', 'source'], label)
      return {
        greeting_id: nonEmptyString(greeting.greetingId, `${label}.greetingId`),
        label: stringValue(greeting.label, `${label}.label`),
        content: stringValue(greeting.content, `${label}.content`),
        source: objectValue(greeting.source, `${label}.source`),
      }
    }),
  }
}

function formatPromptPreview(value: unknown): string {
  const preview = exactObject(
    value,
    ['system', 'messages', 'lorebookHits'],
    ['system', 'messages', 'lorebookHits'],
    '角色提示词预览',
  )
  const sections: string[] = []
  const system = stringValue(preview.system, '角色提示词预览.system')
  if (system.trim()) sections.push(`[system]\n${system}`)
  for (const [index, item] of arrayValue(preview.messages, '角色提示词预览.messages').entries()) {
    const label = `角色提示词预览.messages[${index}]`
    const message = exactObject(item, ['role', 'content', 'assetIds'], ['role', 'content', 'assetIds'], label)
    const role = stringValue(message.role, `${label}.role`)
    if (!['system', 'user', 'assistant'].includes(role)) throw new Error(`${label}.role无效`)
    const content = stringValue(message.content, `${label}.content`)
    const assetIds = stringArray(message.assetIds, `${label}.assetIds`)
    const assets = assetIds.length ? `\n[assets] ${assetIds.join(', ')}` : ''
    sections.push(`[${role}]\n${content}${assets}`)
  }
  const lorebookHits = arrayValue(preview.lorebookHits, '角色提示词预览.lorebookHits').map((item, index) => {
    const label = `角色提示词预览.lorebookHits[${index}]`
    const hit = exactObject(item, ['id', 'comment'], ['id', 'comment'], label)
    return {
      id: nonEmptyString(hit.id, `${label}.id`),
      comment: stringValue(hit.comment, `${label}.comment`),
    }
  })
  if (lorebookHits.length) sections.push(`[lorebook]\n${JSON.stringify(lorebookHits, null, 2)}`)
  return sections.join('\n\n')
}

async function refreshedChatState(documentId: string): Promise<CharacterStudioChatState> {
  const state = mapChatState(await getV2StudioChatState(documentId))
  if (state.doc_id !== documentId) throw new Error('角色聊天状态文档身份不匹配')
  return state
}

export async function getCharacterStudioIndex(bookId: string): Promise<CharacterStudioIndex> {
  const [index, candidates] = await Promise.all([
    getV2StudioIndex(bookId),
    getV2StudioCandidates(bookId),
  ])
  const indexPayload = exactObject(
    index,
    ['bookId', 'documents', 'candidateStatus'],
    ['bookId', 'documents', 'candidateStatus'],
    '角色工作室索引',
  )
  if (nonEmptyString(indexPayload.bookId, '角色工作室索引.bookId') !== bookId) {
    throw new Error('角色工作室索引书籍身份不匹配')
  }
  const candidateStatus = exactObject(
    indexPayload.candidateStatus,
    ['available', 'reason'],
    ['available', 'reason'],
    '角色工作室索引.candidateStatus',
  )
  booleanValue(candidateStatus.available, '角色工作室索引.candidateStatus.available')
  nullableString(candidateStatus.reason, '角色工作室索引.candidateStatus.reason')
  const documents = arrayValue(indexPayload.documents, '角色工作室索引.documents').map((item, itemIndex) => {
    const label = `角色工作室索引.documents[${itemIndex}]`
    const document = exactObject(
      item,
      ['documentId', 'title', 'kind', 'revision', 'avatarAssetId', 'hasAvatar', 'sourceCharacter', 'tags', 'isFavorite', 'updatedAt'],
      ['documentId', 'title', 'kind', 'revision', 'avatarAssetId', 'hasAvatar', 'sourceCharacter', 'tags', 'isFavorite', 'updatedAt'],
      label,
    )
    const kind = stringValue(document.kind, `${label}.kind`)
    if (!['analysis', 'imported', 'manual'].includes(kind)) throw new Error(`${label}.kind无效`)
    nullableString(document.avatarAssetId, `${label}.avatarAssetId`)
    integerValue(document.revision, `${label}.revision`, 1)
    return {
      id: nonEmptyString(document.documentId, `${label}.documentId`),
      title: nonEmptyString(document.title, `${label}.title`),
      origin: kind as CharacterStudioIndex['documents'][number]['origin'],
      source_character: nullableString(document.sourceCharacter, `${label}.sourceCharacter`),
      updated_at: dateValue(document.updatedAt, `${label}.updatedAt`),
      tags: stringArray(document.tags, `${label}.tags`),
      is_favorite: booleanValue(document.isFavorite, `${label}.isFavorite`),
      has_avatar: booleanValue(document.hasAvatar, `${label}.hasAvatar`),
    }
  })
  const candidatePayload = exactObject(
    candidates,
    ['available', 'reason', 'items'],
    ['available', 'reason', 'items'],
    '角色候选列表',
  )
  const hasTimeline = booleanValue(candidatePayload.available, '角色候选列表.available')
  nullableString(candidatePayload.reason, '角色候选列表.reason')
  const mappedCandidates = arrayValue(candidatePayload.items, '角色候选列表.items').map((item, itemIndex) => {
    const label = `角色候选列表.items[${itemIndex}]`
    const candidate = exactObject(
      item,
      ['characterId', 'name', 'aliases', 'description', 'personality', 'arc', 'firstAppearancePage', 'keyMomentCount', 'relatedPageCount', 'relatedPageNumbers'],
      ['characterId', 'name', 'aliases', 'description', 'personality', 'arc', 'firstAppearancePage', 'keyMomentCount', 'relatedPageCount', 'relatedPageNumbers'],
      label,
    )
    stringValue(candidate.description, `${label}.description`)
    stringValue(candidate.personality, `${label}.personality`)
    stringValue(candidate.arc, `${label}.arc`)
    const firstAppearancePage = candidate.firstAppearancePage === null
      ? null
      : integerValue(candidate.firstAppearancePage, `${label}.firstAppearancePage`, 1)
    return {
      id: nonEmptyString(candidate.characterId, `${label}.characterId`),
      name: nonEmptyString(candidate.name, `${label}.name`),
      aliases: stringArray(candidate.aliases, `${label}.aliases`),
      first_appearance_page: firstAppearancePage,
      key_moment_count: integerValue(candidate.keyMomentCount, `${label}.keyMomentCount`, 0),
      related_page_count: integerValue(candidate.relatedPageCount, `${label}.relatedPageCount`, 0),
      related_page_numbers: arrayValue(candidate.relatedPageNumbers, `${label}.relatedPageNumbers`).map((page, pageIndex) =>
        integerValue(page, `${label}.relatedPageNumbers[${pageIndex}]`, 1)
      ),
    }
  })
  return {
    book_id: bookId,
    documents,
    candidates: mappedCandidates,
    count: documents.length,
    has_timeline: hasTimeline,
  }
}

export async function createCharacterStudioDocument(
  bookId: string,
  payload: { candidate_id: string; title?: string } | { candidate_id?: never; title: string }
): Promise<CharacterStudioDocument> {
  const document = await createV2StudioDocument(bookId, {
    ...(payload.title ? { title: payload.title } : {}),
    ...(payload.candidate_id ? { candidateId: payload.candidate_id } : {}),
  })
  const mapped = mapDocument(document)
  if (mapped.bookId !== bookId) throw new Error('新建角色文档书籍身份不匹配')
  return mapped
}

export async function getCharacterStudioDocument(docId: string): Promise<CharacterStudioDocument> {
  const document = mapDocument(await getV2StudioDocument(docId))
  if (document.id !== docId) throw new Error('角色文档身份不匹配')
  return document
}

export async function saveCharacterStudioDocument(
  docId: string,
  payload: CharacterStudioDocument
): Promise<CharacterStudioDocument> {
  if (!Number.isInteger(payload.revision) || payload.revision < 1) {
    throw new Error('角色文档版本缺失，请重新加载')
  }
  const document = await updateV2StudioDocument(docId, {
    baseRevision: payload.revision,
    title: payload.meta.title,
    document: characterStudioDocumentContent(payload),
  })
  const mapped = mapDocument(document)
  if (mapped.id !== docId) throw new Error('保存后的角色文档身份不匹配')
  return mapped
}

export async function deleteCharacterStudioDocument(docId: string): Promise<void> {
  const response = exactObject(
    await deleteV2StudioDocument(docId),
    ['deleted', 'documentId'],
    ['deleted', 'documentId'],
    '角色文档删除结果',
  )
  if (response.deleted !== true || response.documentId !== docId) {
    throw new Error('角色文档删除结果无效')
  }
}

export async function generateCharacterStudioSection(
  docId: string,
  baseRevision: number,
  section: CharacterStudioGenerationSection
): Promise<CharacterStudioDocument> {
  const accepted = await generateV2StudioDocument(docId, baseRevision, section)
  const operation = await waitForOperation(accepted.operationId)
  if (operation.kind !== 'studio_generate' || operation.studioDocumentId !== docId) {
    throw new Error('角色文档生成操作身份不匹配')
  }
  return getCharacterStudioDocument(docId)
}

export async function validateCharacterStudioDocument(
  docId: string,
  baseRevision: number
): Promise<
  ExportDiagnostic & {
    document: CharacterStudioDocument
  }
> {
  const response = await validateV2StudioDocument(docId, baseRevision)
  const validation = exactObject(
    response,
    ['documentRevision', 'diagnostics'],
    ['documentRevision', 'diagnostics'],
    '角色文档校验结果',
  )
  integerValue(validation.documentRevision, '角色文档校验结果.documentRevision', 1)
  const diagnostics = mapDiagnostics(validation.diagnostics, '角色文档校验结果.diagnostics')
  const refreshed = await getV2StudioDocument(docId)
  const document = mapDocument(refreshed)
  if (document.id !== docId) throw new Error('校验后的角色文档身份不匹配')
  return {
    ...diagnostics,
    document,
  }
}

export async function runCharacterStudioAgent(docId: string, message: string): Promise<string> {
  assertBackendActionAllowed()
  const response = await fetch(v2StudioAgentUrl(docId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({ content: message }),
  })
  if (!response.ok) throw new Error(await readApiErrorMessage(response, 'Agent 调用失败'))
  if (!response.headers.get('content-type')?.toLowerCase().includes('text/event-stream')) {
    throw new Error('Agent 接口未返回事件流')
  }
  let content = ''
  let ready = false
  let done = false
  let streamError: string | null = null
  await readSseStream<unknown>(response, {
    missingBodyMessage: '无法读取 Agent 响应流',
    parseErrorMessage: 'Agent 响应格式无效',
    onMessage(event) {
      if (done || streamError !== null) throw new Error('Agent 终态后仍返回事件')
      if (event.event === 'ready') {
        exactObject(event.data, [], [], 'Agent ready 事件')
        if (ready) throw new Error('Agent 重复返回 ready 事件')
        ready = true
        return
      }
      if (!ready) throw new Error('Agent 未就绪即返回数据')
      if (event.event === 'chunk') {
        const chunk = exactObject(event.data, ['text'], ['text'], 'Agent chunk 事件')
        content += nonEmptyString(chunk.text, 'Agent chunk 事件.text')
        return
      }
      if (event.event === 'done') {
        exactObject(event.data, [], [], 'Agent done 事件')
        done = true
        return
      }
      if (event.event === 'error') {
        const error = exactObject(event.data, ['message'], ['message'], 'Agent error 事件')
        streamError = nonEmptyString(error.message, 'Agent error 事件.message')
        return
      }
      throw new Error(`Agent 返回未知事件：${event.event}`)
    },
  })
  if (streamError) throw new Error(streamError)
  if (!done) throw new Error('Agent 响应流提前结束')
  if (!content.trim()) throw new Error('Agent 未返回有效内容')
  return content
}

export async function getCharacterStudioChatState(
  docId: string
): Promise<CharacterStudioChatState> {
  return refreshedChatState(docId)
}

export async function createCharacterStudioChatSession(
  docId: string,
  baseIndexRevision: number,
  greetingId?: string
): Promise<CharacterStudioChatState> {
  const created = mapSession(await createV2StudioSession(docId, {
    baseIndexRevision,
    ...(greetingId ? { greetingId } : {}),
  }))
  if (created.doc_id !== docId) throw new Error('新建角色聊天会话文档身份不匹配')
  return refreshedChatState(docId)
}

export async function switchCharacterStudioChatSession(
  docId: string,
  sessionId: string,
  baseIndexRevision: number
): Promise<CharacterStudioChatState> {
  const activated = mapSession(await activateV2StudioSession(sessionId, baseIndexRevision))
  if (activated.session_id !== sessionId || activated.doc_id !== docId) {
    throw new Error('激活角色聊天会话身份不匹配')
  }
  return refreshedChatState(docId)
}

export async function deleteCharacterStudioChatSession(
  docId: string,
  sessionId: string,
  revision: number
): Promise<CharacterStudioChatState> {
  const deleted = exactObject(
    await deleteV2StudioSession(sessionId, revision),
    ['deleted', 'sessionId'],
    ['deleted', 'sessionId'],
    '角色聊天会话删除结果',
  )
  if (deleted.deleted !== true || deleted.sessionId !== sessionId) {
    throw new Error('角色聊天会话删除结果无效')
  }
  return refreshedChatState(docId)
}

export async function abortCharacterStudioChatOperation(
  sessionId: string,
  operationId: string
): Promise<CharacterStudioChatSession> {
  const aborted = exactObject(
    await abortV2StudioSession(sessionId, operationId),
    ['operationId', 'sessionGeneration', 'sessionRevision', 'status'],
    ['operationId', 'sessionGeneration', 'sessionRevision', 'status'],
    '角色聊天中止结果',
  )
  if (aborted.operationId !== operationId || aborted.status !== 'cancelled') {
    throw new Error('角色聊天中止结果无效')
  }
  integerValue(aborted.sessionGeneration, '角色聊天中止结果.sessionGeneration', 1)
  integerValue(aborted.sessionRevision, '角色聊天中止结果.sessionRevision', 1)
  const session = mapSession(await getV2StudioSession(sessionId))
  if (session.session_id !== sessionId) throw new Error('中止后的角色聊天会话身份不匹配')
  return session
}

export async function editCharacterStudioChatMessage(
  sessionId: string,
  baseRevision: number,
  messageId: string,
  content: string
): Promise<CharacterStudioChatSession> {
  const accepted = await editV2StudioMessage(messageId, baseRevision, content)
  const operation = await waitForOperation(accepted.operationId)
  if (operation.kind !== 'studio_chat' || operation.studioSessionId !== sessionId) {
    throw new Error('角色聊天编辑操作身份不匹配')
  }
  const session = mapSession(await getV2StudioSession(sessionId))
  if (session.session_id !== sessionId) throw new Error('编辑后的角色聊天会话身份不匹配')
  return session
}

export async function deleteCharacterStudioChatMessage(
  sessionId: string,
  baseRevision: number,
  messageId: string
): Promise<CharacterStudioChatSession> {
  const deleted = exactObject(
    await deleteV2StudioMessage(messageId, baseRevision),
    ['sessionId', 'sessionRevision', 'sessionGeneration'],
    ['sessionId', 'sessionRevision', 'sessionGeneration'],
    '角色聊天消息删除结果',
  )
  if (deleted.sessionId !== sessionId) throw new Error('角色聊天消息删除结果会话身份不匹配')
  integerValue(deleted.sessionRevision, '角色聊天消息删除结果.sessionRevision', 1)
  integerValue(deleted.sessionGeneration, '角色聊天消息删除结果.sessionGeneration', 1)
  const session = mapSession(await getV2StudioSession(sessionId))
  if (session.session_id !== sessionId) throw new Error('删除消息后的角色聊天会话身份不匹配')
  return session
}

export async function summarizeCharacterStudioChatSession(
  sessionId: string,
  baseRevision: number
): Promise<CharacterStudioChatSession> {
  const accepted = await summarizeV2StudioSession(sessionId, baseRevision)
  const operation = await waitForOperation(accepted.operationId)
  if (operation.kind !== 'studio_summary' || operation.studioSessionId !== sessionId) {
    throw new Error('角色聊天摘要操作身份不匹配')
  }
  const session = mapSession(await getV2StudioSession(sessionId))
  if (session.session_id !== sessionId) throw new Error('摘要后的角色聊天会话身份不匹配')
  return session
}

export async function exportCharacterStudioChatSession(
  sessionId: string
): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: v2StudioSessionExportUrl(sessionId),
    fallbackFilename: `${sessionId}.chat.json`,
    fallbackErrorMessage: '导出聊天记录失败',
  })
}

export async function importCharacterStudioChatSession(
  docId: string,
  baseIndexRevision: number,
  file: File
): Promise<CharacterStudioChatState> {
  const imported = mapSession(await importV2StudioSession(docId, baseIndexRevision, file))
  if (imported.doc_id !== docId) throw new Error('导入角色聊天会话文档身份不匹配')
  return refreshedChatState(docId)
}

export async function getCharacterStudioChatPromptPreview(sessionId: string): Promise<string> {
  const result = await getV2StudioPromptPreview(sessionId)
  const payload = exactObject(
    result,
    ['sessionId', 'promptPreview'],
    ['sessionId', 'promptPreview'],
    '角色提示词预览结果',
  )
  if (nonEmptyString(payload.sessionId, '角色提示词预览结果.sessionId') !== sessionId) {
    throw new Error('角色提示词预览会话身份不匹配')
  }
  return formatPromptPreview(payload.promptPreview)
}

export type CharacterStudioChatStreamEvent =
  | { type: 'assistant_delta'; delta: string; content: string }
  | { type: 'state'; session: CharacterStudioChatSession }

async function followStudioOperation(
  operationId: string,
  sessionId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  let content = ''
  let streamedCharacters = 0
  const operation = await waitForOperation(operationId, {
    signal,
    onEvent(event) {
      if (event.type !== 'chunk') return
      const payload = exactObject(
        event.payload,
        ['text', 'totalCharacters'],
        ['text', 'totalCharacters'],
        '角色聊天增量事件',
      )
      const delta = nonEmptyString(payload.text, '角色聊天增量事件.text')
      streamedCharacters += Array.from(delta).length
      if (
        integerValue(payload.totalCharacters, '角色聊天增量事件.totalCharacters', 1)
        !== streamedCharacters
      ) {
        throw new Error('角色聊天增量事件字符进度不一致')
      }
      content += delta
      onEvent({ type: 'assistant_delta', delta, content })
    },
  })
  if (operation.kind !== 'studio_chat' || operation.studioSessionId !== sessionId) {
    throw new Error('角色聊天操作身份不匹配')
  }
  const result = exactObject(
    operation.result,
    ['sessionId', 'sessionRevision', 'sessionGeneration', 'assistantMessageId'],
    ['sessionId', 'sessionRevision', 'sessionGeneration', 'assistantMessageId'],
    '角色聊天操作结果',
  )
  if (nonEmptyString(result.sessionId, '角色聊天操作结果.sessionId') !== sessionId) {
    throw new Error('角色聊天操作结果会话身份不匹配')
  }
  integerValue(result.sessionRevision, '角色聊天操作结果.sessionRevision', 1)
  integerValue(result.sessionGeneration, '角色聊天操作结果.sessionGeneration', 1)
  const assistantMessageId = nonEmptyString(
    result.assistantMessageId,
    '角色聊天操作结果.assistantMessageId',
  )
  const session = mapSession(await getV2StudioSession(sessionId))
  if (session.session_id !== sessionId) throw new Error('角色聊天会话身份不匹配')
  const assistant = session.messages.find(message => message.message_id === assistantMessageId)
  if (!assistant || assistant.role !== 'assistant' || !assistant.content.trim()) {
    throw new Error('角色聊天完成消息无效')
  }
  onEvent({ type: 'state', session })
}

export async function streamCharacterStudioChatMessage(payload: {
  sessionId: string
  baseSessionRevision: number
  content: string
  attachments?: File[]
  onEvent: (event: CharacterStudioChatStreamEvent) => void
  onAccepted?: (operationId: string) => void
  signal?: AbortSignal
}): Promise<void> {
  const assetIds: string[] = []
  for (const attachment of payload.attachments ?? []) {
    payload.signal?.throwIfAborted()
    const label = `角色聊天附件 ${attachment.name}`
    const asset = exactObject(
      await uploadV2StudioAsset(attachment),
      ['assetId', 'assetUrl', 'mimeType', 'byteSize', 'width', 'height'],
      ['assetId', 'assetUrl', 'mimeType', 'byteSize', 'width', 'height'],
      label,
    )
    nonEmptyString(asset.assetUrl, `${label}.assetUrl`)
    nonEmptyString(asset.mimeType, `${label}.mimeType`)
    integerValue(asset.byteSize, `${label}.byteSize`, 1)
    if (asset.width !== null) integerValue(asset.width, `${label}.width`, 1)
    if (asset.height !== null) integerValue(asset.height, `${label}.height`, 1)
    assetIds.push(nonEmptyString(asset.assetId, `${label}.assetId`))
  }
  const accepted = await sendV2StudioMessage(payload.sessionId, {
    baseSessionRevision: payload.baseSessionRevision,
    content: payload.content,
    assetIds,
  })
  payload.onAccepted?.(accepted.operationId)
  await followStudioOperation(
    accepted.operationId,
    payload.sessionId,
    payload.onEvent,
    payload.signal
  )
}

export async function regenerateCharacterStudioChatMessage(
  sessionId: string,
  baseRevision: number,
  messageId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal,
  onAccepted?: (operationId: string) => void
): Promise<void> {
  const accepted = await regenerateV2StudioMessage(messageId, baseRevision)
  onAccepted?.(accepted.operationId)
  await followStudioOperation(accepted.operationId, sessionId, onEvent, signal)
}

export async function importCharacterStudioFile(
  bookId: string,
  file: File
): Promise<CharacterStudioDocument> {
  const document = mapDocument(await importV2StudioDocument(bookId, file))
  if (document.bookId !== bookId) throw new Error('导入角色文档书籍身份不匹配')
  return document
}

export async function importWorldbookIntoCharacterStudioDocument(
  docId: string,
  baseRevision: number,
  file: File
): Promise<CharacterStudioDocument> {
  const document = mapDocument(await importV2StudioWorldbook(docId, baseRevision, file))
  if (document.id !== docId) throw new Error('导入世界书后的角色文档身份不匹配')
  return document
}

export function downloadCharacterStudioExport(
  docId: string,
  format: string
): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: v2StudioDocumentExportUrl(docId, format),
    fallbackFilename: `${docId}.${format}`,
    fallbackErrorMessage: '导出失败',
  })
}

export function downloadCharacterStudioWorldbook(
  docId: string
): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: v2StudioDocumentExportUrl(docId, 'worldbook'),
    fallbackFilename: `${docId}.worldbook.json`,
    fallbackErrorMessage: '导出世界书失败',
  })
}
