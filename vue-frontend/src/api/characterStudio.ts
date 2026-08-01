import { downloadBlob, readApiErrorMessage } from './download'
import { readSseStream } from './sse'
import { getOperation, waitForOperation } from '@/api/v2/operations'
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
  v2StudioOperationEventsUrl,
  v2StudioSessionExportUrl,
  type V2StudioCandidate,
  type V2StudioChatState,
  type V2StudioDocument,
  type V2StudioSession,
} from '@/api/v2/studio'
import type {
  CharacterStudioChatAttachment,
  CharacterStudioChatMessage,
  CharacterStudioChatSession,
  CharacterStudioChatSessionSummary,
  CharacterStudioDocument,
  CharacterStudioChatState,
  CharacterStudioIndex,
  ExportDiagnostic,
} from '@/types/characterStudio'

const documentCache = new Map<string, V2StudioDocument>()
const chatStateCache = new Map<string, V2StudioChatState>()
const candidateCache = new Map<string, V2StudioCandidate[]>()
const sessionCache = new Map<string, V2StudioSession>()

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function formatPromptPreview(value: unknown): string {
  if (typeof value === 'string') return value
  const preview = record(value)
  const sections: string[] = []
  const system = typeof preview.system === 'string' ? preview.system.trim() : ''
  if (system) sections.push(`[system]\n${system}`)

  for (const item of array(preview.messages)) {
    const message = record(item)
    const role =
      typeof message.role === 'string' && message.role.trim() ? message.role.trim() : 'message'
    const content = typeof message.content === 'string' ? message.content : ''
    const assetIds = array(message.assetIds).map(String).filter(Boolean)
    const assets = assetIds.length ? `\n[assets] ${assetIds.join(', ')}` : ''
    sections.push(`[${role}]\n${content}${assets}`)
  }

  const lorebookHits = array(preview.lorebookHits)
  if (lorebookHits.length) {
    sections.push(`[lorebook]\n${JSON.stringify(lorebookHits, null, 2)}`)
  }
  if (sections.length) return sections.join('\n\n')
  if (value && typeof value === 'object') return JSON.stringify(value, null, 2)
  return ''
}

function mapDiagnostics(value: unknown): ExportDiagnostic | null {
  const diagnostics = record(value)
  if (!Object.keys(diagnostics).length) return null
  return {
    valid: Boolean(diagnostics.valid),
    errors: array(diagnostics.errors).map(String),
    warnings: array(diagnostics.warnings).map(String),
    checks: record(diagnostics.checks) as Record<string, boolean>,
  }
}

function mapDocument(raw: V2StudioDocument): CharacterStudioDocument {
  documentCache.set(raw.id, raw)
  const origin = record(raw.origin)
  const status = record(raw.status)
  const meta = record(raw.meta)
  return {
    id: raw.id,
    bookId: raw.bookId,
    origin: {
      type: origin.type === 'analysis' || origin.type === 'imported' ? origin.type : 'manual',
      source_character:
        typeof origin.source_character === 'string' ? origin.source_character : null,
    },
    status: {
      is_favorite: Boolean(status.is_favorite),
      frozen_sections: array(status.frozen_sections).map(String),
      last_diagnostics: mapDiagnostics(status.last_diagnostics),
      last_validated_at:
        typeof status.last_validated_at === 'string' ? status.last_validated_at : null,
    },
    meta: {
      title: String(meta.title ?? raw.title),
      tags: array(meta.tags).map(String),
      created_at: raw.createdAt ?? '',
      updated_at: raw.updatedAt ?? '',
    },
    identity: {
      name: String(record(raw.identity).name ?? raw.title),
      aliases: array(record(raw.identity).aliases).map(String),
      description: String(record(raw.identity).description ?? ''),
      personality: String(record(raw.identity).personality ?? ''),
      scenario: String(record(raw.identity).scenario ?? ''),
    },
    coreMessages: {
      first_message: String(record(raw.coreMessages).first_message ?? ''),
      message_example: String(record(raw.coreMessages).message_example ?? ''),
      alternate_greetings: array(record(raw.coreMessages).alternate_greetings).map(String),
      system_prompt: String(record(raw.coreMessages).system_prompt ?? ''),
      post_history_instructions: String(record(raw.coreMessages).post_history_instructions ?? ''),
      creator_notes: String(record(raw.coreMessages).creator_notes ?? ''),
      character_version: String(record(raw.coreMessages).character_version ?? ''),
    },
    lorebook: {
      name: String(record(raw.lorebook).name ?? ''),
      entries: array(
        record(raw.lorebook).entries
      ) as CharacterStudioDocument['lorebook']['entries'],
    },
    regexScripts: array(raw.regexScripts) as CharacterStudioDocument['regexScripts'],
    stateTasks: array(raw.stateTasks) as CharacterStudioDocument['stateTasks'],
    exportArtifacts: record(raw.exportArtifacts),
    revision: raw.revision,
    avatarUrl: raw.avatarUrl,
    createdAt: raw.createdAt,
    updatedAt: raw.updatedAt,
  }
}

function cachedDocument(documentId: string): V2StudioDocument {
  const cached = documentCache.get(documentId)
  if (cached) return cached
  throw new Error('角色文档版本缺失，请重新加载')
}

function mapAttachment(value: Record<string, unknown>): CharacterStudioChatAttachment {
  const assetId = String(value.assetId ?? '')
  return {
    attachment_id: assetId,
    filename: String(value.filename ?? assetId),
    mime_type: String(value.mimeType ?? 'application/octet-stream'),
    asset_path: String(value.assetUrl ?? ''),
    created_at: String(value.createdAt ?? ''),
  }
}

function mapMessage(value: Record<string, unknown>): CharacterStudioChatMessage {
  return {
    message_id: String(value.messageId ?? ''),
    role: value.role === 'user' ? 'user' : 'assistant',
    content: String(value.content ?? ''),
    attachments: array(value.attachments).map(item => mapAttachment(record(item))),
    runtime_log: array(value.runtimeLog) as Array<Record<string, unknown>>,
    variables_snapshot: record(value.variablesSnapshot),
    generation_meta: record(value.generationMeta),
    created_at: String(value.createdAt ?? ''),
    updated_at: String(value.updatedAt ?? ''),
  }
}

function mapSession(raw: V2StudioSession): CharacterStudioChatSession {
  sessionCache.set(raw.sessionId, raw)
  return {
    session_id: raw.sessionId,
    doc_id: raw.documentId,
    title: raw.title,
    created_at: String(raw.createdAt ?? ''),
    updated_at: String(raw.updatedAt ?? ''),
    archived_at: raw.archived ? String(raw.updatedAt ?? '') : null,
    greeting_source: record(raw.greetingSource),
    summary_blocks: raw.summaryBlocks.map(block => ({
      summary_id: String(block.summaryId ?? ''),
      content: String(block.content ?? ''),
      created_at: String(block.createdAt ?? ''),
      covered_message_ids: array(block.coveredMessageIds).map(String),
    })),
    messages: raw.messages.map(mapMessage),
    variables: raw.variables,
    last_prompt_preview: '',
    revision: raw.revision,
    generation: raw.generation,
  }
}

function mapSessionSummary(value: Record<string, unknown>): CharacterStudioChatSessionSummary {
  return {
    session_id: String(value.sessionId ?? ''),
    title: String(value.title ?? ''),
    revision: Number(value.revision ?? 0) || undefined,
    generation: Number(value.generation ?? 0) || undefined,
    message_count: Number(value.messageCount ?? 0),
    updated_at: String(value.updatedAt ?? ''),
    archived_at: value.archived ? String(value.updatedAt ?? '') : null,
    last_message_excerpt: String(value.lastMessageExcerpt ?? ''),
  }
}

function mapChatState(raw: V2StudioChatState): CharacterStudioChatState {
  chatStateCache.set(raw.documentId, raw)
  return {
    doc_id: raw.documentId,
    active_session: raw.activeSession ? mapSession(raw.activeSession) : undefined,
    archived_sessions: raw.sessions.filter(item => Boolean(item.archived)).map(mapSessionSummary),
    available_greetings: array(raw.availableGreetings).map(item => {
      const greeting = record(item)
      return {
        greeting_id: String(greeting.greetingId ?? ''),
        label: String(greeting.label ?? ''),
        content: String(greeting.content ?? ''),
        source: record(greeting.source),
      }
    }),
  }
}

async function refreshedChatState(documentId: string): Promise<CharacterStudioChatState> {
  return mapChatState(await getV2StudioChatState(documentId))
}

function cachedSession(sessionId: string): V2StudioSession {
  const cached = sessionCache.get(sessionId)
  if (cached) return cached
  throw new Error('聊天会话版本缺失，请重新加载')
}

export async function getCharacterStudioIndex(bookId: string): Promise<CharacterStudioIndex> {
  const [index, candidates] = await Promise.all([
    getV2StudioIndex(bookId),
    getV2StudioCandidates(bookId),
  ])
  candidateCache.set(bookId, candidates.items)
  return {
    book_id: bookId,
    documents: index.documents.map(item => ({
      id: item.documentId,
      title: item.title,
      origin: item.kind === 'analysis' || item.kind === 'imported' ? item.kind : 'manual',
      source_character: item.sourceCharacter,
      updated_at: item.updatedAt,
      tags: item.tags,
      is_favorite: item.isFavorite,
      has_avatar: item.hasAvatar,
      sample_pages: [],
    })),
    candidates: candidates.items.map(item => ({
      name: item.name,
      aliases: item.aliases,
      first_appearance: item.firstAppearancePage ?? 0,
      dialogue_count: item.keyMomentCount,
      has_dialogues: item.keyMomentCount > 0,
      sample_pages: item.relatedPageNumbers,
    })),
    count: index.documents.length,
    has_timeline: candidates.available,
  }
}

export async function createCharacterStudioDocument(
  bookId: string,
  payload?: { candidate_name?: string; title?: string }
): Promise<CharacterStudioDocument> {
  const candidate = candidateCache.get(bookId)?.find(item => item.name === payload?.candidate_name)
  const document = await createV2StudioDocument(bookId, {
    title: payload?.title ?? candidate?.name ?? '新角色',
    ...(candidate ? { candidate } : {}),
  })
  return mapDocument(document)
}

export async function getCharacterStudioDocument(docId: string): Promise<CharacterStudioDocument> {
  return mapDocument(await getV2StudioDocument(docId))
}

export async function saveCharacterStudioDocument(
  docId: string,
  payload: CharacterStudioDocument
): Promise<CharacterStudioDocument> {
  if (!payload.revision || payload.revision < 1) {
    throw new Error('角色文档版本缺失，请重新加载')
  }
  const document = await updateV2StudioDocument(docId, {
    baseRevision: payload.revision,
    title: payload.meta.title,
    document: payload as unknown as Record<string, unknown>,
  })
  return mapDocument(document)
}

export async function deleteCharacterStudioDocument(docId: string): Promise<void> {
  await deleteV2StudioDocument(docId)
  documentCache.delete(docId)
  chatStateCache.delete(docId)
}

export async function generateCharacterStudioSection(
  docId: string,
  section: string
): Promise<CharacterStudioDocument> {
  const current = cachedDocument(docId)
  const accepted = await generateV2StudioDocument(docId, current.revision, section)
  await waitForOperation(accepted.operationId)
  return mapDocument(await getV2StudioDocument(docId))
}

export async function validateCharacterStudioDocument(docId: string): Promise<
  ExportDiagnostic & {
    document: CharacterStudioDocument
  }
> {
  const current = cachedDocument(docId)
  const response = await validateV2StudioDocument(docId, current.revision)
  const diagnostics = record(response.diagnostics)
  const refreshed = await getV2StudioDocument(docId)
  const document = mapDocument(refreshed)
  return {
    valid: Boolean(diagnostics.valid),
    errors: array(diagnostics.errors).map(String),
    warnings: array(diagnostics.warnings).map(String),
    checks: record(diagnostics.checks) as Record<string, boolean>,
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
  let content = ''
  let streamError = ''
  await readSseStream<Record<string, unknown>>(response, {
    missingBodyMessage: '无法读取 Agent 响应流',
    parseErrorMessage: 'Agent 响应格式无效',
    onMessage(event) {
      if (event.event === 'chunk') content += String(event.data.text ?? '')
      if (event.event === 'error') streamError = String(event.data.message ?? 'Agent 调用失败')
    },
  })
  if (streamError) throw new Error(streamError)
  return content
}

export async function getCharacterStudioChatState(
  docId: string
): Promise<CharacterStudioChatState> {
  return refreshedChatState(docId)
}

export async function createCharacterStudioChatSession(
  docId: string,
  greetingId?: string
): Promise<CharacterStudioChatState> {
  const state = chatStateCache.get(docId) ?? (await getV2StudioChatState(docId))
  await createV2StudioSession(docId, {
    baseIndexRevision: state.indexRevision,
    ...(greetingId ? { greetingId } : {}),
  })
  return refreshedChatState(docId)
}

export async function switchCharacterStudioChatSession(
  docId: string,
  sessionId: string
): Promise<CharacterStudioChatState> {
  const state = chatStateCache.get(docId) ?? (await getV2StudioChatState(docId))
  await activateV2StudioSession(sessionId, state.indexRevision)
  return refreshedChatState(docId)
}

export async function deleteCharacterStudioChatSession(
  docId: string,
  sessionId: string,
  revision?: number
): Promise<CharacterStudioChatState> {
  const current =
    revision ??
    sessionCache.get(sessionId)?.revision ??
    (await getV2StudioSession(sessionId)).revision
  await deleteV2StudioSession(sessionId, current)
  sessionCache.delete(sessionId)
  return refreshedChatState(docId)
}

export async function abortCharacterStudioChatOperation(
  sessionId: string,
  operationId: string
): Promise<CharacterStudioChatSession> {
  await abortV2StudioSession(sessionId, operationId)
  return mapSession(await getV2StudioSession(sessionId))
}

export async function editCharacterStudioChatMessage(
  sessionId: string,
  messageId: string,
  content: string
): Promise<CharacterStudioChatSession> {
  const session = cachedSession(sessionId)
  const accepted = await editV2StudioMessage(messageId, session.revision, content)
  await waitForOperation(accepted.operationId)
  return mapSession(await getV2StudioSession(sessionId))
}

export async function deleteCharacterStudioChatMessage(
  sessionId: string,
  messageId: string
): Promise<CharacterStudioChatSession> {
  const session = cachedSession(sessionId)
  await deleteV2StudioMessage(messageId, session.revision)
  return mapSession(await getV2StudioSession(sessionId))
}

export async function summarizeCharacterStudioChatSession(
  sessionId: string
): Promise<CharacterStudioChatSession> {
  const session = cachedSession(sessionId)
  const accepted = await summarizeV2StudioSession(sessionId, session.revision)
  await waitForOperation(accepted.operationId)
  return mapSession(await getV2StudioSession(sessionId))
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
  file: File
): Promise<CharacterStudioChatState> {
  const state = chatStateCache.get(docId) ?? (await getV2StudioChatState(docId))
  await importV2StudioSession(docId, state.indexRevision, file)
  return refreshedChatState(docId)
}

export async function getCharacterStudioChatPromptPreview(sessionId: string): Promise<string> {
  const result = await getV2StudioPromptPreview(sessionId)
  return formatPromptPreview(result.promptPreview)
}

export type CharacterStudioChatStreamEvent =
  | { type: 'assistant_delta'; delta: string; content: string }
  | { type: 'assistant_done'; message_id: string; content: string }
  | {
      type: 'runtime'
      runtime_log: Array<Record<string, unknown>>
      variables: Record<string, unknown>
    }
  | { type: 'state'; session: unknown }
  | { type: 'error'; message: string }
  | { type: 'heartbeat'; ok: boolean }

async function followStudioOperation(
  operationId: string,
  sessionId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(v2StudioOperationEventsUrl(operationId), {
    headers: { Accept: 'text/event-stream' },
    signal,
  })
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, '聊天事件订阅失败'))
  }
  let content = ''
  await readSseStream<Record<string, unknown>>(response, {
    missingBodyMessage: '无法读取聊天事件流',
    parseErrorMessage: '解析聊天事件流失败',
    onMessage(message) {
      if (message.event !== 'chunk') return
      const payload = record(message.data.payload)
      const delta = String(payload.text ?? '')
      content += delta
      onEvent({ type: 'assistant_delta', delta, content })
    },
  })
  const operation = await getOperation(operationId, signal)
  if (operation.status !== 'completed') {
    throw new Error(
      operation.error?.message ||
        (operation.status === 'cancelled' ? '聊天生成已取消' : '聊天生成失败')
    )
  }
  const session = mapSession(await getV2StudioSession(sessionId))
  const assistant = [...session.messages].reverse().find(message => message.role === 'assistant')
  onEvent({
    type: 'assistant_done',
    message_id: assistant?.message_id ?? '',
    content: assistant?.content ?? content,
  })
  onEvent({ type: 'state', session })
}

export async function streamCharacterStudioChatMessage(payload: {
  sessionId: string
  content: string
  attachments?: File[]
  onEvent: (event: CharacterStudioChatStreamEvent) => void
  onAccepted?: (operationId: string) => void
  signal?: AbortSignal
}): Promise<void> {
  const session = cachedSession(payload.sessionId)
  const assets = await Promise.all((payload.attachments ?? []).map(uploadV2StudioAsset))
  const accepted = await sendV2StudioMessage(payload.sessionId, {
    baseSessionRevision: session.revision,
    content: payload.content,
    assetIds: assets.map(asset => asset.assetId),
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
  messageId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal,
  onAccepted?: (operationId: string) => void
): Promise<void> {
  const session = cachedSession(sessionId)
  const accepted = await regenerateV2StudioMessage(messageId, session.revision)
  onAccepted?.(accepted.operationId)
  await followStudioOperation(accepted.operationId, sessionId, onEvent, signal)
}

export async function importCharacterStudioFile(
  bookId: string,
  file: File
): Promise<CharacterStudioDocument> {
  return mapDocument(await importV2StudioDocument(bookId, file))
}

export async function importWorldbookIntoCharacterStudioDocument(
  docId: string,
  file: File
): Promise<CharacterStudioDocument> {
  const current = cachedDocument(docId)
  return mapDocument(await importV2StudioWorldbook(docId, current.revision, file))
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

export function getCharacterStudioAvatarUrl(docId: string): string {
  return documentCache.get(docId)?.avatarUrl ?? ''
}

export function getCharacterStudioChatAttachmentUrl(assetPath: string): string {
  return assetPath
}
