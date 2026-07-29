import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { newIdempotencyKey } from './content'

const ROOT = '/api/v2/studio'

export type V2StudioCandidate = components['schemas']['StudioCandidate']
export type V2StudioChatState = components['schemas']['StudioChatState']
export type V2StudioDocument = components['schemas']['StudioDocument']
export type V2StudioIndex = components['schemas']['StudioIndex']
export type V2StudioOperationAccepted = components['schemas']['OperationAccepted']
export type V2StudioSession = components['schemas']['StudioChatSession']

function commandHeaders(): Record<string, string> {
  return { 'Idempotency-Key': newIdempotencyKey() }
}

export function getV2StudioIndex(bookId: string): Promise<V2StudioIndex> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/index`)
}

export function getV2StudioCandidates(bookId: string): Promise<{
  available: boolean
  items: V2StudioCandidate[]
  reason: string | null
}> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/candidates`)
}

export function createV2StudioDocument(
  bookId: string,
  command: Record<string, unknown>,
): Promise<V2StudioDocument> {
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/documents`,
    command,
    { headers: commandHeaders() },
  )
}

export function getV2StudioDocument(documentId: string): Promise<V2StudioDocument> {
  return apiClient.get(`${ROOT}/documents/${encodeURIComponent(documentId)}`)
}

export function updateV2StudioDocument(
  documentId: string,
  command: {
    baseRevision: number
    document: Record<string, unknown>
    title?: string
  },
): Promise<V2StudioDocument> {
  return apiClient.put(
    `${ROOT}/documents/${encodeURIComponent(documentId)}`,
    command,
    { headers: commandHeaders() },
  )
}

export function deleteV2StudioDocument(documentId: string): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/documents/${encodeURIComponent(documentId)}`,
    { headers: commandHeaders() },
  )
}

export function generateV2StudioDocument(
  documentId: string,
  baseRevision: number,
  section: string,
): Promise<V2StudioOperationAccepted> {
  return apiClient.post(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/generate`,
    { baseRevision, section },
    { headers: commandHeaders() },
  )
}

export function validateV2StudioDocument(
  documentId: string,
  baseRevision: number,
): Promise<Record<string, unknown>> {
  return apiClient.post(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/validate`,
    { baseRevision },
    { headers: commandHeaders() },
  )
}

export function getV2StudioChatState(documentId: string): Promise<V2StudioChatState> {
  return apiClient.get(`${ROOT}/documents/${encodeURIComponent(documentId)}/chat`)
}

export function createV2StudioSession(
  documentId: string,
  command: {
    baseIndexRevision: number
    greetingId?: string
    title?: string
  },
): Promise<V2StudioSession> {
  return apiClient.post(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/chat/sessions`,
    command,
    { headers: commandHeaders() },
  )
}

export function getV2StudioSession(sessionId: string): Promise<V2StudioSession> {
  return apiClient.get(`${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}`)
}

export function activateV2StudioSession(
  sessionId: string,
  baseIndexRevision: number,
): Promise<V2StudioSession> {
  return apiClient.post(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/activate`,
    { baseIndexRevision },
    { headers: commandHeaders() },
  )
}

export function sendV2StudioMessage(
  sessionId: string,
  command: {
    assetIds: string[]
    baseSessionRevision: number
    content: string
  },
): Promise<V2StudioOperationAccepted> {
  return apiClient.post(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    command,
    { headers: commandHeaders() },
  )
}

export function editV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
  content: string,
): Promise<V2StudioOperationAccepted> {
  return apiClient.put(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}`,
    { baseSessionRevision, content },
    { headers: commandHeaders() },
  )
}

export function regenerateV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
): Promise<V2StudioOperationAccepted> {
  return apiClient.post(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}/regenerate`,
    { baseSessionRevision },
    { headers: commandHeaders() },
  )
}

export function deleteV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
): Promise<V2StudioSession> {
  return apiClient.delete(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}`,
    {
      data: { baseSessionRevision },
      headers: commandHeaders(),
    },
  )
}

export function summarizeV2StudioSession(
  sessionId: string,
  baseSessionRevision: number,
): Promise<V2StudioOperationAccepted> {
  return apiClient.post(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/summarize`,
    { baseSessionRevision },
    { headers: commandHeaders() },
  )
}

export function uploadV2StudioAsset(file: File): Promise<{ assetId: string; assetUrl: string }> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload(
    `${ROOT}/assets`,
    form,
    { headers: commandHeaders() },
  )
}

export function importV2StudioDocument(
  bookId: string,
  file: File,
): Promise<V2StudioDocument> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload(
    `${ROOT}/books/${encodeURIComponent(bookId)}/imports`,
    form,
    { headers: commandHeaders() },
  )
}

export function importV2StudioWorldbook(
  documentId: string,
  baseRevision: number,
  file: File,
): Promise<V2StudioDocument> {
  const form = new FormData()
  form.append('file', file)
  form.append('baseRevision', String(baseRevision))
  return apiClient.upload(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/worldbook/import`,
    form,
    { headers: commandHeaders() },
  )
}

export function importV2StudioSession(
  documentId: string,
  baseIndexRevision: number,
  file: File,
): Promise<V2StudioSession> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/chat/import`,
    form,
    {
      headers: {
        ...commandHeaders(),
        'If-Match': String(baseIndexRevision),
      },
    },
  )
}

export function getV2StudioPromptPreview(sessionId: string): Promise<{
  promptPreview: string
  sessionId: string
}> {
  return apiClient.get(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/prompt-preview`,
  )
}

export function v2StudioDocumentExportUrl(documentId: string, format: string): string {
  const query = new URLSearchParams({ format })
  return `${ROOT}/documents/${encodeURIComponent(documentId)}/export?${query}`
}

export function v2StudioSessionExportUrl(sessionId: string): string {
  return `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/export`
}

export function v2StudioOperationEventsUrl(operationId: string): string {
  return `/api/v2/operations/${encodeURIComponent(operationId)}/events?stream=1`
}

export function v2StudioAgentUrl(documentId: string): string {
  return `${ROOT}/documents/${encodeURIComponent(documentId)}/agent`
}
