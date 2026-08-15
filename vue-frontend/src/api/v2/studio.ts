import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import { newIdempotencyKey } from './content'
import { parseOperationAccepted } from './operations'

const ROOT = '/api/v2/studio'

export type V2StudioCandidateList = components['schemas']['StudioCandidateList']
export type V2StudioChatState = components['schemas']['StudioChatState']
export type V2StudioDocument = components['schemas']['StudioDocument']
export type V2StudioDocumentContent = components['schemas']['StudioDocumentContent']
export type V2StudioIndex = components['schemas']['StudioIndex']
export type V2StudioOperationAccepted = components['schemas']['OperationAccepted']
export type V2StudioMessageChainMutation = components['schemas']['StudioMessageChainMutation']
export type V2StudioSession = components['schemas']['StudioChatSession']
type V2StudioDocumentMutation = components['schemas']['StudioDocumentMutation']
type V2StudioGenerateCommand = components['schemas']['StudioGenerateCommand']
type V2StudioMessageCommand = components['schemas']['StudioMessageCommand']
type V2StudioMessageEditCommand = components['schemas']['StudioMessageEditCommand']
type V2StudioSessionCreateCommand = components['schemas']['StudioSessionCreateCommand']

function commandHeaders(): Record<string, string> {
  return { 'Idempotency-Key': newIdempotencyKey() }
}

export function getV2StudioIndex(bookId: string): Promise<V2StudioIndex> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/index`)
}

export function getV2StudioCandidates(bookId: string): Promise<V2StudioCandidateList> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/candidates`)
}

export function createV2StudioDocument(
  bookId: string,
  command: components['schemas']['StudioDocumentCreateCommand'],
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
  command: V2StudioDocumentMutation,
): Promise<V2StudioDocument> {
  return apiClient.put(
    `${ROOT}/documents/${encodeURIComponent(documentId)}`,
    command,
    { headers: commandHeaders() },
  )
}

export function deleteV2StudioDocument(
  documentId: string,
): Promise<{ deleted: true; documentId: string }> {
  return apiClient.delete(
    `${ROOT}/documents/${encodeURIComponent(documentId)}`,
    { headers: commandHeaders() },
  )
}

export async function generateV2StudioDocument(
  documentId: string,
  baseRevision: number,
  section: V2StudioGenerateCommand['section'],
): Promise<V2StudioOperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.post<unknown>(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/generate`,
    { baseRevision, section },
    { headers: commandHeaders() },
  ), 'studio_generate')
}

export function validateV2StudioDocument(
  documentId: string,
  baseRevision: number,
): Promise<components['schemas']['StudioValidationResult']> {
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
  command: V2StudioSessionCreateCommand,
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

export function deleteV2StudioSession(
  sessionId: string,
  baseRevision: number,
): Promise<{ deleted: boolean; sessionId: string }> {
  return apiClient.delete(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}`,
    {
      headers: {
        ...commandHeaders(),
        'If-Match': String(baseRevision),
      },
    },
  )
}

export async function sendV2StudioMessage(
  sessionId: string,
  command: V2StudioMessageCommand,
): Promise<V2StudioOperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.post<unknown>(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    command,
    { headers: commandHeaders() },
  ), 'studio_chat')
}

export function abortV2StudioSession(
  sessionId: string,
  operationId: string,
): Promise<{
  operationId: string
  sessionGeneration: number
  sessionRevision: number
  status: 'cancelled'
}> {
  return apiClient.post(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/abort`,
    { operationId },
    { headers: commandHeaders() },
  )
}

export async function editV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
  content: V2StudioMessageEditCommand['content'],
): Promise<V2StudioOperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.put<unknown>(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}`,
    { baseSessionRevision, content },
    { headers: commandHeaders() },
  ), 'studio_chat')
}

export async function regenerateV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
): Promise<V2StudioOperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.post<unknown>(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}/regenerate`,
    { baseSessionRevision },
    { headers: commandHeaders() },
  ), 'studio_chat')
}

export function deleteV2StudioMessage(
  messageId: string,
  baseSessionRevision: number,
): Promise<V2StudioMessageChainMutation> {
  return apiClient.delete(
    `${ROOT}/chat/messages/${encodeURIComponent(messageId)}`,
    {
      data: { baseSessionRevision },
      headers: commandHeaders(),
    },
  )
}

export async function summarizeV2StudioSession(
  sessionId: string,
  baseSessionRevision: number,
): Promise<V2StudioOperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.post<unknown>(
    `${ROOT}/chat/sessions/${encodeURIComponent(sessionId)}/summarize`,
    { baseSessionRevision },
    { headers: commandHeaders() },
  ), 'studio_summary')
}

export function uploadV2StudioAsset(file: File): Promise<components['schemas']['StudioAsset']> {
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
  return apiClient.upload(
    `${ROOT}/documents/${encodeURIComponent(documentId)}/worldbook/import`,
    form,
    {
      headers: {
        ...commandHeaders(),
        'If-Match': String(baseRevision),
      },
    },
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

export function getV2StudioPromptPreview(
  sessionId: string,
): Promise<components['schemas']['StudioPromptPreviewResult']> {
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

export function v2StudioAgentUrl(documentId: string): string {
  return `${ROOT}/documents/${encodeURIComponent(documentId)}/agent`
}
