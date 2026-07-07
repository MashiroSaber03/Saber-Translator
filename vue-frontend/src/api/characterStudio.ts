import { apiClient } from './client'
import { downloadBlob, readApiErrorMessage } from './download'
import { readSseStream } from './sse'
import type {
  CharacterStudioDocument,
  CharacterStudioChatStateResponse,
  CharacterStudioDocumentResponse,
  CharacterStudioIndexResponse,
  ExportDiagnostic,
} from '@/types/characterStudio'

function characterStudioPathSegment(value: string): string {
  return encodeURIComponent(value)
}

function characterStudioEndpoint(bookId: string, suffix = ''): string {
  return `/api/manga-insight/${characterStudioPathSegment(bookId)}/character-studio${suffix}`
}

function characterStudioDocumentEndpoint(bookId: string, docId: string, suffix = ''): string {
  return characterStudioEndpoint(bookId, `/documents/${characterStudioPathSegment(docId)}${suffix}`)
}

function characterStudioQuery(params: Record<string, string | null | undefined>): string {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      query.set(key, value)
    }
  }
  const serialized = query.toString()
  return serialized ? `?${serialized}` : ''
}

export async function getCharacterStudioIndex(bookId: string): Promise<CharacterStudioIndexResponse> {
  return apiClient.get<CharacterStudioIndexResponse>(characterStudioEndpoint(bookId, '/index'))
}

export async function getCharacterStudioCandidates(bookId: string): Promise<CharacterStudioIndexResponse> {
  return apiClient.get<CharacterStudioIndexResponse>(characterStudioEndpoint(bookId, '/candidates'))
}

export async function createCharacterStudioDocument(
  bookId: string,
  payload?: { candidate_name?: string; title?: string }
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.post<CharacterStudioDocumentResponse>(characterStudioEndpoint(bookId, '/documents'), payload || {})
}

export async function getCharacterStudioDocument(bookId: string, docId: string): Promise<CharacterStudioDocumentResponse> {
  return apiClient.get<CharacterStudioDocumentResponse>(characterStudioDocumentEndpoint(bookId, docId))
}

export async function saveCharacterStudioDocument(
  bookId: string,
  docId: string,
  payload: CharacterStudioDocument
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.put<CharacterStudioDocumentResponse>(
    characterStudioDocumentEndpoint(bookId, docId),
    payload
  )
}

export async function deleteCharacterStudioDocument(bookId: string, docId: string): Promise<{ success: boolean; error?: string; message?: string }> {
  return apiClient.delete(characterStudioDocumentEndpoint(bookId, docId))
}

export async function generateCharacterStudioSection(
  bookId: string,
  docId: string,
  section: string
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.post<CharacterStudioDocumentResponse>(
    characterStudioDocumentEndpoint(bookId, docId, `/generate/${characterStudioPathSegment(section)}`),
    {}
  )
}

export async function validateCharacterStudioDocument(bookId: string, docId: string): Promise<ExportDiagnostic & { success: boolean; message?: string; error?: string }> {
  return apiClient.post(characterStudioDocumentEndpoint(bookId, docId, '/validate'), {})
}

export async function runCharacterStudioAgent(
  bookId: string,
  docId: string,
  message: string
): Promise<{ success: boolean; content?: string; context?: string; error?: string; message?: string }> {
  return apiClient.post(characterStudioDocumentEndpoint(bookId, docId, '/agent'), { message })
}

export async function getCharacterStudioChatState(
  bookId: string,
  docId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.get(characterStudioDocumentEndpoint(bookId, docId, '/chat'))
}

export async function createCharacterStudioChatSession(
  bookId: string,
  docId: string,
  greetingId?: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.post(characterStudioDocumentEndpoint(bookId, docId, '/chat/sessions'), {
    greeting_id: greetingId || null,
  })
}

export async function switchCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.post(
    characterStudioDocumentEndpoint(bookId, docId, `/chat/sessions/${characterStudioPathSegment(sessionId)}/activate`),
    {}
  )
}

export async function editCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  sessionId: string,
  messageId: string,
  content: string,
): Promise<{ success: boolean; session?: unknown; error?: string; message?: string }> {
  return apiClient.put(characterStudioDocumentEndpoint(
    bookId,
    docId,
    `/chat/messages/${characterStudioPathSegment(messageId)}`
  ), {
    session_id: sessionId,
    content,
  })
}

export async function deleteCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  sessionId: string,
  messageId: string,
): Promise<{ success: boolean; session?: unknown; error?: string; message?: string }> {
  return apiClient.delete(
    characterStudioDocumentEndpoint(
      bookId,
      docId,
      `/chat/messages/${characterStudioPathSegment(messageId)}${characterStudioQuery({ session_id: sessionId })}`
    )
  )
}

export async function summarizeCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
  cutoffMessageId?: string,
): Promise<{ success: boolean; session?: unknown; error?: string; message?: string }> {
  return apiClient.post(characterStudioDocumentEndpoint(bookId, docId, '/chat/summary'), {
    session_id: sessionId,
    cutoff_message_id: cutoffMessageId || null,
  })
}

export async function exportCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: characterStudioDocumentEndpoint(bookId, docId, `/chat/export${characterStudioQuery({ session_id: sessionId })}`),
    fallbackFilename: `${docId}.${sessionId}.chat.json`,
    fallbackErrorMessage: '导出聊天记录失败',
  })
}

export async function importCharacterStudioChatSession(
  bookId: string,
  docId: string,
  file: File,
): Promise<CharacterStudioChatStateResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload(characterStudioDocumentEndpoint(bookId, docId, '/chat/import'), form)
}

export async function getCharacterStudioChatPromptPreview(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.get(
    characterStudioDocumentEndpoint(bookId, docId, `/chat/prompt-preview${characterStudioQuery({ session_id: sessionId })}`)
  )
}

export type CharacterStudioChatStreamEvent =
  | { type: 'assistant_delta'; delta: string; content: string }
  | { type: 'assistant_done'; message_id: string; content: string }
  | { type: 'runtime'; runtime_log: Array<Record<string, unknown>>; variables: Record<string, unknown> }
  | { type: 'state'; session: unknown }
  | { type: 'error'; message: string }
  | { type: 'heartbeat'; ok: boolean }

async function readCharacterStudioChatStream(
  response: Response,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
): Promise<void> {
  await readSseStream<Record<string, unknown>>(response, {
    missingBodyMessage: '无法读取聊天事件流',
    parseErrorMessage: '解析聊天事件流失败',
    onMessage(message) {
      const event = {
        type: message.event,
        ...message.data,
      } as CharacterStudioChatStreamEvent
      onEvent(event)
      if (event.type === 'error') {
        throw new Error(event.message || '聊天事件流返回错误')
      }
    },
  })
}

export async function streamCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  payload: {
    sessionId: string
    content: string
    attachments?: File[]
    onEvent: (event: CharacterStudioChatStreamEvent) => void
    signal?: AbortSignal
  },
): Promise<void> {
  const form = new FormData()
  form.append('session_id', payload.sessionId)
  form.append('content', payload.content)
  for (const file of payload.attachments || []) {
    form.append(file.name || 'attachment', file)
  }
  const response = await fetch(characterStudioDocumentEndpoint(bookId, docId, '/chat/messages/stream'), {
    method: 'POST',
    body: form,
    signal: payload.signal,
    headers: { Accept: 'text/event-stream' },
  })
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, '聊天消息发送失败'))
  }
  await readCharacterStudioChatStream(response, payload.onEvent)
}

export async function regenerateCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  sessionId: string,
  messageId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(characterStudioDocumentEndpoint(
    bookId,
    docId,
    `/chat/messages/${characterStudioPathSegment(messageId)}/regenerate/stream`
  ), {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
    signal,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
  })
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, '消息重生失败'))
  }
  await readCharacterStudioChatStream(response, onEvent)
}

export async function importCharacterStudioFile(
  bookId: string,
  file: File
): Promise<CharacterStudioDocumentResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload<CharacterStudioDocumentResponse>(characterStudioEndpoint(bookId, '/imports'), form)
}

export async function importWorldbookIntoCharacterStudioDocument(
  bookId: string,
  docId: string,
  file: File
): Promise<CharacterStudioDocumentResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload<CharacterStudioDocumentResponse>(
    characterStudioDocumentEndpoint(bookId, docId, '/worldbook/import'),
    form
  )
}

export async function downloadCharacterStudioExport(bookId: string, docId: string, format: string): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: characterStudioDocumentEndpoint(bookId, docId, `/export${characterStudioQuery({ format })}`),
    fallbackFilename: `${docId}.${format}`,
    fallbackErrorMessage: '导出失败',
  })
}

export async function downloadCharacterStudioWorldbook(bookId: string, docId: string): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: characterStudioDocumentEndpoint(bookId, docId, '/worldbook/export'),
    fallbackFilename: `${docId}.worldbook.json`,
    fallbackErrorMessage: '导出世界书失败',
  })
}

export function getCharacterStudioAvatarUrl(bookId: string, docId: string): string {
  return characterStudioDocumentEndpoint(bookId, docId, '/avatar')
}

export function getCharacterStudioChatAttachmentUrl(bookId: string, docId: string, assetPath: string): string {
  return characterStudioDocumentEndpoint(bookId, docId, `/chat/attachment${characterStudioQuery({ asset_path: assetPath })}`)
}
