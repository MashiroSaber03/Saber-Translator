import { apiClient } from './client'
import type {
  CharacterStudioChatStateResponse,
  CharacterStudioDocumentResponse,
  CharacterStudioIndexResponse,
  ExportDiagnostic,
} from '@/types/characterStudio'

async function parseApiErrorMessage(response: Response, fallback: string): Promise<string> {
  const text = await response.text()
  if (!text) return fallback
  try {
    const parsed = JSON.parse(text) as { error?: string; message?: string }
    return parsed.error || parsed.message || text
  } catch {
    return text
  }
}

export async function getCharacterStudioIndex(bookId: string): Promise<CharacterStudioIndexResponse> {
  return apiClient.get<CharacterStudioIndexResponse>(`/api/manga-insight/${bookId}/character-studio/index`)
}

export async function getCharacterStudioCandidates(bookId: string): Promise<CharacterStudioIndexResponse> {
  return apiClient.get<CharacterStudioIndexResponse>(`/api/manga-insight/${bookId}/character-studio/candidates`)
}

export async function createCharacterStudioDocument(
  bookId: string,
  payload?: { candidate_name?: string; title?: string }
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.post<CharacterStudioDocumentResponse>(`/api/manga-insight/${bookId}/character-studio/documents`, payload || {})
}

export async function getCharacterStudioDocument(bookId: string, docId: string): Promise<CharacterStudioDocumentResponse> {
  return apiClient.get<CharacterStudioDocumentResponse>(`/api/manga-insight/${bookId}/character-studio/documents/${docId}`)
}

export async function saveCharacterStudioDocument(
  bookId: string,
  docId: string,
  payload: Record<string, unknown>
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.put<CharacterStudioDocumentResponse>(
    `/api/manga-insight/${bookId}/character-studio/documents/${docId}`,
    payload
  )
}

export async function deleteCharacterStudioDocument(bookId: string, docId: string): Promise<{ success: boolean; error?: string; message?: string }> {
  return apiClient.delete(`/api/manga-insight/${bookId}/character-studio/documents/${docId}`)
}

export async function generateCharacterStudioSection(
  bookId: string,
  docId: string,
  section: string
): Promise<CharacterStudioDocumentResponse> {
  return apiClient.post<CharacterStudioDocumentResponse>(
    `/api/manga-insight/${bookId}/character-studio/documents/${docId}/generate/${section}`,
    {}
  )
}

export async function validateCharacterStudioDocument(bookId: string, docId: string): Promise<ExportDiagnostic & { success: boolean; message?: string; error?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/validate`, {})
}

export async function runCharacterStudioAgent(
  bookId: string,
  docId: string,
  message: string
): Promise<{ success: boolean; content?: string; context?: string; error?: string; message?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/agent`, { message })
}

export async function getCharacterStudioChatState(
  bookId: string,
  docId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.get(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat`)
}

export async function createCharacterStudioChatSession(
  bookId: string,
  docId: string,
  greetingId?: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/sessions`, {
    greeting_id: greetingId || null,
  })
}

export async function switchCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/sessions/${encodeURIComponent(sessionId)}/activate`, {})
}

export async function editCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  sessionId: string,
  messageId: string,
  content: string,
): Promise<{ success: boolean; session?: unknown; error?: string; message?: string }> {
  return apiClient.put(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/messages/${encodeURIComponent(messageId)}`, {
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
  return apiClient.delete(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/messages/${encodeURIComponent(messageId)}?session_id=${encodeURIComponent(sessionId)}`)
}

export async function summarizeCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
  cutoffMessageId?: string,
): Promise<{ success: boolean; session?: unknown; error?: string; message?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/summary`, {
    session_id: sessionId,
    cutoff_message_id: cutoffMessageId || null,
  })
}

export async function exportCharacterStudioChatSession(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/export?session_id=${encodeURIComponent(sessionId)}`)
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '导出聊天记录失败'))
  }
  const disposition = response.headers.get('content-disposition') || ''
  const match = disposition.match(/filename="?([^";]+)"?/)
  return {
    blob: await response.blob(),
    filename: match?.[1] || `${docId}.${sessionId}.chat.json`,
  }
}

export async function importCharacterStudioChatSession(
  bookId: string,
  docId: string,
  file: File,
): Promise<CharacterStudioChatStateResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/import`, form)
}

export async function getCharacterStudioChatPromptPreview(
  bookId: string,
  docId: string,
  sessionId: string,
): Promise<CharacterStudioChatStateResponse> {
  return apiClient.get(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/prompt-preview?session_id=${encodeURIComponent(sessionId)}`)
}

export type CharacterStudioChatStreamEvent =
  | { type: 'assistant_delta'; delta: string; content: string }
  | { type: 'assistant_done'; message_id: string; content: string }
  | { type: 'runtime'; runtime_log: Array<Record<string, unknown>>; variables: Record<string, unknown> }
  | { type: 'state'; session: unknown }
  | { type: 'error'; message: string }
  | { type: 'heartbeat'; ok: boolean }

async function readSseResponse(
  response: Response,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
): Promise<void> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('无法读取聊天事件流')
  }
  const decoder = new TextDecoder()
  let buffer = ''
  let eventType = ''
  let eventData = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const rawLine of lines) {
        const line = rawLine.trimEnd()
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventData = line.slice(5).trim()
        } else if (line === '' && eventType && eventData) {
          const parsed = JSON.parse(eventData) as Record<string, unknown>
          const event = { type: eventType, ...parsed } as CharacterStudioChatStreamEvent
          onEvent(event)
          if (event.type === 'error') {
            throw new Error(event.message || '聊天事件流返回错误')
          }
          eventType = ''
          eventData = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
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
  const response = await fetch(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/messages/stream`, {
    method: 'POST',
    body: form,
    signal: payload.signal,
    headers: { Accept: 'text/event-stream' },
  })
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '聊天消息发送失败'))
  }
  await readSseResponse(response, payload.onEvent)
}

export async function regenerateCharacterStudioChatMessage(
  bookId: string,
  docId: string,
  sessionId: string,
  messageId: string,
  onEvent: (event: CharacterStudioChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/messages/${encodeURIComponent(messageId)}/regenerate/stream`, {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
    signal,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
  })
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '消息重生失败'))
  }
  await readSseResponse(response, onEvent)
}

export async function importCharacterStudioFile(
  bookId: string,
  file: File
): Promise<CharacterStudioDocumentResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload<CharacterStudioDocumentResponse>(`/api/manga-insight/${bookId}/character-studio/imports`, form)
}

export async function importWorldbookIntoCharacterStudioDocument(
  bookId: string,
  docId: string,
  file: File
): Promise<CharacterStudioDocumentResponse> {
  const form = new FormData()
  form.append('file', file)
  return apiClient.upload<CharacterStudioDocumentResponse>(
    `/api/manga-insight/${bookId}/character-studio/documents/${docId}/worldbook/import`,
    form
  )
}

export async function downloadCharacterStudioExport(bookId: string, docId: string, format: string): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/export?format=${encodeURIComponent(format)}`)
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '导出失败'))
  }
  const disposition = response.headers.get('content-disposition') || ''
  const match = disposition.match(/filename="?([^";]+)"?/)
  const filename = match?.[1] || `${docId}.${format}`
  return { blob: await response.blob(), filename }
}

export async function downloadCharacterStudioWorldbook(bookId: string, docId: string): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/worldbook/export`)
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '导出世界书失败'))
  }
  const disposition = response.headers.get('content-disposition') || ''
  const match = disposition.match(/filename="?([^";]+)"?/)
  const filename = match?.[1] || `${docId}.worldbook.json`
  return { blob: await response.blob(), filename }
}

export function getCharacterStudioAvatarUrl(bookId: string, docId: string): string {
  return `/api/manga-insight/${bookId}/character-studio/documents/${docId}/avatar`
}

export function getCharacterStudioChatAttachmentUrl(bookId: string, docId: string, assetPath: string): string {
  return `/api/manga-insight/${bookId}/character-studio/documents/${docId}/chat/attachment?asset_path=${encodeURIComponent(assetPath)}`
}
