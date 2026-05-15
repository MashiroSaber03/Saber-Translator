import { apiClient } from './client'
import type {
  CharacterStudioDocumentResponse,
  CharacterStudioIndexResponse,
  ExportDiagnostic,
  PreviewSessionState,
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

export async function previewCharacterStudioChat(
  bookId: string,
  docId: string,
  message: string
): Promise<PreviewSessionState & { success: boolean; error?: string; message?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/preview/chat`, { message })
}

export async function resetCharacterStudioPreview(
  bookId: string,
  docId: string
): Promise<PreviewSessionState & { success: boolean; error?: string; message?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/preview/reset`, {})
}

export async function runCharacterStudioAgent(
  bookId: string,
  docId: string,
  message: string
): Promise<{ success: boolean; content?: string; context?: string; error?: string; message?: string }> {
  return apiClient.post(`/api/manga-insight/${bookId}/character-studio/documents/${docId}/agent`, { message })
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
