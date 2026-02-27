import { apiClient } from './client'
import type { GlossaryEntry } from '@/types'

export interface GlossaryListResponse {
  success: boolean
  entries?: GlossaryEntry[]
  total?: number
  error?: string
}

export interface GlossaryEntryResponse {
  success: boolean
  entry?: GlossaryEntry
  error?: string
}

export interface GlossaryCreatePayload {
  source_text: string
  target_text: string
  note?: string
  entry_type?: 'term' | 'memory'
  scope?: 'global' | 'book'
  book_id?: string
  enabled?: boolean
}

export async function getGlossaryEntries(params?: {
  book_id?: string
  include_global?: boolean
  query?: string
  entry_type?: 'term' | 'memory'
}): Promise<GlossaryListResponse> {
  return apiClient.get<GlossaryListResponse>('/api/glossary', { params })
}

export async function createGlossaryEntry(payload: GlossaryCreatePayload): Promise<GlossaryEntryResponse> {
  return apiClient.post<GlossaryEntryResponse>('/api/glossary', payload)
}

export async function updateGlossaryEntry(
  id: string,
  payload: Partial<GlossaryCreatePayload>
): Promise<GlossaryEntryResponse> {
  return apiClient.put<GlossaryEntryResponse>(`/api/glossary/${id}`, payload)
}

export async function deleteGlossaryEntry(id: string): Promise<{ success: boolean; error?: string }> {
  return apiClient.delete<{ success: boolean; error?: string }>(`/api/glossary/${id}`)
}

