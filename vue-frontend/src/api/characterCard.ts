/**
 * 角色卡工坊 API
 */

import { apiClient } from './client'
import type {
  CharacterCardCompatResponse,
  CharacterCandidatesResponse,
  CharacterCardCompileResponse,
  CharacterCardDraftPayload,
  CharacterCardDraftResponse,
} from '@/types/characterCard'

export interface GenerateCharacterCardDraftResponse extends CharacterCardDraftPayload {
  success: boolean
  message?: string
  error?: string
}

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

export async function getCharacterCardCandidates(bookId: string): Promise<CharacterCandidatesResponse> {
  return apiClient.get<CharacterCandidatesResponse>(`/api/manga-insight/${bookId}/character-cards/candidates`)
}

export async function generateCharacterCardDrafts(
  bookId: string,
  characterNames: string[],
  style: string = 'balanced'
): Promise<GenerateCharacterCardDraftResponse> {
  return apiClient.post<GenerateCharacterCardDraftResponse>(
    `/api/manga-insight/${bookId}/character-cards/generate`,
    {
      character_names: characterNames,
      style,
    },
    { timeout: 0 }
  )
}

export async function getCharacterCardDraft(bookId: string): Promise<CharacterCardDraftResponse> {
  return apiClient.get<CharacterCardDraftResponse>(`/api/manga-insight/${bookId}/character-cards/draft`)
}

export async function saveCharacterCardDraft(
  bookId: string,
  draft: CharacterCardDraftPayload
): Promise<{ success: boolean; message?: string; error?: string }> {
  return apiClient.put(`/api/manga-insight/${bookId}/character-cards/draft`, { draft })
}

export async function compileCharacterCards(
  bookId: string,
  payload?: { character_names?: string[]; draft?: CharacterCardDraftPayload }
): Promise<CharacterCardCompileResponse> {
  return apiClient.post<CharacterCardCompileResponse>(
    `/api/manga-insight/${bookId}/character-cards/compile`,
    payload || {}
  )
}

export async function getCharacterCardCompatibility(
  bookId: string,
  character: string
): Promise<CharacterCardCompatResponse> {
  return apiClient.get<CharacterCardCompatResponse>(
    `/api/manga-insight/${bookId}/character-cards/compat?character=${encodeURIComponent(character)}`
  )
}

export async function exportCharacterCardPng(bookId: string, character: string): Promise<Blob> {
  const response = await fetch(
    `/api/manga-insight/${bookId}/character-cards/export/png?character=${encodeURIComponent(character)}`
  )
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '导出 PNG 失败'))
  }
  return response.blob()
}

export async function exportCharacterCardsBatch(bookId: string, characterNames: string[]): Promise<Blob> {
  const response = await fetch(`/api/manga-insight/${bookId}/character-cards/export/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ character_names: characterNames }),
  })
  if (!response.ok) {
    throw new Error(await parseApiErrorMessage(response, '批量导出失败'))
  }
  return response.blob()
}
