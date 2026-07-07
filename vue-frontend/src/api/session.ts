import { apiClient } from './client'
import type { ApiResponse, SessionListItem } from '@/types'
import type { SessionData } from '@/stores/sessionStore'

export interface SessionListResponse {
  success: boolean
  sessions?: SessionListItem[]
  error?: string
}

export interface SessionLoadResponse {
  success: boolean
  session?: SessionData
  error?: string
}

interface SessionWireResponse {
  success: boolean
  session_data?: SessionData
  error?: string
}

function toSessionLoadResponse(response: SessionWireResponse): SessionLoadResponse {
  return {
    success: response.success,
    session: response.session_data,
    error: response.error,
  }
}

export async function getSessionList(): Promise<SessionListResponse> {
  return apiClient.get<SessionListResponse>('/api/sessions/list')
}

export async function loadSession(name: string): Promise<SessionLoadResponse> {
  const response = await apiClient.get<SessionWireResponse>('/api/sessions/load', {
    params: { name },
  })
  return toSessionLoadResponse(response)
}

export async function loadSessionByPath(path: string): Promise<SessionLoadResponse> {
  const response = await apiClient.post<SessionWireResponse>('/api/sessions/load_by_path', {
    path,
  })
  return toSessionLoadResponse(response)
}

export async function deleteSession(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/sessions/delete', { session_name: name })
}

export async function renameSession(currentName: string, newName: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/sessions/rename', {
    old_name: currentName,
    new_name: newName,
  })
}
