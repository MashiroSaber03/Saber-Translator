/**
 * 会话 API
 * 包含会话保存、加载、列表、删除、重命名等功能
 */

import { apiClient } from './client'
import type { ApiResponse, SessionListItem } from '@/types'
import type { SessionData } from '@/stores/sessionStore'

// ==================== 会话响应类型 ====================

/**
 * 会话列表响应
 */
export interface SessionListResponse {
  success: boolean
  sessions?: SessionListItem[]
  error?: string
}

/**
 * 会话加载响应
 */
export interface SessionLoadResponse {
  success: boolean
  session?: SessionData
  error?: string
}


// ==================== 基础会话 API ====================

/**
 * 获取会话列表
 */
export async function getSessionList(): Promise<SessionListResponse> {
  return apiClient.get<SessionListResponse>('/api/sessions/list')
}

// 注意：旧的 saveSession 函数已移除，保存功能请使用 pageStorage.ts

/**
 * 加载会话
 * @param name 会话名称
 */
export async function loadSession(name: string): Promise<SessionLoadResponse> {
  // 后端返回 session_data 字段，需要映射为 session
  const response = await apiClient.get<{ success: boolean; session_data?: SessionData; error?: string }>(
    '/api/sessions/load',
    { params: { name } }
  )
  return {
    success: response.success,
    session: response.session_data,
    error: response.error,
  }
}

/**
 * 按路径加载会话
 * @param path 会话文件路径
 */
export async function loadSessionByPath(path: string): Promise<SessionLoadResponse> {
  // 后端是 POST 方法，且返回 session_data 字段
  const response = await apiClient.post<{ success: boolean; session_data?: SessionData; error?: string }>(
    '/api/sessions/load_by_path',
    { path }
  )
  // 转换响应格式以匹配 SessionLoadResponse
  return {
    success: response.success,
    session: response.session_data,
    error: response.error,
  }
}

/**
 * 删除会话
 * @param name 会话名称
 */
export async function deleteSession(name: string): Promise<ApiResponse> {
  // 后端期望 session_name 字段
  return apiClient.post<ApiResponse>('/api/sessions/delete', { session_name: name })
}

/**
 * 重命名会话
 * @param oldName 原名称
 * @param newName 新名称
 */
export async function renameSession(oldName: string, newName: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/sessions/rename', {
    old_name: oldName,
    new_name: newName,
  })
}

// 注意：旧的分批保存 API (batchSaveStartApi, batchSaveImageApi, batchSaveCompleteApi) 已移除
// 新的单页保存 API 请使用 pageStorage.ts

// ==================== 导出别名（兼容旧接口） ====================

/**
 * 按路径加载会话 API（别名）
 * 用于 sessionStore 中的动态导入
 */
export const loadSessionByPathApi = loadSessionByPath
