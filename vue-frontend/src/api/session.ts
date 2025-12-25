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

/**
 * 保存会话
 * @param name 会话名称
 * @param data 会话数据
 */
export async function saveSession(sessionName: string, data: SessionData): Promise<ApiResponse> {
  // 后端期望 session_name 和 session_data 字段
  return apiClient.post<ApiResponse>('/api/sessions/save', {
    session_name: sessionName,
    session_data: data,
  })
}

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

// ==================== 分批保存 API（与原版 api.js 参数格式一致） ====================

/**
 * 分批保存开始响应
 */
export interface BatchSaveStartResponseCompat {
  success: boolean
  session_folder?: string
  error?: string
}

/**
 * 分批保存 - 第一步：开始保存（原版兼容）
 * @param sessionPath - 会话路径（如 bookshelf/book_id/chapter_id）
 * @param metadata - 元数据（包含 ui_settings, images_meta, currentImageIndex）
 */
export async function batchSaveStartApi(
  sessionPath: string,
  metadata: {
    ui_settings: Record<string, unknown>
    images_meta: Array<Record<string, unknown>>
    currentImageIndex: number
  }
): Promise<BatchSaveStartResponseCompat> {
  return apiClient.post<BatchSaveStartResponseCompat>('/api/sessions/batch_save/start', {
    session_path: sessionPath,
    metadata: metadata,
  })
}

/**
 * 分批保存 - 第二步：保存单张图片（原版兼容）
 * @param sessionFolder - 会话文件夹路径（后端返回的绝对路径）
 * @param imageIndex - 图片索引
 * @param imageType - 图片类型 (original/translated/clean)
 * @param imageData - Base64 图片数据
 */
export async function batchSaveImageApi(
  sessionFolder: string,
  imageIndex: number,
  imageType: 'original' | 'translated' | 'clean',
  imageData: string
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/sessions/batch_save/image', {
    session_folder: sessionFolder,
    image_index: imageIndex,
    image_type: imageType,
    image_data: imageData,
  })
}

/**
 * 分批保存 - 第三步：完成保存（原版兼容）
 * @param sessionFolder - 会话文件夹路径
 * @param imagesMeta - 更新后的图片元数据
 */
export async function batchSaveCompleteApi(
  sessionFolder: string,
  imagesMeta: Array<Record<string, unknown>>
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/sessions/batch_save/complete', {
    session_folder: sessionFolder,
    images_meta: imagesMeta,
  })
}

// ==================== 导出别名（兼容旧接口） ====================

/**
 * 按路径加载会话 API（别名）
 * 用于 sessionStore 中的动态导入
 */
export const loadSessionByPathApi = loadSessionByPath
