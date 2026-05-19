/**
 * 插件 API
 * 包含插件列表、热重载刷新、启用/禁用、配置管理等功能
 * 
 * 所有涉及插件 ID 拼接 URL 的函数都使用 encodeURIComponent 编码，
 * 以防止插件标识包含特殊字符时请求失败。
 */

import { apiClient } from './client'
import type { ApiResponse, PluginData } from '@/types'

// ==================== 插件响应类型 ====================

/**
 * 插件列表响应
 */
export interface PluginListResponse {
  success: boolean
  plugins?: PluginData[]
  error?: string
}

/**
 * 插件配置规范响应
 */
export interface PluginConfigSchemaResponse {
  success: boolean
  schema?: Record<string, unknown>
  error?: string
}

/**
 * 插件配置响应
 */
export interface PluginConfigResponse {
  success: boolean
  config?: Record<string, unknown>
  error?: string
}

/**
 * 插件默认状态响应
 */
export interface PluginDefaultStatesResponse {
  success: boolean
  default_states?: Record<string, boolean>
  error?: string
}

export interface PluginRefreshSummary {
  added: number
  reloaded: number
  removed: number
  failed: number
}

export interface PluginRefreshFailure {
  plugin_name?: string
  plugin_id?: string
  source_path?: string
  error: string
}

export interface PluginRefreshResponse {
  success: boolean
  partial_success?: boolean
  plugins?: PluginData[]
  default_states?: Record<string, boolean>
  summary?: PluginRefreshSummary
  failures?: PluginRefreshFailure[]
  error?: string
}

export interface PluginImportResponse {
  success: boolean
  plugin?: PluginData
  message?: string
  error?: string
  details?: Record<string, unknown>
}

// ==================== 插件列表 API ====================

/**
 * 获取插件列表
 */
export async function getPlugins(): Promise<PluginListResponse> {
  return apiClient.get<PluginListResponse>('/api/plugins')
}

/**
 * 刷新插件列表并热重载插件目录
 */
export async function refreshPlugins(): Promise<PluginRefreshResponse> {
  return apiClient.post<PluginRefreshResponse>('/api/plugins/refresh')
}

// ==================== 插件启用/禁用 API ====================

/**
 * 启用插件
 * @param name 插件 ID
 */
export async function enablePlugin(name: string): Promise<ApiResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.post<ApiResponse>(`/api/plugins/${safeName}/enable`)
}

/**
 * 禁用插件
 * @param name 插件 ID
 */
export async function disablePlugin(name: string): Promise<ApiResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.post<ApiResponse>(`/api/plugins/${safeName}/disable`)
}

/**
 * 删除插件
 * @param name 插件 ID
 */
export async function deletePlugin(name: string): Promise<ApiResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.delete<ApiResponse>(`/api/plugins/${safeName}`)
}

/**
 * 导出插件压缩包
 * @param name 插件 ID
 */
export async function exportPlugin(name: string): Promise<{ blob: Blob; filename: string }> {
  const safeName = encodeURIComponent(name)
  const response = await fetch(`/api/plugins/${safeName}/export`)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || '导出插件失败')
  }
  const disposition = response.headers.get('content-disposition') || ''
  const match = disposition.match(/filename="?([^";]+)"?/)
  return {
    blob: await response.blob(),
    filename: match?.[1] || `${name}.zip`,
  }
}

/**
 * 导入插件压缩包
 * @param file 插件 zip
 * @param replace 是否替换同名插件
 */
export async function importPlugin(file: File, replace = false): Promise<PluginImportResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('replace', replace ? 'true' : 'false')
  return apiClient.upload<PluginImportResponse>('/api/plugins/import', formData)
}

// ==================== 插件配置 API ====================

/**
 * 获取插件配置规范
 * @param name 插件 ID
 */
export async function getPluginConfigSchema(name: string): Promise<PluginConfigSchemaResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.get<PluginConfigSchemaResponse>(`/api/plugins/${safeName}/config_schema`)
}

/**
 * 获取插件配置
 * @param name 插件 ID
 */
export async function getPluginConfig(name: string): Promise<PluginConfigResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.get<PluginConfigResponse>(`/api/plugins/${safeName}/config`)
}

/**
 * 保存插件配置
 * @param name 插件 ID
 * @param config 配置数据
 */
export async function savePluginConfig(
  name: string,
  config: Record<string, unknown>
): Promise<ApiResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.post<ApiResponse>(`/api/plugins/${safeName}/config`, config)
}

// ==================== 插件默认状态 API ====================

/**
 * 获取所有插件的默认启用状态
 */
export async function getPluginDefaultStates(): Promise<PluginDefaultStatesResponse> {
  return apiClient.get<PluginDefaultStatesResponse>('/api/plugins/default_states')
}

/**
 * 设置插件的默认启用状态
 * @param name 插件 ID
 * @param enabled 是否默认启用
 */
export async function setPluginDefaultState(name: string, enabled: boolean): Promise<ApiResponse> {
  const safeName = encodeURIComponent(name)
  return apiClient.post<ApiResponse>(`/api/plugins/${safeName}/set_default_state`, {
    enabled,
  })
}
