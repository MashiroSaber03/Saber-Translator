/**
 * 插件 API
 * 包含插件列表、启用/禁用、配置管理等功能
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
 * 插件详情响应
 */
export interface PluginDetailResponse {
  success: boolean
  plugin?: PluginData
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

// ==================== 插件列表 API ====================

/**
 * 获取插件列表
 */
export async function getPlugins(): Promise<PluginListResponse> {
  return apiClient.get<PluginListResponse>('/api/plugins')
}

/**
 * 获取插件详情
 * @param name 插件名称
 */
export async function getPluginDetail(name: string): Promise<PluginDetailResponse> {
  return apiClient.get<PluginDetailResponse>(`/api/plugins/${name}`)
}

// ==================== 插件启用/禁用 API ====================

/**
 * 启用插件
 * @param name 插件名称
 */
export async function enablePlugin(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/plugins/${name}/enable`)
}

/**
 * 禁用插件
 * @param name 插件名称
 */
export async function disablePlugin(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/plugins/${name}/disable`)
}

/**
 * 删除插件
 * @param name 插件名称
 */
export async function deletePlugin(name: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/plugins/${name}`)
}

// ==================== 插件配置 API ====================

/**
 * 获取插件配置规范
 * @param name 插件名称
 */
export async function getPluginConfigSchema(name: string): Promise<PluginConfigSchemaResponse> {
  return apiClient.get<PluginConfigSchemaResponse>(`/api/plugins/${name}/config_schema`)
}

/**
 * 获取插件配置
 * @param name 插件名称
 */
export async function getPluginConfig(name: string): Promise<PluginConfigResponse> {
  return apiClient.get<PluginConfigResponse>(`/api/plugins/${name}/config`)
}

/**
 * 保存插件配置
 * @param name 插件名称
 * @param config 配置数据
 */
export async function savePluginConfig(
  name: string,
  config: Record<string, unknown>
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/plugins/${name}/config`, config)
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
 * @param name 插件名称
 * @param enabled 是否默认启用
 */
export async function setPluginDefaultState(name: string, enabled: boolean): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/plugins/${name}/set_default_state`, {
    enabled,
  })
}

// ==================== 别名导出（兼容旧调用方式） ====================

/** 获取默认状态（别名） */
export async function getDefaultStates(): Promise<{ states?: Record<string, boolean> }> {
  const result = await getPluginDefaultStates()
  return { states: result.default_states }
}

/** 设置默认状态（别名） */
export const setDefaultState = setPluginDefaultState

/** 获取配置规范（别名） */
export const getConfigSchema = getPluginConfigSchema

/** 获取配置（别名） */
export const getConfig = getPluginConfig

/** 保存配置（别名） */
export const saveConfig = savePluginConfig
