import { apiClient } from './client'
import { downloadBlob } from './download'
import type { ApiResponse, PluginData } from '@/types'

export interface PluginListResponse {
  success: boolean
  plugins?: PluginData[]
  error?: string
}

export interface PluginConfigSchemaResponse {
  success: boolean
  schema?: Record<string, unknown>
  error?: string
}

export interface PluginConfigResponse {
  success: boolean
  config?: Record<string, unknown>
  error?: string
}

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

function pluginEndpoint(name: string, suffix: string): string {
  return `/api/plugins/${encodeURIComponent(name)}${suffix}`
}

export async function getPlugins(): Promise<PluginListResponse> {
  return apiClient.get<PluginListResponse>('/api/plugins')
}

export async function refreshPlugins(): Promise<PluginRefreshResponse> {
  return apiClient.post<PluginRefreshResponse>('/api/plugins/refresh')
}

export async function enablePlugin(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(pluginEndpoint(name, '/enable'))
}

export async function disablePlugin(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(pluginEndpoint(name, '/disable'))
}

export async function deletePlugin(name: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(pluginEndpoint(name, ''))
}

export async function exportPlugin(name: string): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: pluginEndpoint(name, '/export'),
    fallbackFilename: `${name}.zip`,
    fallbackErrorMessage: '导出插件失败',
  })
}

export async function importPlugin(file: File, replace = false): Promise<PluginImportResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('replace', replace ? 'true' : 'false')
  return apiClient.upload<PluginImportResponse>('/api/plugins/import', formData)
}

export async function getPluginConfigSchema(name: string): Promise<PluginConfigSchemaResponse> {
  return apiClient.get<PluginConfigSchemaResponse>(pluginEndpoint(name, '/config_schema'))
}

export async function getPluginConfig(name: string): Promise<PluginConfigResponse> {
  return apiClient.get<PluginConfigResponse>(pluginEndpoint(name, '/config'))
}

export async function savePluginConfig(
  name: string,
  config: Record<string, unknown>
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(pluginEndpoint(name, '/config'), config)
}

export async function getPluginDefaultStates(): Promise<PluginDefaultStatesResponse> {
  return apiClient.get<PluginDefaultStatesResponse>('/api/plugins/default_states')
}

export async function setPluginDefaultState(name: string, enabled: boolean): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(pluginEndpoint(name, '/set_default_state'), {
    enabled,
  })
}
