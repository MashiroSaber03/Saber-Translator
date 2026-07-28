import JSZip from 'jszip'

import { apiClient } from './client'
import { downloadBlob } from './download'
import { newIdempotencyKey } from './v2/content'
import type { ApiResponse, PluginData } from '@/types'

interface PluginV2 {
  pluginId: string
  displayName: string
  author: string
  description: string
  state: 'enabled' | 'disabled' | 'error'
  defaultEnabled: boolean
  runtimeEnabled: boolean
  config: Record<string, unknown>
  configRevision: number
  errorMessage?: string | null
  pluginVersionId: string
  packageVersion: string
  currentRevision: number
  manifest: {
    supported_steps?: string[]
    supported_modes?: string[]
    priority?: number
    failure_policy?: string
  }
  configSchema: Record<string, unknown>
}

interface PluginListV2 {
  items: PluginV2[]
}

interface PluginConfigV2 {
  pluginId: string
  schema: Record<string, unknown>
  value: Record<string, unknown>
  configRevision: number
}

export interface PluginListResponse {
  success: boolean
  plugins: PluginData[]
}

export interface PluginConfigSchemaResponse {
  success: boolean
  schema: Record<string, unknown>
}

export interface PluginConfigResponse {
  success: boolean
  config: Record<string, unknown>
  configRevision: number
}

export interface PluginDefaultStatesResponse {
  success: boolean
  default_states: Record<string, boolean>
}

export interface PluginRefreshResponse extends PluginListResponse {
  partial_success: boolean
  default_states: Record<string, boolean>
  summary: {
    checked: number
    failed: number
  }
  failures?: Array<{ error: string }>
}

export interface PluginImportResponse {
  success: boolean
  plugin?: PluginData
}

const pluginRevisions = new Map<string, number>()
const configRevisions = new Map<string, number>()
const defaultStates = new Map<string, boolean>()

function pluginEndpoint(pluginId: string, suffix = ''): string {
  return `/api/v2/plugins/${encodeURIComponent(pluginId)}${suffix}`
}

function toPluginData(plugin: PluginV2): PluginData {
  pluginRevisions.set(plugin.pluginId, plugin.currentRevision)
  configRevisions.set(plugin.pluginId, plugin.configRevision)
  defaultStates.set(plugin.pluginId, plugin.defaultEnabled)
  return {
    id: plugin.pluginId,
    display_name: plugin.displayName,
    description: plugin.description,
    version: plugin.packageVersion,
    author: plugin.author,
    enabled: plugin.runtimeEnabled,
    default_enabled: plugin.defaultEnabled,
    has_config: Object.keys(plugin.configSchema || {}).length > 0,
    supported_steps: plugin.manifest.supported_steps || [],
    supported_modes: plugin.manifest.supported_modes || [],
    priority: plugin.manifest.priority,
    failure_policy: plugin.manifest.failure_policy,
    configSchema: normalizeConfigSchema(plugin.configSchema),
    config: plugin.config,
    current_revision: plugin.currentRevision,
    config_revision: plugin.configRevision,
    state: plugin.state,
    error_message: plugin.errorMessage || null,
  }
}

function normalizeConfigSchema(
  schema: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(schema || {}).map(([key, raw]) => {
      if (!raw || typeof raw !== 'object') return [key, raw]
      const field = { ...(raw as Record<string, unknown>) }
      if (field.minimum !== undefined && field.min === undefined) {
        field.min = field.minimum
      }
      if (field.maximum !== undefined && field.max === undefined) {
        field.max = field.maximum
      }
      if (Array.isArray(field.options)) {
        field.options = field.options.map(option => (
          option && typeof option === 'object'
            ? option
            : { value: String(option), label: String(option) }
        ))
      }
      return [key, field]
    }),
  )
}

export async function getPlugins(): Promise<PluginListResponse> {
  const result = await apiClient.get<PluginListV2>('/api/v2/plugins')
  return {
    success: true,
    plugins: result.items.map(toPluginData),
  }
}

export async function refreshPlugins(): Promise<PluginRefreshResponse> {
  const integrity = await apiClient.post<{
    checkedVersions: number
    failedVersions: number
  }>('/api/v2/plugins/refresh', {})
  const current = await getPlugins()
  return {
    ...current,
    partial_success: integrity.failedVersions > 0,
    default_states: Object.fromEntries(defaultStates),
    summary: {
      checked: integrity.checkedVersions,
      failed: integrity.failedVersions,
    },
  }
}

async function setRuntimeEnabled(
  pluginId: string,
  enabled: boolean,
): Promise<ApiResponse> {
  const updated = await apiClient.put<PluginV2>(
    pluginEndpoint(pluginId, '/runtime-enabled'),
    { enabled },
  )
  toPluginData(updated)
  return { success: true }
}

export function enablePlugin(pluginId: string): Promise<ApiResponse> {
  return setRuntimeEnabled(pluginId, true)
}

export function disablePlugin(pluginId: string): Promise<ApiResponse> {
  return setRuntimeEnabled(pluginId, false)
}

export async function deletePlugin(pluginId: string): Promise<ApiResponse> {
  const revision = pluginRevisions.get(pluginId)
  if (!revision) throw new Error('插件版本已变化，请刷新后重试')
  await apiClient.delete(pluginEndpoint(pluginId), {
    headers: {
      'If-Match': String(revision),
      'Idempotency-Key': newIdempotencyKey(),
    },
  })
  pluginRevisions.delete(pluginId)
  configRevisions.delete(pluginId)
  defaultStates.delete(pluginId)
  return { success: true }
}

export async function exportPlugin(pluginId: string): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: pluginEndpoint(pluginId, '/export'),
    fallbackFilename: `${pluginId}.zip`,
    fallbackErrorMessage: '导出插件失败',
  })
}

async function readPluginId(file: File): Promise<string> {
  const archive = await JSZip.loadAsync(file)
  const manifestEntry = archive.file('plugin.json')
  if (!manifestEntry) throw new Error('插件包缺少 plugin.json')
  const manifest = JSON.parse(await manifestEntry.async('text')) as {
    plugin_id?: unknown
  }
  const pluginId = String(manifest.plugin_id || '').trim()
  if (!pluginId) throw new Error('plugin.json 缺少 plugin_id')
  return pluginId
}

export async function importPlugin(file: File, replace = false): Promise<PluginImportResponse> {
  const pluginId = await readPluginId(file)
  const baseRevision = replace ? (pluginRevisions.get(pluginId) || 0) : 0
  const formData = new FormData()
  formData.append('file', file)
  formData.append('baseRevision', String(baseRevision))
  await apiClient.upload('/api/v2/plugins/import', formData, {
    headers: { 'Idempotency-Key': newIdempotencyKey() },
  })
  const current = await getPlugins()
  return {
    success: true,
    plugin: current.plugins.find(plugin => plugin.id === pluginId),
  }
}

async function getConfig(pluginId: string): Promise<PluginConfigV2> {
  const result = await apiClient.get<PluginConfigV2>(
    pluginEndpoint(pluginId, '/config'),
  )
  configRevisions.set(pluginId, result.configRevision)
  return result
}

export async function getPluginConfigSchema(
  pluginId: string,
): Promise<PluginConfigSchemaResponse> {
  const result = await getConfig(pluginId)
  return {
    success: true,
    schema: normalizeConfigSchema(result.schema),
  }
}

export async function getPluginConfig(
  pluginId: string,
): Promise<PluginConfigResponse> {
  const result = await getConfig(pluginId)
  return {
    success: true,
    config: result.value,
    configRevision: result.configRevision,
  }
}

export async function savePluginConfig(
  pluginId: string,
  config: Record<string, unknown>,
): Promise<ApiResponse> {
  const baseRevision = configRevisions.get(pluginId)
  if (!baseRevision) throw new Error('插件配置已变化，请重新打开配置')
  const result = await apiClient.put<PluginConfigV2>(
    pluginEndpoint(pluginId, '/config'),
    { baseRevision, config },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  configRevisions.set(pluginId, result.configRevision)
  return { success: true }
}

export async function getPluginDefaultStates(): Promise<PluginDefaultStatesResponse> {
  if (defaultStates.size === 0) await getPlugins()
  return {
    success: true,
    default_states: Object.fromEntries(defaultStates),
  }
}

export async function setPluginDefaultState(
  pluginId: string,
  enabled: boolean,
): Promise<ApiResponse> {
  const updated = await apiClient.put<PluginV2>(
    pluginEndpoint(pluginId, '/default-enabled'),
    { enabled },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  toPluginData(updated)
  return { success: true }
}
