import { apiClient } from './client'
import { downloadBlob } from './download'
import { newIdempotencyKey } from './v2/content'
import type { components } from './generated/v2'

export type PluginData = components['schemas']['Plugin']
type PluginV2 = PluginData
type PluginListV2 = components['schemas']['PluginList']
type PluginConfigV2 = components['schemas']['PluginConfig']
type PluginRefreshResultV2 = components['schemas']['PluginRefreshResult']

type PluginImportResultV2 = components['schemas']['PluginImportResult']

interface PluginImportConflict {
  currentRevision: number
  pluginId: string
}

export interface PluginRefreshResult {
  plugins: PluginData[]
  partialSuccess: boolean
  defaultStates: Record<string, boolean>
  summary: {
    checked: number
    failed: number
  }
}

const pluginRevisions = new Map<string, number>()
const configRevisions = new Map<string, number>()
const defaultStates = new Map<string, boolean>()
const importConflicts = new WeakMap<File, PluginImportConflict>()

function pluginEndpoint(pluginId: string, suffix = ''): string {
  return `/api/v2/plugins/${encodeURIComponent(pluginId)}${suffix}`
}

function rememberPlugin(plugin: PluginV2): PluginData {
  pluginRevisions.set(plugin.pluginId, plugin.currentRevision)
  configRevisions.set(plugin.pluginId, plugin.configRevision)
  defaultStates.set(plugin.pluginId, plugin.defaultEnabled)
  return plugin
}

function normalizeConfigSchema(
  schema: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(schema).map(([key, raw]) => {
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

export async function getPlugins(): Promise<PluginData[]> {
  const result = await apiClient.get<PluginListV2>('/api/v2/plugins')
  return result.items.map(rememberPlugin)
}

export async function refreshPlugins(): Promise<PluginRefreshResult> {
  const integrity = await apiClient.post<PluginRefreshResultV2>('/api/v2/plugins/refresh', {})
  const plugins = await getPlugins()
  return {
    plugins,
    partialSuccess: integrity.failedVersions > 0,
    defaultStates: Object.fromEntries(defaultStates),
    summary: {
      checked: integrity.checkedVersions,
      failed: integrity.failedVersions,
    },
  }
}

async function setRuntimeEnabled(
  pluginId: string,
  enabled: boolean,
): Promise<void> {
  const updated = await apiClient.put<PluginV2>(
    pluginEndpoint(pluginId, '/runtime-enabled'),
    { enabled },
  )
  rememberPlugin(updated)
}

export function enablePlugin(pluginId: string): Promise<void> {
  return setRuntimeEnabled(pluginId, true)
}

export function disablePlugin(pluginId: string): Promise<void> {
  return setRuntimeEnabled(pluginId, false)
}

export async function deletePlugin(pluginId: string): Promise<void> {
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
}

export async function exportPlugin(pluginId: string): Promise<{ blob: Blob; filename: string }> {
  return downloadBlob({
    url: pluginEndpoint(pluginId, '/export'),
    fallbackFilename: `${pluginId}.zip`,
    fallbackErrorMessage: '导出插件失败',
  })
}

function pluginImportConflict(error: unknown): PluginImportConflict | null {
  if (!error || typeof error !== 'object') return null
  const apiError = error as {
    details?: Record<string, unknown>
    status?: number
  }
  if (apiError.status !== 409) return null
  const pluginId = String(apiError.details?.pluginId || '').trim()
  const currentRevision = Number(apiError.details?.currentRevision)
  if (!pluginId || !Number.isInteger(currentRevision) || currentRevision < 1) {
    return null
  }
  return { pluginId, currentRevision }
}

export async function importPlugin(file: File, replace = false): Promise<void> {
  const conflict = importConflicts.get(file)
  if (replace && !conflict) {
    throw new Error('插件替换上下文已失效，请重新选择插件包')
  }
  const formData = new FormData()
  formData.append('file', file)
  formData.append('baseRevision', String(replace ? conflict!.currentRevision : 0))
  let imported: PluginImportResultV2
  try {
    imported = await apiClient.upload<PluginImportResultV2>(
      '/api/v2/plugins/import',
      formData,
      { headers: { 'Idempotency-Key': newIdempotencyKey() } },
    )
  } catch (error) {
    const nextConflict = pluginImportConflict(error)
    if (nextConflict) importConflicts.set(file, nextConflict)
    throw error
  }
  importConflicts.delete(file)
  pluginRevisions.set(imported.pluginId, imported.currentRevision)
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
): Promise<Record<string, unknown>> {
  const result = await getConfig(pluginId)
  return normalizeConfigSchema(result.schema)
}

export async function getPluginConfig(
  pluginId: string,
): Promise<Record<string, unknown>> {
  const result = await getConfig(pluginId)
  return result.value
}

export async function savePluginConfig(
  pluginId: string,
  config: Record<string, unknown>,
): Promise<void> {
  const baseRevision = configRevisions.get(pluginId)
  if (!baseRevision) throw new Error('插件配置已变化，请重新打开配置')
  const result = await apiClient.put<PluginConfigV2>(
    pluginEndpoint(pluginId, '/config'),
    { baseRevision, config },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  configRevisions.set(pluginId, result.configRevision)
}

export async function getPluginDefaultStates(): Promise<Record<string, boolean>> {
  if (defaultStates.size === 0) await getPlugins()
  return Object.fromEntries(defaultStates)
}

export async function setPluginDefaultState(
  pluginId: string,
  enabled: boolean,
): Promise<void> {
  const updated = await apiClient.put<PluginV2>(
    pluginEndpoint(pluginId, '/default-enabled'),
    { enabled },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  rememberPlugin(updated)
}
