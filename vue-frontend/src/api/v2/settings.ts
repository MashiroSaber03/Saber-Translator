import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'

export interface V2SettingEntry {
  domain: string
  payload: Record<string, unknown>
  revision: number
  schemaVersion: number
}

export interface V2BookSettingEntry extends V2SettingEntry {
  bookId: string
}

export interface V2ProviderSettingEntry extends V2SettingEntry {
  credentialVersionId: string | null
  provider: string
}

export interface V2CredentialSummary {
  credentialId: string
  credentialVersionId: string
  currentVersion: number
  domain: string
  hasKey: boolean
  provider: string
  revision: number
}

export interface V2SettingsDocument {
  settings: V2SettingEntry[]
  bookSettings: V2BookSettingEntry[]
  providerSettings: V2ProviderSettingEntry[]
  credentials: V2CredentialSummary[]
}

export interface V2SettingMutation {
  baseRevision: number
  domain: string
  payload: Record<string, unknown>
  schemaVersion?: number
}

export interface V2ProviderSettingMutation extends V2SettingMutation {
  credentialEditRef?: string
  credentialVersionId?: string
  provider: string
}

export interface V2CredentialEdit {
  baseRevision: number
  clientRef: string
  credentialId?: string
  domain: string
  provider: string
  secret: Record<string, unknown>
}

export interface V2SettingsTransaction {
  settings?: V2SettingMutation[]
  bookSettings?: Array<V2SettingMutation & { bookId: string }>
  providerSettings?: V2ProviderSettingMutation[]
  credentialEdits?: V2CredentialEdit[]
}

interface V2MutationResult {
  domain: string
  revision: number
}

export interface V2SettingsTransactionResult {
  settings: V2MutationResult[]
  bookSettings: Array<V2MutationResult & { bookId: string }>
  providerSettings: Array<V2MutationResult & { provider: string }>
  credentials: V2CredentialSummary[]
}

export interface V2Font {
  assetUrl: string | null
  builtinKey: string | null
  displayName: string
  id: string
  kind: 'builtin' | 'uploaded'
}

export function getV2Settings(
  domains: string[] = [],
  bookId?: string,
): Promise<V2SettingsDocument> {
  const params: Record<string, string> = {}
  if (domains.length > 0) params.domains = domains.join(',')
  if (bookId) params.book_id = bookId
  return apiClient.get<V2SettingsDocument>('/api/v2/settings', { params })
}

export function saveV2SettingsTransaction(
  payload: V2SettingsTransaction,
): Promise<V2SettingsTransactionResult> {
  return apiClient.put<V2SettingsTransactionResult>(
    '/api/v2/settings/transactions',
    payload,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteV2Credential(credentialId: string): Promise<{ deleted: boolean }> {
  return apiClient.delete<{ deleted: boolean }>(
    `/api/v2/credentials/${encodeURIComponent(credentialId)}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function listV2Fonts(): Promise<V2Font[]> {
  const result = await apiClient.get<{ items: V2Font[] }>('/api/v2/fonts')
  return result.items
}

export function uploadV2Font(file: File): Promise<{ assetUrl: string; id: string }> {
  const form = new FormData()
  form.append('file', file)
  form.append('displayName', file.name.replace(/\.[^.]+$/, ''))
  return apiClient.upload<{ assetUrl: string; id: string }>(
    '/api/v2/fonts',
    form,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function cleanV2DebugFiles(): Promise<{ removed: number }> {
  return apiClient.post<{ removed: number }>(
    '/api/v2/maintenance/clean-debug',
    undefined,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function cleanV2TempFiles(): Promise<{ recovered: number }> {
  return apiClient.post<{ recovered: number }>(
    '/api/v2/maintenance/clean-temp',
    undefined,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
