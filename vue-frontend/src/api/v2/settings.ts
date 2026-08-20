import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { newIdempotencyKey } from './content'

export type V2CredentialEdit = components['schemas']['CredentialEdit']
export type V2CredentialSummary = components['schemas']['CredentialSummary']
export type V2DiagnosticRequest = components['schemas']['ProviderDiagnosticRequest']
export type V2ConnectionTestResult = components['schemas']['ConnectionTestResponse']
export type V2Font = components['schemas']['FontResource']
export type V2Prompt = components['schemas']['PromptResource']
export type V2PromptMutation = components['schemas']['PromptMutation']
export type V2ProviderSettingEntry = components['schemas']['ProviderSettingEntry']
export type V2ProviderSettingMutation = components['schemas']['ProviderSettingMutation']
export type V2SettingEntry = components['schemas']['SettingEntry']
export type V2SettingsDocument = components['schemas']['SettingsDocument']
export type V2SettingsTransaction = components['schemas']['SettingsTransaction']
export type V2SettingsTransactionResult = components['schemas']['SettingsTransactionResult']

export type V2WorkflowPreferences = components['schemas']['WorkflowPreferences']

type V2FontList = components['schemas']['FontList']
type V2ModelCatalogResponse = components['schemas']['ModelCatalogResponse']
type V2PromptList = components['schemas']['PromptList']

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

export async function listV2Fonts(): Promise<V2Font[]> {
  const result = await apiClient.get<V2FontList>('/api/v2/fonts')
  return result.items
}

export function uploadV2Font(file: File): Promise<V2Font> {
  const form = new FormData()
  form.append('file', file)
  form.append('displayName', file.name.replace(/\.[^.]+$/, ''))
  return apiClient.upload<V2Font>(
    '/api/v2/fonts',
    form,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function listV2Prompts(type?: V2Prompt['type']): Promise<V2Prompt[]> {
  const result = await apiClient.get<V2PromptList>(
    '/api/v2/prompts',
    { params: type ? { type } : {} },
  )
  return result.items
}

export function createV2Prompt(
  type: V2Prompt['type'],
  name: string,
  content: string,
): Promise<V2Prompt> {
  return apiClient.post<V2Prompt>(
    '/api/v2/prompts',
    { type, name, content },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2Prompt(prompt: V2Prompt): Promise<V2Prompt> {
  return apiClient.put<V2Prompt>(
    `/api/v2/prompts/${encodeURIComponent(prompt.id)}`,
    {
      name: prompt.name,
      content: prompt.content,
      baseRevision: prompt.revision,
    },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteV2Prompt(promptId: string): Promise<{ deleted: boolean }> {
  return apiClient.delete<{ deleted: boolean }>(
    `/api/v2/prompts/${encodeURIComponent(promptId)}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function resetV2Prompt(prompt: V2Prompt): Promise<V2Prompt> {
  return apiClient.post<V2Prompt>(
    `/api/v2/prompts/${encodeURIComponent(prompt.id)}/reset`,
    { baseRevision: prompt.revision },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function fetchV2ModelCatalog(
  request: V2DiagnosticRequest,
): Promise<V2ModelCatalogResponse> {
  return apiClient.post('/api/v2/model-catalog', request)
}

export function runV2ConnectionTest(
  kind: string,
  request: V2DiagnosticRequest = {},
): Promise<V2ConnectionTestResult> {
  return apiClient.post(
    `/api/v2/connection-tests/${encodeURIComponent(kind)}`,
    request,
  )
}

export function updateV2WorkflowPreferences(
  payload: V2WorkflowPreferences,
  baseRevision: number,
): Promise<V2SettingEntry> {
  return apiClient.patch<V2SettingEntry>(
    '/api/v2/settings/workflow-preferences',
    { payload, baseRevision },
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
