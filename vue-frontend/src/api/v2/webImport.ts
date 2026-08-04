import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import { newIdempotencyKey } from './content'
import { runV2ConnectionTest, type V2ConnectionTestResult } from './settings'

const ROOT = '/api/v2/web-import'

export type WebImportDraft = components['schemas']['WebImportDraft']
export type WebImportDraftAccepted = components['schemas']['WebImportDraftAccepted']
export type WebImportDraftPage = components['schemas']['WebImportDraftPage']
export type WebImportDraftPageList = components['schemas']['WebImportDraftPageList']
export type WebImportSelection = components['schemas']['WebImportSelection']
export type WebImportSupport = components['schemas']['WebImportSupport']

export function checkWebImportSupport(sourceUrl: string): Promise<WebImportSupport> {
  return apiClient.post(`${ROOT}/support-checks`, { sourceUrl })
}

export function createWebImportDraft(command: {
  chapterId: string
  engine: string
  sourceUrl: string
}): Promise<WebImportDraftAccepted> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/drafts`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getWebImportDraft(draftId: string): Promise<WebImportDraft> {
  return apiClient.get(`${ROOT}/drafts/${encodeURIComponent(draftId)}`)
}

export function listWebImportDraftPages(
  draftId: string,
  options: { cursor?: number; limit?: number } = {},
): Promise<WebImportDraftPageList> {
  const query = new URLSearchParams({
    cursor: String(options.cursor ?? 0),
    limit: String(options.limit ?? 100),
  })
  return apiClient.get(
    `${ROOT}/drafts/${encodeURIComponent(draftId)}/pages?${query.toString()}`,
  )
}

export function updateWebImportSelection(
  draftId: string,
  baseRevision: number,
  selectedPageIds: string[],
): Promise<WebImportSelection> {
  return apiClient.put(
    `${ROOT}/drafts/${encodeURIComponent(draftId)}/selection`,
    { baseRevision, selectedPageIds },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function commitWebImportDraft(
  draftId: string,
  baseRevision: number,
): Promise<WebImportDraftAccepted> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/drafts/${encodeURIComponent(draftId)}/commit`,
    { baseRevision },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function testFirecrawlConnection(
  apiKey: string,
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest(
    'firecrawl',
    apiKey
      ? { secret: { api_key: apiKey } }
      : { domain: 'web_import_firecrawl' },
  )
}

export function testAgentConnection(
  provider: string,
  apiKey: string,
  customBaseUrl: string,
  modelName: string,
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('web_import_agent', {
    provider,
    baseUrl: customBaseUrl || undefined,
    model: modelName,
    ...(apiKey
      ? { secret: { api_key: apiKey } }
      : { domain: 'web_import_agent' }),
  })
}
