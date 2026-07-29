import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { newIdempotencyKey } from './content'
import { runV2ConnectionTest } from './settings'

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
  config: Record<string, unknown>
  engine: string
  sourceUrl: string
}): Promise<WebImportDraftAccepted> {
  return apiClient.post(
    `${ROOT}/drafts`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getWebImportDraft(draftId: string): Promise<WebImportDraft> {
  return apiClient.get(`${ROOT}/drafts/${encodeURIComponent(draftId)}`)
}

export async function listAllWebImportDraftPages(
  draftId: string,
): Promise<WebImportDraftPage[]> {
  const pages: WebImportDraftPage[] = []
  let cursor = 0
  do {
    const result = await apiClient.get<WebImportDraftPageList>(
      `${ROOT}/drafts/${encodeURIComponent(draftId)}/pages?cursor=${cursor}&limit=200`,
    )
    pages.push(...result.items)
    cursor = result.nextCursor ?? 0
  } while (cursor > 0)
  return pages
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
  return apiClient.post(
    `${ROOT}/drafts/${encodeURIComponent(draftId)}/commit`,
    { baseRevision },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function testFirecrawlConnection(
  apiKey: string,
): Promise<{ success: boolean; message?: string }> {
  return runV2ConnectionTest(
    'firecrawl',
    apiKey
      ? { secret: { apiKey } }
      : { domain: 'web_import_firecrawl' },
  )
}

export function testAgentConnection(
  provider: string,
  apiKey: string,
  customBaseUrl: string,
  modelName: string,
): Promise<{ success: boolean; message?: string }> {
  return runV2ConnectionTest('web_import_agent', {
    provider,
    baseUrl: customBaseUrl || undefined,
    model: modelName,
    ...(apiKey
      ? { secret: { apiKey } }
      : { domain: 'web_import_agent' }),
  })
}
