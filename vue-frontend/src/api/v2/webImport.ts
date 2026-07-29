import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'
import { runV2ConnectionTest } from './settings'

const ROOT = '/api/v2/web-import'

export interface WebImportSupport {
  galleryDlAvailable: boolean
  galleryDlSupported: boolean
  recommendedEngine: string
  sourceUrl: string
}

export interface WebImportDraftAccepted {
  batchId: string
  draftId: string
  jobIds: string[]
  status: 'queued'
}

export interface WebImportDraft {
  actualEngine: string | null
  candidateCount: number
  chapterId: string
  expiresAt: string
  failedCount: number
  id: string
  jobs: Array<{ id: string; kind: string; status: string }>
  requestedEngine: string
  revision: number
  selectedCount: number
  sourceUrl: string
  status: 'committing' | 'completed' | 'extracting' | 'failed' | 'ready'
}

export interface WebImportDraftPage {
  checksum: string | null
  error: Record<string, unknown> | null
  id: string
  ordinal: number
  selected: boolean
  sourceMediaUrl: string | null
  sourceUrl: string
  thumbnailUrl: string | null
}

export interface WebImportDraftPageList {
  items: WebImportDraftPage[]
  nextCursor: number | null
}

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
): Promise<{ draftId: string; revision: number; selectedPageIds: string[] }> {
  return apiClient.put(
    `${ROOT}/drafts/${encodeURIComponent(draftId)}/selection`,
    { baseRevision, selectedPageIds },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function commitWebImportDraft(
  draftId: string,
  baseRevision: number,
): Promise<{ batchId: string; jobIds: string[]; status: 'queued' }> {
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
