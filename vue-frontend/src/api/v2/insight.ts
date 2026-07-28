import { apiClient } from '@/api/client'
import { jobsApi } from './jobs'
import { newIdempotencyKey } from './content'

const ROOT = '/api/v2/insight'

export interface V2InsightBootstrap {
  activeJobs: Array<{
    bookId: string | null
    jobId: string
    progress: Record<string, unknown>
    status: string
  }>
  books: Array<{
    activeRun: { publishedAt: string | null; runId: string; status: string } | null
    analyzedPageCount: number
    bookId: string
    coverUrl: string | null
    pageCount: number
    title: string
  }>
  qa: { available: boolean; reason: string }
}

export interface V2InsightChapter {
  analysisCounts: Record<string, number>
  chapterId: string
  ordinal: number
  pageCount: number
  title: string
}

export interface V2InsightPageSummary {
  activeAnalysisId: string | null
  analysisState: 'failed' | 'not_analyzed' | 'ready' | 'running' | 'stale'
  chapterId: string
  displayPageNumber: number
  pageId: string
  sourceAssetId: string
  thumbnailUrl: string | null
}

export interface V2InsightPageDetail {
  analysis: Record<string, unknown> | null
  analysisState: string
  bookId: string
  chapterId: string
  chapterTitle: string
  displayPageNumber: number
  generatedAt: string | null
  pageId: string
  preview: boolean
  runId: string | null
  sourceAssetId: string
  sourceUrl: string
  staleReasons: string[]
}

export interface V2InsightArtifact {
  artifactId: string
  bookId: string
  kind: string
  payload: Record<string, unknown>
  revision: number
  runId: string | null
  status: string
  template: string
}

export interface V2InsightNote {
  bookId: string
  citations: Array<{
    excerpt: string
    pageId: string | null
    pageIdSnapshot: string
    pageNumberSnapshot: number
    score: number | null
    sourceAnalysisId: string | null
  }>
  commentCount: number
  comments?: Array<Record<string, unknown> | string>
  content: string | null
  createdAt: string
  excerpt: string | null
  kind: 'qa' | 'text'
  noteId: string
  revision: number
  tags: string[]
  title: string
  updatedAt: string
}

export interface V2AcceptedJob {
  batchId: string
  jobIds: string[]
  runId?: string
  status: 'queued'
}

export function getInsightBootstrap(): Promise<V2InsightBootstrap> {
  return apiClient.get(`${ROOT}/bootstrap`)
}

export function listInsightChapters(bookId: string): Promise<{ items: V2InsightChapter[] }> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/chapters`)
}

export async function listAllInsightPages(bookId: string): Promise<V2InsightPageSummary[]> {
  const items: V2InsightPageSummary[] = []
  let cursor = 0
  do {
    const response = await apiClient.get<{
      items: V2InsightPageSummary[]
      nextCursor: number | null
    }>(`${ROOT}/books/${encodeURIComponent(bookId)}/pages`, {
      params: { cursor, limit: 100 },
    })
    items.push(...response.items)
    cursor = response.nextCursor ?? 0
  } while (cursor > 0)
  return items
}

export function getInsightPage(pageId: string): Promise<V2InsightPageDetail> {
  return apiClient.get(`${ROOT}/pages/${encodeURIComponent(pageId)}`)
}

export function createInsightAnalysisJob(command: {
  bookId: string
  chapterIds?: string[]
  force?: boolean
  pageIds?: string[]
  scope: 'chapter' | 'full' | 'incremental' | 'page'
}): Promise<V2AcceptedJob> {
  return apiClient.post(
    `${ROOT}/analysis-jobs`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getInsightOverview(
  bookId: string,
  template: string,
): Promise<V2InsightArtifact> {
  return apiClient.get(`${ROOT}/artifacts/overviews/${encodeURIComponent(template)}`, {
    params: { bookId },
  })
}

export function rebuildInsightOverview(
  bookId: string,
  template: string,
): Promise<V2AcceptedJob> {
  return apiClient.post(
    `${ROOT}/artifacts/overviews/${encodeURIComponent(template)}`,
    { bookId },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getInsightTimeline(bookId: string): Promise<Record<string, unknown>> {
  return apiClient.get(`${ROOT}/timeline`, { params: { bookId } })
}

export function rebuildInsightTimeline(bookId: string): Promise<V2AcceptedJob> {
  return apiClient.post(
    `${ROOT}/timeline`,
    { bookId },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getInsightQaStatus(bookId: string): Promise<Record<string, unknown>> {
  return apiClient.get(`${ROOT}/qa/status`, { params: { bookId } })
}

export function rebuildInsightVectors(bookId: string): Promise<V2AcceptedJob> {
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/vector-rebuild`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getInsightJob(jobId: string) {
  return jobsApi.get(jobId)
}

export async function listAllInsightNotes(
  bookId: string,
  kind?: 'qa' | 'text',
): Promise<V2InsightNote[]> {
  const items: V2InsightNote[] = []
  let cursor: string | null = null
  do {
    const response = await apiClient.get<{
      items: V2InsightNote[]
      nextCursor: string | null
    }>(`${ROOT}/notes`, {
      params: {
        bookId,
        limit: 200,
        detail: 1,
        ...(cursor ? { cursor } : {}),
        ...(kind ? { kind } : {}),
      },
    })
    items.push(...response.items)
    cursor = response.nextCursor
  } while (cursor)
  return items
}

export function createInsightNote(command: {
  bookId: string
  citations: Array<Record<string, unknown> | string>
  comments: Array<Record<string, unknown> | string>
  content: string
  kind: 'qa' | 'text'
  tags: string[]
  title: string
}): Promise<V2InsightNote> {
  return apiClient.post(
    `${ROOT}/notes`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateInsightNote(
  noteId: string,
  command: {
    baseRevision: number
    citations: Array<Record<string, unknown> | string>
    comments: Array<Record<string, unknown> | string>
    content: string
    kind: 'qa' | 'text'
    tags: string[]
    title: string
  },
): Promise<V2InsightNote> {
  return apiClient.patch(
    `${ROOT}/notes/${encodeURIComponent(noteId)}`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteInsightNote(noteId: string, baseRevision: number): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/notes/${encodeURIComponent(noteId)}?baseRevision=${baseRevision}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createInsightExport(bookId: string): Promise<V2AcceptedJob> {
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/exports`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function insightCurrentExportUrl(bookId: string, template: string, format = 'markdown'): string {
  const query = new URLSearchParams({ template, format })
  return `${ROOT}/books/${encodeURIComponent(bookId)}/export/current?${query}`
}

export function insightPageExportUrl(pageId: string, format = 'markdown'): string {
  return `${ROOT}/pages/${encodeURIComponent(pageId)}/export?format=${encodeURIComponent(format)}`
}

export function insightQaUrl(bookId: string): string {
  return `${ROOT}/books/${encodeURIComponent(bookId)}/qa`
}
