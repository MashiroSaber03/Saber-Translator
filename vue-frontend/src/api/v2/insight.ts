import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import { newIdempotencyKey } from './content'

const ROOT = '/api/v2/insight'

export type V2AcceptedJob = components['schemas']['JobBatchAccepted']
export type V2InsightAnalysisJobAccepted = components['schemas']['InsightAnalysisJobAccepted']
export type V2InsightArtifact = components['schemas']['InsightArtifact']
export type V2InsightBootstrap = components['schemas']['InsightBootstrap']
export type V2InsightNote = components['schemas']['InsightNote']
export type V2InsightPageDetail = components['schemas']['InsightPageDetail']
export type V2InsightPageSummary = components['schemas']['InsightPageSummary']
export type V2InsightQaStatus = components['schemas']['InsightQaStatus']
export type V2InsightOverviewTemplateList = components['schemas']['InsightOverviewTemplateList']
export type V2InsightRecentPageAnalysisList = components['schemas']['InsightRecentPageAnalysisList']

type V2InsightChapterList = components['schemas']['InsightChapterList']
type V2InsightNoteCreateCommand = components['schemas']['InsightNoteCreateCommand']
type V2InsightNoteList = components['schemas']['InsightNoteList']
type V2InsightNoteUpdateCommand = components['schemas']['InsightNoteUpdateCommand']
type V2InsightPageList = components['schemas']['InsightPageList']
type V2InsightTimeline = components['schemas']['InsightTimeline']

export function getInsightBootstrap(): Promise<V2InsightBootstrap> {
  return apiClient.get(`${ROOT}/bootstrap`)
}

export function listInsightChapters(bookId: string): Promise<V2InsightChapterList> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/chapters`)
}

export function listInsightPages(
  bookId: string,
  options: { chapterId?: string; cursor?: number; limit?: number } = {}
): Promise<V2InsightPageList> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/pages`, {
    params: {
      cursor: options.cursor ?? 0,
      limit: options.limit ?? 50,
      ...(options.chapterId ? { chapterId: options.chapterId } : {}),
    },
  })
}

export function listRecentInsightPageAnalyses(
  bookId: string,
  limit = 5,
): Promise<V2InsightRecentPageAnalysisList> {
  return apiClient.get(
    `${ROOT}/books/${encodeURIComponent(bookId)}/recent-page-analyses`,
    { params: { limit } },
  )
}

export function getInsightPage(
  pageId: string,
  options: { runId?: string } = {}
): Promise<V2InsightPageDetail> {
  const url = `${ROOT}/pages/${encodeURIComponent(pageId)}`
  return options.runId
    ? apiClient.get(url, { params: { runId: options.runId } })
    : apiClient.get(url)
}

export function createInsightAnalysisJob(command: {
  bookId: string
  chapterIds?: string[]
  pageIds?: string[]
  scope: 'chapter' | 'full' | 'incremental' | 'page'
}): Promise<V2InsightAnalysisJobAccepted> {
  assertBackendActionAllowed()
  return apiClient.post(`${ROOT}/analysis-jobs`, command, {
    headers: { 'Idempotency-Key': newIdempotencyKey() },
  })
}

export function getInsightOverview(bookId: string, template: string): Promise<V2InsightArtifact> {
  return apiClient.get(`${ROOT}/artifacts/overviews/${encodeURIComponent(template)}`, {
    params: { bookId },
  })
}

export function listInsightOverviewTemplates(bookId: string): Promise<V2InsightOverviewTemplateList> {
  return apiClient.get(`${ROOT}/artifacts/overviews`, {
    params: { bookId },
  })
}

export function rebuildInsightOverview(bookId: string, template: string): Promise<V2AcceptedJob> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/artifacts/overviews/${encodeURIComponent(template)}`,
    { bookId },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function getInsightTimeline(
  bookId: string,
  options: {
    eventCursor?: number
    eventLimit?: number
    characterCursor?: string
    characterLimit?: number
  } = {}
): Promise<V2InsightTimeline> {
  return apiClient.get(`${ROOT}/timeline`, {
    params: {
      bookId,
      eventCursor: options.eventCursor ?? 0,
      eventLimit: options.eventLimit ?? 50,
      characterLimit: options.characterLimit ?? 50,
      ...(options.characterCursor ? { characterCursor: options.characterCursor } : {}),
    },
  })
}

export function rebuildInsightTimeline(bookId: string): Promise<V2AcceptedJob> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/timeline`,
    { bookId },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function getInsightQaStatus(
  bookId: string,
  mode: 'exact' | 'global' = 'exact'
): Promise<V2InsightQaStatus> {
  return apiClient.get(`${ROOT}/qa/status`, { params: { bookId, mode } })
}

export function rebuildInsightCompressedContext(bookId: string): Promise<V2AcceptedJob> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/compressed-context/rebuild`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function rebuildInsightVectors(bookId: string): Promise<V2AcceptedJob> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/vector-rebuild`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function listInsightNotes(
  bookId: string,
  options: { cursor?: string; kind?: 'qa' | 'text'; limit?: number } = {}
): Promise<V2InsightNoteList> {
  return apiClient.get(`${ROOT}/notes`, {
    params: {
      bookId,
      limit: options.limit ?? 50,
      ...(options.cursor ? { cursor: options.cursor } : {}),
      ...(options.kind ? { kind: options.kind } : {}),
    },
  })
}

export function getInsightNote(noteId: string): Promise<V2InsightNote> {
  return apiClient.get(`${ROOT}/notes/${encodeURIComponent(noteId)}`)
}

export function createInsightNote(command: V2InsightNoteCreateCommand): Promise<V2InsightNote> {
  return apiClient.post(`${ROOT}/notes`, command, {
    headers: { 'Idempotency-Key': newIdempotencyKey() },
  })
}

export function updateInsightNote(
  noteId: string,
  command: V2InsightNoteUpdateCommand
): Promise<V2InsightNote> {
  return apiClient.patch(`${ROOT}/notes/${encodeURIComponent(noteId)}`, command, {
    headers: { 'Idempotency-Key': newIdempotencyKey() },
  })
}

export function deleteInsightNote(
  noteId: string,
  baseRevision: number
): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/notes/${encodeURIComponent(noteId)}?baseRevision=${baseRevision}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function createInsightExport(bookId: string): Promise<V2AcceptedJob> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/exports`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export function insightCurrentExportUrl(
  bookId: string,
  template: string,
  format = 'markdown'
): string {
  const query = new URLSearchParams({ template, format })
  return `${ROOT}/books/${encodeURIComponent(bookId)}/export/current?${query}`
}

export function insightPageExportUrl(pageId: string, format = 'markdown'): string {
  return `${ROOT}/pages/${encodeURIComponent(pageId)}/export?format=${encodeURIComponent(format)}`
}

export function insightQaUrl(bookId: string): string {
  return `${ROOT}/books/${encodeURIComponent(bookId)}/qa`
}
