import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'

export interface V2TranslationBatchAccepted {
  batchId: string
  jobIds: string[]
  status: 'queued'
}

export interface V2TextImportPreviewPage {
  baseDocumentRevision: number | null
  changes: Array<{
    bubbleId: string
    differences: Record<string, { after: unknown; before: unknown }>
    fields: Record<string, string>
  }>
  issues: string[]
  pageId: string
  sourceAssetId: string | null
  sourceChecksum: string | null
  status: 'conflict' | 'match'
}

export interface V2TextImportPreview {
  chapterId: string
  conflictedPages: number
  matchedPages: number
  pages: V2TextImportPreviewPage[]
  schemaVersion: number
}

export interface TranslationJobConfig {
  executionMode: 'parallel' | 'sequential'
  mode: 'hq' | 'proofread' | 'remove_text' | 'standard'
  reuseExistingBubbles?: boolean
  skipCompleted?: boolean
}

export async function createChapterTranslationJob(
  chapterId: string,
  pageIds: string[],
  config: TranslationJobConfig,
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/translation-jobs`,
    { config, pageIds },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function createChapterDetectJob(
  chapterId: string,
  pageIds?: string[],
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/detect-jobs`,
    pageIds ? { pageIds } : {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function createChapterStyleApplyJob(
  chapterId: string,
  command: {
    selectedFields: string[]
    sourceDocumentRevision: number
    sourcePageId: string
  },
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/style-apply-jobs`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createChapterExportJob(
  chapterId: string,
  format: 'cbz' | 'pdf' | 'zip',
  pageIds?: string[],
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/export-jobs`,
    { format, ...(pageIds ? { pageIds } : {}) },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function getChapterTextExportUrl(chapterId: string): string {
  return `/api/v2/chapters/${encodeURIComponent(chapterId)}/text-export`
}

export function previewChapterTextImport(
  chapterId: string,
  file: File,
): Promise<V2TextImportPreview> {
  const form = new FormData()
  form.append('file', file, file.name)
  return apiClient.upload(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/text-import/preview`,
    form,
  )
}

export function commitChapterTextImport(
  chapterId: string,
  confirmedPages: V2TextImportPreviewPage[],
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/text-import/commit`,
    { confirmedPages },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
