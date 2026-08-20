import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { newIdempotencyKey } from './content'

type TranslationJobConfig = components['schemas']['TranslationJobConfig']
type V2TextImportPreview = components['schemas']['TextImportPreview']
type V2TextImportPreviewPage = components['schemas']['TextImportPreviewPage']
type V2TranslationBatchAccepted = components['schemas']['JobBatchAccepted']
type V2MultiChapterTranslationBatchAccepted =
  components['schemas']['TranslationBatchAccepted']

interface TranslationStyleSource {
  pageId: string
  documentRevision: number
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

export async function createChapterRemoveTextJob(
  chapterId: string,
  pageIds: string[],
  executionMode: 'sequential' | 'parallel',
  styleSource?: TranslationStyleSource,
): Promise<V2TranslationBatchAccepted> {
  return apiClient.post<V2TranslationBatchAccepted>(
    `/api/v2/chapters/${encodeURIComponent(chapterId)}/remove-text-jobs`,
    {
      executionMode,
      pageIds,
      ...(styleSource
        ? {
            styleSourcePageId: styleSource.pageId,
            styleSourceDocumentRevision: styleSource.documentRevision,
          }
        : {}),
    },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createTranslationBatch(
  target: { bookIds: string[] } | { chapterIds: string[] },
  config: TranslationJobConfig = { mode: 'standard' },
): Promise<V2MultiChapterTranslationBatchAccepted> {
  return apiClient.post<V2MultiChapterTranslationBatchAccepted>(
    '/api/v2/translation-batches',
    { ...target, config },
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
  return apiClient.upload<V2TextImportPreview>(
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
