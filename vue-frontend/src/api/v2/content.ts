import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type V2Book = components['schemas']['Book']
export type V2BookDetail = components['schemas']['BookDetail']
export type V2Chapter = components['schemas']['Chapter']
export type V2ChapterNavigation = components['schemas']['ChapterNavigation']
export type V2ContainerImportAccepted = components['schemas']['JobBatchAccepted']
export type V2ImportLease = components['schemas']['ImportLease']
export type V2PageDocument = components['schemas']['PageDocument']
export type V2PageDocumentBatchMutation = components['schemas']['PageDocumentBatchMutation']
export type V2PageImportResult = components['schemas']['PageImportResult']
export type V2PageList = components['schemas']['PageList']
export type V2PageSummary = components['schemas']['PageSummary']
export type V2TranslationBootstrap = components['schemas']['TranslationBootstrap']

const API_ROOT = '/api/v2'
const naturalPathCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
})

export interface BrowserImportFile {
  file: File
  logicalPath: string
}

export interface SequentialImportProgress {
  completed: number
  currentPath: string
  result: V2PageImportResult
  total: number
}

export function newIdempotencyKey(): string {
  return crypto.randomUUID()
}

export function browserImportFiles(files: FileList | File[]): BrowserImportFile[] {
  return Array.from(files)
    .map(file => ({
      file,
      logicalPath: file.webkitRelativePath || file.name,
    }))
    .sort((left, right) => naturalPathCollator.compare(
      left.logicalPath,
      right.logicalPath,
    ))
}

export async function listChapterPages(
  chapterId: string,
  options: { all?: boolean; cursor?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<V2PageList> {
  const query = new URLSearchParams()
  if (options.all) query.set('all', '1')
  if (options.cursor) query.set('cursor', String(options.cursor))
  if (options.limit) query.set('limit', String(options.limit))
  const suffix = query.size > 0 ? `?${query}` : ''
  return apiClient.get<V2PageList>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/pages${suffix}`,
    { signal: options.signal },
  )
}

export async function getPageSummary(
  pageId: string,
  signal?: AbortSignal,
): Promise<V2PageSummary> {
  return apiClient.get<V2PageSummary>(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}`,
    { signal },
  )
}

export async function getBook(bookId: string, signal?: AbortSignal): Promise<V2BookDetail> {
  return apiClient.get<V2BookDetail>(
    `${API_ROOT}/books/${encodeURIComponent(bookId)}`,
    { signal },
  )
}

export async function getTranslationBootstrap(
  options: { bookId?: string; chapterId?: string; signal?: AbortSignal } = {},
): Promise<V2TranslationBootstrap> {
  const query = new URLSearchParams()
  if (options.bookId && options.chapterId) {
    query.set('bookId', options.bookId)
    query.set('chapterId', options.chapterId)
  }
  const suffix = query.size > 0 ? `?${query}` : ''
  return apiClient.get<V2TranslationBootstrap>(
    `${API_ROOT}/translation/bootstrap${suffix}`,
    { signal: options.signal },
  )
}

export async function createImportLease(
  chapterId: string,
): Promise<V2ImportLease> {
  return apiClient.post<V2ImportLease>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/import-leases`,
    undefined,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function releaseImportLease(
  chapterId: string,
  lease: V2ImportLease,
): Promise<void> {
  await apiClient.delete(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/import-leases/${encodeURIComponent(lease.leaseId)}`,
    {
      headers: {
        'Idempotency-Key': newIdempotencyKey(),
        'Import-Lease-Token': lease.ownerToken,
      },
    },
  )
}

export async function importChapterPage(
  chapterId: string,
  entry: BrowserImportFile,
  lease: V2ImportLease,
  options: { idempotencyKey: string; signal?: AbortSignal },
): Promise<V2PageImportResult> {
  const body = new FormData()
  body.append('file', entry.file, entry.file.name)
  body.append('logicalPath', entry.logicalPath)
  return apiClient.upload<V2PageImportResult>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/pages`,
    body,
    {
      signal: options.signal,
      headers: {
        'Idempotency-Key': options.idempotencyKey,
        'Import-Lease-Id': lease.leaseId,
        'Import-Lease-Token': lease.ownerToken,
      },
    },
  )
}

export async function importImagesSequentially(
  chapterId: string,
  files: FileList | File[],
  options: {
    onProgress?: (progress: SequentialImportProgress) => void
    signal?: AbortSignal
  } = {},
): Promise<V2PageImportResult[]> {
  const ordered = browserImportFiles(files)
  const lease = await createImportLease(chapterId)
  const results: V2PageImportResult[] = []
  try {
    for (const [index, entry] of ordered.entries()) {
      options.signal?.throwIfAborted()
      const result = await importChapterPage(chapterId, entry, lease, {
        idempotencyKey: newIdempotencyKey(),
        signal: options.signal,
      })
      results.push(result)
      options.onProgress?.({
        completed: index + 1,
        currentPath: entry.logicalPath,
        result,
        total: ordered.length,
      })
    }
    return results
  } finally {
    await releaseImportLease(chapterId, lease)
  }
}

export async function createContainerImportJob(
  chapterId: string,
  file: File,
): Promise<V2ContainerImportAccepted> {
  const body = new FormData()
  body.append('file', file, file.name)
  return apiClient.upload<V2ContainerImportAccepted>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/container-import-jobs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function deletePage(pageId: string): Promise<void> {
  await apiClient.delete(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function resetQuickWorkspace(): Promise<V2TranslationBootstrap> {
  await apiClient.post(
    `${API_ROOT}/quick-workspace/reset`,
    undefined,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
  return getTranslationBootstrap()
}

export function updateLastVisitedPage(
  chapterId: string,
  pageId: string,
  baseRevision: number,
): Promise<V2ChapterNavigation> {
  return apiClient.patch<V2ChapterNavigation>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/last-visited-page`,
    { pageId, baseRevision },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function getPageDocument(
  pageId: string,
  signal?: AbortSignal,
): Promise<V2PageDocument> {
  return apiClient.get<V2PageDocument>(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}/document`,
    { signal },
  )
}

export async function mutatePageDocument(
  pageId: string,
  command: V2PageDocumentBatchMutation,
): Promise<V2PageDocument> {
  return apiClient.patch<V2PageDocument>(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}/document`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
