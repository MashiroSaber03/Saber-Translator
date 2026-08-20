import { apiClient, ApiClientError } from '@/api/client'
import type { components } from '@/api/generated/v2'
import type { TextStyleSettings } from '@/types/settings'

export type V2BookDetail = components['schemas']['BookDetail']
export type V2Chapter = components['schemas']['Chapter']
export type V2ChapterNavigation = components['schemas']['ChapterNavigation']
export type V2ChapterSettingsMemory = components['schemas']['ChapterSettingsMemory']
export type V2ContainerImportAccepted = components['schemas']['JobBatchAccepted']
export type V2PageDocument = components['schemas']['PageDocument']
export type V2PageDocumentBatchMutation = components['schemas']['PageDocumentBatchMutation']
export type V2PageDocumentMutationResponse = components['schemas']['PageDocumentMutationResponse']
export type V2CompleteBubbleMutationFields = components['schemas']['CompleteBubbleMutationFields']
export type V2PageImportResult = components['schemas']['PageImportResult']
export type V2PageList = components['schemas']['PageList']
export type V2PageRenderStatus = components['schemas']['PageRenderStatus']
export type V2PageSummary = components['schemas']['PageSummary']
export type V2QuickWorkspaceContext = components['schemas']['QuickWorkspaceContext']
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
  failed: number
  result?: V2PageImportResult
  error?: Error
  succeeded: number
  total: number
}

export interface SequentialImportRetry {
  attempt: number
  currentPath: string
  maxAttempts: number
}

export interface SequentialImportFailure {
  entry: BrowserImportFile
  error: Error
  idempotencyKey: string
}

export interface SequentialImportSummary {
  failures: SequentialImportFailure[]
  results: V2PageImportResult[]
}

interface PendingImageImport {
  entry: BrowserImportFile
  idempotencyKey: string
}

export interface SequentialImportOptions {
  onProgress?: (progress: SequentialImportProgress) => void
  onRetry?: (retry: SequentialImportRetry) => void
  signal?: AbortSignal
}

const UPLOAD_RETRY_DELAYS_MS = [250, 750, 1_500, 3_000] as const
const RETRYABLE_UPLOAD_CODES = new Set([
  'ECONNABORTED',
  'ERR_NETWORK',
  'ETIMEDOUT',
  'network_error',
  'proxy_connection_error',
])
const RETRYABLE_UPLOAD_STATUSES = new Set([408, 425, 429, 502, 503, 504])

export function newIdempotencyKey(): string {
  return crypto.randomUUID()
}

function browserImportFiles(files: FileList | File[]): BrowserImportFile[] {
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
  if (options.cursor !== undefined) query.set('cursor', String(options.cursor))
  if (options.limit !== undefined) query.set('limit', String(options.limit))
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

export async function getPageRenderStatus(
  pageId: string,
  signal?: AbortSignal,
): Promise<V2PageRenderStatus> {
  return apiClient.get<V2PageRenderStatus>(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}/render-status`,
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
  if (options.bookId !== undefined) {
    query.set('bookId', options.bookId)
  }
  if (options.chapterId !== undefined) {
    query.set('chapterId', options.chapterId)
  }
  const suffix = query.size > 0 ? `?${query}` : ''
  return apiClient.get<V2TranslationBootstrap>(
    `${API_ROOT}/translation/bootstrap${suffix}`,
    { signal: options.signal },
  )
}

async function importChapterPage(
  chapterId: string,
  entry: BrowserImportFile,
  options: {
    idempotencyKey: string
    signal?: AbortSignal
    textStyleJson: string
  },
): Promise<V2PageImportResult> {
  const body = new FormData()
  body.append('file', entry.file, entry.file.name)
  body.append('logicalPath', entry.logicalPath)
  body.append('textStyle', options.textStyleJson)
  return apiClient.upload<V2PageImportResult>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/pages`,
    body,
    {
      signal: options.signal,
      headers: {
        'Idempotency-Key': options.idempotencyKey,
      },
    },
  )
}

function normalizeImportError(error: unknown): Error {
  return error instanceof Error ? error : new Error('图片写入后端失败')
}

function isRetryableUploadError(error: Error): boolean {
  return error instanceof ApiClientError
    && (
      error.status === 0
      || RETRYABLE_UPLOAD_CODES.has(error.code)
      || RETRYABLE_UPLOAD_STATUSES.has(error.status)
    )
}

function waitForUploadRetry(delayMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort)
      resolve()
    }, delayMs)
    const abort = () => {
      clearTimeout(timer)
      reject(signal?.reason)
    }
    signal?.addEventListener('abort', abort, { once: true })
  })
}

async function importPendingImagesSequentially(
  chapterId: string,
  pending: PendingImageImport[],
  textStyle: TextStyleSettings,
  options: SequentialImportOptions,
): Promise<SequentialImportSummary> {
  const results: V2PageImportResult[] = []
  const failures: SequentialImportFailure[] = []
  const maxAttempts = UPLOAD_RETRY_DELAYS_MS.length + 1
  const textStyleJson = JSON.stringify(textStyle)

  for (const item of pending) {
    options.signal?.throwIfAborted()
    let result: V2PageImportResult | undefined
    let finalError: Error | undefined

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        result = await importChapterPage(chapterId, item.entry, {
          idempotencyKey: item.idempotencyKey,
          signal: options.signal,
          textStyleJson,
        })
        break
      } catch (error) {
        if (options.signal?.aborted) throw error
        const normalized = normalizeImportError(error)
        if (!isRetryableUploadError(normalized) || attempt === maxAttempts) {
          finalError = normalized
          break
        }
        options.onRetry?.({
          attempt: attempt + 1,
          currentPath: item.entry.logicalPath,
          maxAttempts,
        })
        await waitForUploadRetry(UPLOAD_RETRY_DELAYS_MS[attempt - 1], options.signal)
      }
    }

    if (result) {
      results.push(result)
    } else {
      failures.push({
        entry: item.entry,
        error: finalError || new Error('图片写入后端失败'),
        idempotencyKey: item.idempotencyKey,
      })
    }
    options.onProgress?.({
      completed: results.length + failures.length,
      currentPath: item.entry.logicalPath,
      failed: failures.length,
      ...(result ? { result } : { error: finalError }),
      succeeded: results.length,
      total: pending.length,
    })
  }
  return { failures, results }
}

export async function importImagesSequentially(
  chapterId: string,
  files: FileList | File[],
  textStyle: TextStyleSettings,
  options: SequentialImportOptions = {},
): Promise<SequentialImportSummary> {
  const pending = browserImportFiles(files).map(entry => ({
    entry,
    idempotencyKey: newIdempotencyKey(),
  }))
  return importPendingImagesSequentially(chapterId, pending, textStyle, options)
}

export async function retryFailedImageImports(
  chapterId: string,
  failures: SequentialImportFailure[],
  textStyle: TextStyleSettings,
  options: SequentialImportOptions = {},
): Promise<SequentialImportSummary> {
  return importPendingImagesSequentially(
    chapterId,
    failures.map(({ entry, idempotencyKey }) => ({ entry, idempotencyKey })),
    textStyle,
    options,
  )
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
  await apiClient.delete(`${API_ROOT}/pages/${encodeURIComponent(pageId)}`)
}

export async function clearChapterPages(chapterId: string): Promise<number> {
  const result = await apiClient.delete<{ deletedCount: number }>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/pages`,
  )
  return result.deletedCount
}

export async function resetQuickWorkspace(): Promise<V2QuickWorkspaceContext> {
  return apiClient.post<V2QuickWorkspaceContext>(
    `${API_ROOT}/quick-workspace/reset`,
  )
}

export type QuickWorkspacePromoteCommand =
  | { mode: 'new_book'; title: string; chapterTitle: string }
  | { mode: 'existing_book'; bookId: string; chapterTitle: string }

export type QuickWorkspacePromotion = components['schemas']['QuickWorkspacePromotion']

export function promoteQuickWorkspace(
  command: QuickWorkspacePromoteCommand,
): Promise<QuickWorkspacePromotion> {
  return apiClient.post<QuickWorkspacePromotion>(
    `${API_ROOT}/quick-workspace/promote`,
    command,
  )
}

export function updateLastVisitedPage(
  chapterId: string,
  pageId: string,
): Promise<V2ChapterNavigation> {
  return apiClient.patch<V2ChapterNavigation>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/last-visited-page`,
    { pageId },
  )
}

export function updateChapterSettingsMemory(
  chapterId: string,
  payload: Record<string, unknown>,
  baseRevision: number,
): Promise<V2ChapterSettingsMemory> {
  return apiClient.patch<V2ChapterSettingsMemory>(
    `${API_ROOT}/chapters/${encodeURIComponent(chapterId)}/settings-memory`,
    { payload, baseRevision },
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
  idempotencyKey: string,
): Promise<V2PageDocumentMutationResponse> {
  return apiClient.patch<V2PageDocumentMutationResponse>(
    `${API_ROOT}/pages/${encodeURIComponent(pageId)}/document`,
    command,
    { headers: { 'Idempotency-Key': idempotencyKey } },
  )
}
