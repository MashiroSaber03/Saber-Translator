import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type V2Job = components['schemas']['Job']
export type V2JobDetail = components['schemas']['JobDetail']
export type V2JobEvent = components['schemas']['JobEvent']
export type V2JobStatus = components['schemas']['JobStatus']
export type JobListResponse = components['schemas']['JobList']
export type JobSnapshotResponse = components['schemas']['JobSnapshot']
export type JobRetryAccepted = components['schemas']['JobRetryAccepted']
export type JobEventList = components['schemas']['JobEventList']

export const NONTERMINAL_JOB_STATUSES: ReadonlySet<V2JobStatus> = new Set([
  'queued',
  'running',
  'pausing',
  'paused',
  'cancelling',
  'interrupted',
])

export const HISTORY_JOB_STATUSES: ReadonlySet<V2JobStatus> = new Set([
  'cancelled',
  'completed',
  'completed_with_errors',
  'failed',
  'interrupted',
])

export const CURRENT_JOB_STATUSES: ReadonlySet<V2JobStatus> = new Set([
  'running',
  'pausing',
  'paused',
  'cancelling',
])

function replayHeaders(): Record<string, string> {
  return { 'Idempotency-Key': crypto.randomUUID() }
}

export const jobsApi = {
  list(
    scope: 'queue' | 'history',
    filters: { status?: V2JobStatus; type?: V2Job['kind']; bookId?: string } = {},
  ): Promise<JobListResponse> {
    const query = new URLSearchParams({ scope })
    if (scope === 'history') query.set('limit', '200')
    if (filters.status) query.set('status', filters.status)
    if (filters.type) query.set('type', filters.type)
    if (filters.bookId) query.set('book_id', filters.bookId)
    return apiClient.get(`/api/v2/jobs?${query}`)
  },

  get(jobId: string): Promise<V2JobDetail> {
    return apiClient.get(`/api/v2/jobs/${encodeURIComponent(jobId)}`)
  },

  snapshot(jobIds: string[]): Promise<JobSnapshotResponse> {
    const query = new URLSearchParams()
    const uniqueJobIds = [...new Set(jobIds)]
    if (uniqueJobIds.length > 200) {
      throw new Error('一次最多读取 200 个任务快照')
    }
    for (const jobId of uniqueJobIds) {
      query.append('job_id', jobId)
    }
    return apiClient.get(`/api/v2/jobs/snapshot?${query}`)
  },

  events(
    jobId: string,
    cursor: { after?: number; before?: number; limit?: number } = {},
  ): Promise<JobEventList> {
    const query = new URLSearchParams({ limit: String(cursor.limit ?? 200) })
    if (cursor.after !== undefined) query.set('after', String(cursor.after))
    if (cursor.before !== undefined) query.set('before', String(cursor.before))
    return apiClient.get(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/events?${query}`,
    )
  },

  pause(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/pause`,
    )
  },

  resume(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/resume`,
    )
  },

  continue(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/continue`,
    )
  },

  cancel(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/cancel`,
    )
  },

  retry(jobId: string, strategy: 'current' | 'original' = 'current'): Promise<JobRetryAccepted> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/retry`,
      { strategy },
      { headers: replayHeaders() },
    )
  },

  retryFailed(
    jobId: string,
    strategy: 'current' | 'original' = 'current',
  ): Promise<JobRetryAccepted> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/retry-failed`,
      { strategy },
      { headers: replayHeaders() },
    )
  },

  reorder(orderedJobIds: string[], baseRevision: number): Promise<{ queueRevision: number }> {
    return apiClient.post(
      '/api/v2/jobs/reorder',
      { orderedJobIds, baseRevision },
    )
  },

  cancelQueued(): Promise<{ cancelled: number }> {
    return apiClient.post('/api/v2/jobs/cancel-queued')
  },

  clearHistory(): Promise<{ removed: number }> {
    return apiClient.post('/api/v2/jobs/history/clear')
  },

  cancelBatch(batchId: string): Promise<{ cancelled: number }> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/cancel`,
    )
  },

  prioritizeBatch(
    batchId: string,
    baseRevision: number,
  ): Promise<{ queueRevision: number }> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/prioritize`,
      { baseRevision },
    )
  },

  continueBatch(batchId: string): Promise<components['schemas']['BatchContinueResult']> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/continue`,
    )
  },
}
