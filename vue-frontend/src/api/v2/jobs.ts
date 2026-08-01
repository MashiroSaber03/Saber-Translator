import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type V2Job = components['schemas']['Job']
export type V2JobDetail = components['schemas']['JobDetail']
export type V2JobEvent = components['schemas']['JobEvent']
export type V2JobStatus = components['schemas']['JobStatus']
export type JobListResponse = components['schemas']['JobList']
export type JobRetryAccepted = components['schemas']['JobRetryAccepted']
export type JobEventList = components['schemas']['JobEventList']

function commandHeaders(): Record<string, string> {
  return { 'Idempotency-Key': crypto.randomUUID() }
}

export const jobsApi = {
  list(
    scope: 'queue' | 'history',
    filters: { status?: V2JobStatus; type?: V2Job['kind']; bookId?: string } = {},
  ): Promise<JobListResponse> {
    const query = new URLSearchParams({ scope, limit: '200' })
    if (filters.status) query.set('status', filters.status)
    if (filters.type) query.set('type', filters.type)
    if (filters.bookId) query.set('book_id', filters.bookId)
    return apiClient.get(`/api/v2/jobs?${query}`)
  },

  get(jobId: string): Promise<V2JobDetail> {
    return apiClient.get(`/api/v2/jobs/${encodeURIComponent(jobId)}`)
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
      `/api/v2/jobs/${jobId}/pause`,
      undefined,
      { headers: commandHeaders() },
    )
  },

  resume(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${jobId}/resume`,
      undefined,
      { headers: commandHeaders() },
    )
  },

  continue(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${jobId}/continue`,
      undefined,
      { headers: commandHeaders() },
    )
  },

  cancel(jobId: string): Promise<V2Job> {
    return apiClient.post(
      `/api/v2/jobs/${jobId}/cancel`,
      undefined,
      { headers: commandHeaders() },
    )
  },

  retry(jobId: string, strategy: 'current' | 'original' = 'current'): Promise<JobRetryAccepted> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/retry`,
      { strategy },
      { headers: commandHeaders() },
    )
  },

  retryFailed(
    jobId: string,
    strategy: 'current' | 'original' = 'current',
  ): Promise<JobRetryAccepted> {
    return apiClient.post(
      `/api/v2/jobs/${encodeURIComponent(jobId)}/retry-failed`,
      { strategy },
      { headers: commandHeaders() },
    )
  },

  reorder(orderedJobIds: string[], baseRevision: number): Promise<{ queueRevision: number }> {
    return apiClient.post(
      '/api/v2/jobs/reorder',
      { orderedJobIds, baseRevision },
      { headers: commandHeaders() },
    )
  },

  cancelQueued(): Promise<{ cancelled: number }> {
    return apiClient.post(
      '/api/v2/jobs/cancel-queued',
      undefined,
      { headers: commandHeaders() },
    )
  },

  clearHistory(): Promise<{ removed: number }> {
    return apiClient.post(
      '/api/v2/jobs/history/clear',
      undefined,
      { headers: commandHeaders() },
    )
  },

  cancelBatch(batchId: string): Promise<{ cancelled: number }> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/cancel`,
      undefined,
      { headers: commandHeaders() },
    )
  },

  prioritizeBatch(
    batchId: string,
    baseRevision: number,
  ): Promise<{ queueRevision: number }> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/prioritize`,
      { baseRevision },
      { headers: commandHeaders() },
    )
  },

  continueBatch(batchId: string): Promise<components['schemas']['BatchContinueResult']> {
    return apiClient.post(
      `/api/v2/job-batches/${encodeURIComponent(batchId)}/continue`,
      undefined,
      { headers: commandHeaders() },
    )
  },
}
