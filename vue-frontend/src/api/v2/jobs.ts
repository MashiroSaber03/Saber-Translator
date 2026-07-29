import apiClient from '@/api/client'
import type { components } from '@/api/generated/v2'

export type V2Job = components['schemas']['Job']
export type V2JobDetail = components['schemas']['JobDetail']
export type V2JobEvent = components['schemas']['JobEvent']
export type V2JobBatch = components['schemas']['JobBatch']
export type V2JobStatus = components['schemas']['JobStatus']
export type JobListResponse = components['schemas']['JobList']

function commandHeaders(): Record<string, string> {
  return { 'Idempotency-Key': crypto.randomUUID() }
}

export const jobsApi = {
  list(scope: 'queue' | 'history'): Promise<JobListResponse> {
    return apiClient.get(`/api/v2/jobs?scope=${scope}&limit=200`)
  },

  get(jobId: string): Promise<V2JobDetail> {
    return apiClient.get(`/api/v2/jobs/${jobId}`)
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
}
