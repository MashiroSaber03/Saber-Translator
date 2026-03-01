import { apiClient } from './client'
import type { JobItem, CompareRun, QualityReport } from '@/types'

export interface JobsListResponse {
  success: boolean
  jobs?: JobItem[]
  total?: number
  error?: string
}

export interface JobActionResponse {
  success: boolean
  job?: JobItem
  run?: CompareRun
  report?: QualityReport
  error?: string
}

export async function getJobs(params?: {
  status?: string
  type?: string
  limit?: number
}): Promise<JobsListResponse> {
  return apiClient.get<JobsListResponse>('/api/jobs', { params })
}

export async function retryJob(jobId: string): Promise<JobActionResponse> {
  return apiClient.post<JobActionResponse>(`/api/jobs/${jobId}/retry`, {})
}

export async function cancelJob(jobId: string, reason = ''): Promise<JobActionResponse> {
  return apiClient.post<JobActionResponse>(`/api/jobs/${jobId}/cancel`, { reason })
}

