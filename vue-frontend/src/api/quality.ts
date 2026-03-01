import { apiClient } from './client'
import type { JobItem, QualityReport } from '@/types'

export interface AnalyzeQualityResponse {
  success: boolean
  job?: JobItem
  report?: QualityReport
  error?: string
  job_id?: string
}

export interface QualityReportResponse {
  success: boolean
  report?: QualityReport
  error?: string
}

export async function analyzeQuality(session: string): Promise<AnalyzeQualityResponse> {
  return apiClient.post<AnalyzeQualityResponse>('/api/quality/analyze', { session })
}

export async function getQualityReport(session: string): Promise<QualityReportResponse> {
  return apiClient.get<QualityReportResponse>(`/api/quality/report/${session}`)
}
