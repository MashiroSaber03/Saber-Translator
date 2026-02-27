import { apiClient } from './client'
import type { CompareRun, JobItem } from '@/types'

export interface CompareVariantInput {
  id?: string
  name?: string
  text: string
}

export interface CompareRunPayload {
  session?: string
  page_index?: number
  baseline?: CompareVariantInput
  candidates?: CompareVariantInput[]
  baseline_text?: string
  candidate_text?: string
}

export interface CompareRunResponse {
  success: boolean
  run?: CompareRun
  job?: JobItem
  error?: string
}

export async function runCompare(payload: CompareRunPayload): Promise<CompareRunResponse> {
  return apiClient.post<CompareRunResponse>('/api/compare/run', payload)
}

export async function getCompareRun(runId: string): Promise<CompareRunResponse> {
  return apiClient.get<CompareRunResponse>(`/api/compare/${runId}`)
}

