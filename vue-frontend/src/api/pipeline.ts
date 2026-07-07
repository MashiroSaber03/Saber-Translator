import { apiClient } from './client'
import type { ApiError, ApiResponse } from '@/types'

export type PipelineMode = 'standard' | 'hq' | 'proofread' | 'remove_text'
export type PipelineScope = 'current' | 'all' | 'selection' | 'failed'

export interface PipelineBeforePayload {
  pipeline_id: string
  mode: PipelineMode
  scope: PipelineScope
  page_indexes: number[]
  total_images: number
  settings_snapshot?: Record<string, unknown>
}

export interface PipelineAfterPayload {
  pipeline_id: string
  mode: PipelineMode
  scope: PipelineScope
  completed: number
  failed: number
  errors?: string[]
  warnings_count?: number
  duration_ms?: number
}

interface PipelineResponse extends ApiResponse {
  pipeline_id?: string
  payload?: Record<string, unknown>
}

export class PipelineCancelledError extends Error {
  readonly pipelineId: string
  readonly details: Record<string, unknown>

  constructor(message: string, pipelineId: string, details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'PipelineCancelledError'
    this.pipelineId = pipelineId
    this.details = details
  }
}

function isPluginCancelError(err: unknown): err is ApiError {
  if (!err || typeof err !== 'object') {
    return false
  }
  const candidate = err as Partial<ApiError>
  return candidate.status === 409 && Boolean(candidate.details?.cancelled_by_plugin)
}

export async function notifyPipelineBefore(
  payload: PipelineBeforePayload
): Promise<PipelineResponse> {
  try {
    return await apiClient.post<PipelineResponse>('/api/pipeline/before', payload)
  } catch (err) {
    if (isPluginCancelError(err)) {
      throw new PipelineCancelledError(
        err.message || '任务被插件取消',
        payload.pipeline_id,
        err.details ?? {}
      )
    }
    throw err
  }
}

export async function notifyPipelineAfter(payload: PipelineAfterPayload): Promise<void> {
  try {
    await apiClient.post<PipelineResponse>('/api/pipeline/after', payload)
  } catch {
    return
  }
}
