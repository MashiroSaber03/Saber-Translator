/**
 * 翻译 Pipeline 生命周期 API
 *
 * 在每次「完整翻译任务」开始 / 结束时，分别调用 /api/pipeline/before 与
 * /api/pipeline/after，让后端的 before_pipeline / after_pipeline 插件钩子有机会执行。
 */

import { apiClient } from './client'
import type { ApiError, ApiResponse } from '@/types'

/** 翻译模式（与后端 PLUGIN_MODES 对齐） */
export type PipelineMode = 'standard' | 'hq' | 'proofread' | 'remove_text'

/** 翻译范围（与前端 PipelineConfig.scope 对齐） */
export type PipelineScope = 'current' | 'all' | 'range' | 'failed'

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

/** 插件取消任务时抛出的专用错误 */
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

/**
 * apiClient 的响应拦截器会把 axios 错误转成 ApiError（{code, message, status, details}），
 * 这里直接通过该形状识别 409 + cancelled_by_plugin 信号。
 */
function isPluginCancelError(err: unknown): err is ApiError {
  if (!err || typeof err !== 'object') {
    return false
  }
  const candidate = err as Partial<ApiError>
  return candidate.status === 409 && Boolean(candidate.details?.cancelled_by_plugin)
}

/**
 * 通知后端「一次完整翻译任务即将开始」。
 *
 * 若任一插件的 before_pipeline 抛出 PluginException（且 failure_policy='fail'），
 * 本函数会抛 PipelineCancelledError；调用方应捕获并中止后续翻译流程。
 */
export async function notifyPipelineBefore(payload: PipelineBeforePayload): Promise<PipelineResponse> {
  try {
    return await apiClient.post<PipelineResponse>('/api/pipeline/before', payload)
  } catch (err) {
    if (isPluginCancelError(err)) {
      throw new PipelineCancelledError(err.message || '任务被插件取消', payload.pipeline_id, err.details ?? {})
    }
    throw err
  }
}

/**
 * 通知后端「一次完整翻译任务已结束」。
 *
 * after 阶段不应阻断流程：本函数捕获所有异常并打印 warning，永远不会抛出。
 */
export async function notifyPipelineAfter(payload: PipelineAfterPayload): Promise<void> {
  try {
    await apiClient.post<PipelineResponse>('/api/pipeline/after', payload)
  } catch (err) {
    console.warn('[pipeline.after] 触发失败（已忽略）:', err)
  }
}
