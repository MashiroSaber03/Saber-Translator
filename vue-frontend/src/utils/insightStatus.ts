import type { InsightStatusResponse } from '@/types'
import type { AnalysisStatus } from '@/types/insight'

/**
 * 统一解析分析状态：
 * 1) 有 current_task 时优先任务状态
 * 2) 无任务时优先 fully_analyzed，再兼容 analyzed
 */
export function resolveAnalysisStatus(response: InsightStatusResponse): AnalysisStatus {
  const taskStatus = response.current_task?.status
  const hasFullyAnalyzedField = response.fully_analyzed !== undefined && response.fully_analyzed !== null

  if (taskStatus === 'running') return 'running'
  if (taskStatus === 'paused') return 'paused'
  if (taskStatus === 'failed') return 'failed'
  if (taskStatus === 'completed') {
    // completed 任务不代表全书完成，仍以 fully_analyzed 作为完成语义基准
    if (response.fully_analyzed === true) return 'completed'
    if (hasFullyAnalyzedField) return 'idle'
    return response.analyzed ? 'completed' : 'idle'
  }
  if (response.fully_analyzed === true) {
    return 'completed'
  }
  if (hasFullyAnalyzedField) {
    return 'idle'
  }
  if (response.analyzed) {
    return 'completed'
  }
  return 'idle'
}
