import type { AnalysisStatus, InsightAnalysisSnapshot } from '@/types/insight'

/**
 * 统一解析分析状态：
 * 1) 有 currentTask 时优先任务状态
 * 2) 无任务时以 fullyAnalyzed 判断全书是否完成
 */
export function resolveAnalysisStatus(snapshot: InsightAnalysisSnapshot): AnalysisStatus {
  const taskStatus = snapshot.currentTask?.status
  if (taskStatus === 'running') return 'running'
  if (taskStatus === 'paused') return 'paused'
  if (taskStatus === 'failed') return 'failed'
  if (taskStatus === 'completed') {
    // completed 任务不代表全书完成，仍以 fully_analyzed 作为完成语义基准
    return snapshot.fullyAnalyzed ? 'completed' : 'idle'
  }
  if (snapshot.fullyAnalyzed) {
    return 'completed'
  }
  return 'idle'
}
