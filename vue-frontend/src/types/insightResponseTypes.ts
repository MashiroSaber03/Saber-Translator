export type InsightTaskStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'cancelled'
  | 'failed'

export interface InsightAnalysisSnapshot {
  fullyAnalyzed: boolean
  analyzedPagesCount: number
  currentTask?: {
    jobId: string
    status: InsightTaskStatus
    progress: {
      analyzedPages: number
      totalPages: number
    }
  }
}
