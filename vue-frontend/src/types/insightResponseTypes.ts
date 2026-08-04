export type InsightTaskStatus =
  | 'queued'
  | 'running'
  | 'pausing'
  | 'paused'
  | 'cancelling'
  | 'interrupted'
  | 'completed'
  | 'completed_with_errors'
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
