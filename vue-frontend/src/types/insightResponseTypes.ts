import type { components } from '@/api/generated/v2'

export type InsightTaskStatus = components['schemas']['JobStatus']

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
