import type {
  AnalysisProgress,
  AnalysisTask,
  TaskStatus,
} from './insightAnalysisTypes'
import type { TimelineData } from './insightTimelineTypes'

export interface InsightStatusResponse {
  success: boolean
  book_id?: string
  analyzed?: boolean
  fully_analyzed?: boolean
  completion_ratio?: number
  status?: TaskStatus
  task?: AnalysisTask
  current_task?: AnalysisTask
  progress?: AnalysisProgress
  total_pages?: number
  analyzed_pages?: number
  analyzed_pages_count?: number
  has_overview?: boolean
  has_timeline?: boolean
  error?: string
}

export interface InsightOverviewResponse {
  success: boolean
  content?: string
  template_key?: string
  generated_at?: string
  error?: string
}

export interface InsightTimelineResponse {
  success: boolean
  timeline?: TimelineData
  task_id?: string
  message?: string
  error?: string
}
