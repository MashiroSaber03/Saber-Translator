import type {
  AnalysisProgress,
  AnalysisTask,
  BookOverview,
  PageAnalysis,
  TaskStatus,
} from './insightAnalysisTypes'
import type { NoteData } from './insightNotesQaTypes'
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
  overview?: BookOverview
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

export interface PageDataResponse {
  success: boolean
  page?: {
    page_num: number
    summary?: string
    dialogues?: Array<{
      character?: string
      text: string
      translated_text?: string
    }>
    analyzed: boolean
  }
  analysis?: PageAnalysis
  error?: string
}

export interface InsightChapterListResponse {
  success: boolean
  chapters?: Array<{
    id: string
    title: string
    start_page: number
    end_page: number
  }>
  error?: string
}

export interface NoteListResponse {
  success: boolean
  notes?: NoteData[]
  error?: string
}

export interface ConnectionTestResponse {
  success: boolean
  message?: string
  error?: string
}
