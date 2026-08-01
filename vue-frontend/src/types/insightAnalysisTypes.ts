export interface PageRange {
  start: number
  end: number
}

export type TaskStatus = 'pending' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed'

export type TaskType = 'full_book' | 'chapter' | 'incremental' | 'reanalyze' | 'embeddings_rebuild'

export interface AnalysisProgress {
  current_phase: string
  current_page: number
  analyzed_pages: number
  total_pages: number
  percentage?: number
}

export interface AnalysisTask {
  task_id: string
  book_id: string
  task_type: TaskType
  status: TaskStatus
  progress: AnalysisProgress
  target_chapters?: string[]
  target_pages?: number[]
  is_incremental?: boolean
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  failed_pages?: number[]
}
