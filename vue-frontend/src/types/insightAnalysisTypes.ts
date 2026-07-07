export interface PageRange {
  start: number
  end: number
}

export interface PageAnalysis {
  page_number?: number
  page_num?: number
  page_summary?: string
  summary?: string
  scene?: string
  mood?: string
  from_batch?: boolean
  batch_range?: PageRange
  analyzed_at?: string
  panels?: Array<{
    dialogues?: Array<{
      speaker_name?: string
      character?: string
      text?: string
      translated_text?: string
    }>
  }>
}

export interface BatchAnalysis {
  page_range: PageRange
  pages: PageAnalysis[]
  batch_summary: string
  key_events: string[]
  continuity_notes?: string
  analyzed_at?: string
  parse_error?: boolean
}

export interface SegmentSummary {
  segment_id: string
  page_range: PageRange
  summary: string
  key_events?: string[]
  plot_progression?: string
  themes?: string[]
  batch_count?: number
  generated_at?: string
}

export interface ChapterAnalysis {
  chapter_id: string
  title: string
  page_range: PageRange
  summary: string
  main_plot?: string
  plot_events?: string[]
  themes?: string[]
  atmosphere?: string
  connections?: {
    previous?: string
    foreshadowing?: string
  }
  segment_count?: number
  batch_count?: number
  analysis_mode?: string
  analyzed_at?: string
}

export interface BookOverview {
  book_id: string
  title: string
  total_pages: number
  total_chapters: number
  summary: string
  section_summaries?: string[]
  summary_source?: string
  themes?: string[]
  generated_at?: string
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
