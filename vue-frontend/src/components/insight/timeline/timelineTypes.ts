export interface TimelinePageRange {
  start: number
  end: number
}

export interface TimelineGroup {
  id: string
  page_range: TimelinePageRange
  events: string[]
  summary?: string
  thumbnail_page?: number
  plot_arc?: string
  characters?: string[]
  clues?: string[]
  mood?: string
}

export interface TimelineStats {
  total_events: number
  total_pages: number
  total_arcs?: number
  total_characters?: number
  total_threads?: number
}

export interface TimelineCharacter {
  name: string
  description: string
  first_appearance: number
  arc?: string
  key_moments?: TimelineKeyMoment[]
}

export interface TimelineKeyMoment {
  page?: number
  summary: string
}

export interface TimelineArc {
  id?: string
  name: string
  description?: string
  page_range?: TimelinePageRange
  start_page?: number
  end_page?: number
  mood?: string
  event_ids?: string[]
}

export interface PlotThread {
  id: string
  name: string
  type: string
  status: string
  description?: string
  introduced_at?: number
  resolved_at?: number | null
}

export interface TimelineData {
  mode?: string
  groups?: TimelineGroup[]
  events?: unknown[]
  stats?: TimelineStats
  story_summary?: string
  main_characters?: TimelineCharacter[]
  plot_arcs?: TimelineArc[]
  plot_threads?: PlotThread[]
  cached?: boolean
}
