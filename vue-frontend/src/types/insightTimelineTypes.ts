export interface TimelinePageRange {
  start: number
  end: number
}

export interface TimelineGroup {
  id: string
  page_range: TimelinePageRange
  events: string[]
  summary: string
  thumbnail_page: number
}

export interface TimelineStats {
  total_events: number
  total_pages: number
  total_arcs?: number
  total_characters?: number
  total_threads?: number
}

export interface TimelineCharacter {
  character_id: string
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
  id: string
  name: string
  description: string
  page_range: TimelinePageRange
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
  timeline_version_id: string
  mode: 'enhanced' | 'compressed' | 'simple'
  groups: TimelineGroup[]
  events: TimelineEvent[]
  stats: TimelineStats
  story_summary: string
  main_characters: TimelineCharacter[]
  page_thumbnails: Record<number, string>
  plot_arcs?: TimelineArc[]
  plot_threads?: PlotThread[]
  next_event_cursor: number | null
  next_character_cursor: string | null
}

export interface TimelineEvent {
  eventId: string
  summary: string
  page_ids: string[]
  page_numbers: number[]
  importance?: string
}
