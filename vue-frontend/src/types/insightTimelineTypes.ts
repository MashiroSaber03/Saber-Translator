import type { PageRange } from './insightAnalysisTypes'

export interface TimelineEvent {
  id: string
  page_range: PageRange
  title: string
  description: string
  type?: string
  importance?: number
  characters?: string[]
  arc_id?: string
}

export interface StoryArc {
  id: string
  name: string
  description?: string
  page_range: PageRange
  events: TimelineEvent[]
}

export interface TimelineData {
  events: TimelineEvent[]
  arcs?: StoryArc[]
  characters?: Array<{
    name: string
    appearances: number[]
  }>
  mode?: string
  stats?: {
    total_events: number
    total_arcs?: number
    total_characters?: number
  }
  generated_at?: string
}
