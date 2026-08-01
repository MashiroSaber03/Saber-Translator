import type { BookTranslationConstraints } from './bookTranslationConstraints'

export type JobStatusSummary = Partial<Record<
  'queued' | 'running' | 'pausing' | 'paused' | 'cancelling' | 'interrupted' | 'failed',
  number
>>

export interface BookData {
  id: string
  title: string
  cover?: string
  description?: string
  tags?: string[]
  translationConstraints?: BookTranslationConstraints
  chapters?: ChapterData[]
  chapterCount?: number
  totalPages?: number
  createdAt?: string
  updatedAt?: string
  chapterOrderRevision?: number
  jobStatusSummary?: JobStatusSummary
}

export interface ChapterData {
  id: string
  title: string
  order: number
  imageCount?: number
  ordinal?: number
  pageOrderRevision?: number
  jobStatusSummary?: JobStatusSummary
}

export interface TagData {
  id?: string
  name: string
  color?: string
  bookCount?: number
}
