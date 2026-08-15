import type { BookTranslationConstraints } from './bookTranslationConstraints'
import type { components } from '@/api/generated/v2'

export type JobStatusSummary = components['schemas']['JobStatusSummary']

export interface BookData {
  id: string
  title: string
  cover?: string
  tags?: string[]
  translationConstraints?: BookTranslationConstraints
  chapters?: ChapterData[]
  chapterCount?: number
  totalPages?: number
  createdAt?: string
  updatedAt?: string
  jobStatusSummary?: JobStatusSummary
}

export interface ChapterData {
  id: string
  title: string
  order: number
  imageCount?: number
  jobStatusSummary?: JobStatusSummary
}

export type TagData = components['schemas']['Tag']
