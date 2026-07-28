import type { BookTranslationConstraints } from './bookTranslationConstraints'

export interface BookData {
  id: string
  title: string
  cover?: string
  description?: string
  tags?: string[]
  translation_constraints?: BookTranslationConstraints
  chapters?: ChapterData[]
  chapterCount?: number
  chapter_count?: number
  totalPages?: number
  total_pages?: number
  createdAt?: string
  updatedAt?: string
  created_at?: string
  updated_at?: string
  chapterOrderRevision?: number
}

export interface ChapterData {
  id: string
  title: string
  order: number
  imageCount?: number
  image_count?: number
  page_count?: number
  hasSession?: boolean
  has_session?: boolean
  session_path?: string
  ordinal?: number
  pageOrderRevision?: number
}

export interface TagData {
  id?: string
  name: string
  color?: string
  book_count?: number
}
