import type { BookData, ChapterData } from '@/types/api'

export function normalizeChapterData(chapter: ChapterData): ChapterData {
  return {
    ...chapter,
    imageCount: chapter.imageCount ?? chapter.image_count ?? chapter.page_count ?? 0,
    hasSession: chapter.hasSession ?? chapter.has_session ?? Boolean(chapter.session_path),
  }
}

export function normalizeBookData(book: BookData): BookData {
  const chapters = book.chapters?.map(normalizeChapterData)

  return {
    ...book,
    chapters,
    chapterCount: book.chapterCount ?? book.chapter_count ?? chapters?.length ?? 0,
    totalPages: book.totalPages ?? book.total_pages ?? 0,
    createdAt: book.createdAt ?? book.created_at ?? '',
    updatedAt: book.updatedAt ?? book.updated_at ?? '',
  }
}
