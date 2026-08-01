import type { BookData, ChapterData } from '@/types/api'

function normalizeTextField(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return fallback
}

export function normalizeChapterData(chapter: ChapterData): ChapterData {
  return {
    ...chapter,
    title: normalizeTextField(chapter.title),
    imageCount: chapter.imageCount ?? 0,
  }
}

export function normalizeBookData(book: BookData): BookData {
  const chapters = book.chapters?.map(normalizeChapterData)

  return {
    ...book,
    title: normalizeTextField(book.title),
    description: normalizeTextField(book.description, ''),
    chapters,
    chapterCount: book.chapterCount ?? chapters?.length ?? 0,
    totalPages: book.totalPages ?? 0,
    createdAt: normalizeTextField(book.createdAt),
    updatedAt: normalizeTextField(book.updatedAt),
  }
}
