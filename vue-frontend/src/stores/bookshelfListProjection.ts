import type { BookData } from '@/types/api'
import { naturalSortCompare } from '@/utils'

export type BookSortBy = 'title' | 'createdAt' | 'updatedAt'
export type SortOrder = 'asc' | 'desc'

export interface BookshelfListProjectionOptions {
  searchKeyword: string
  selectedTagNames: string[]
  sortBy: BookSortBy
  sortOrder: SortOrder
}

function getBookSortValue(book: BookData, sortBy: BookSortBy): string {
  if (sortBy === 'title') return book.title
  if (sortBy === 'createdAt') return book.createdAt || ''
  return book.updatedAt || ''
}

export function projectBookshelfBooks(
  books: BookData[],
  options: BookshelfListProjectionOptions
): BookData[] {
  const keyword = options.searchKeyword.trim().toLowerCase()
  const filtered = books.filter(book => {
    const matchesKeyword =
      !keyword ||
      book.title.toLowerCase().includes(keyword) ||
      Boolean(book.description?.toLowerCase().includes(keyword))
    const matchesTags =
      options.selectedTagNames.length === 0 ||
      options.selectedTagNames.every(tagName => book.tags?.includes(tagName))

    return matchesKeyword && matchesTags
  })

  const direction = options.sortOrder === 'asc' ? 1 : -1
  return [...filtered].sort((a, b) => {
    const primary = naturalSortCompare(
      getBookSortValue(a, options.sortBy),
      getBookSortValue(b, options.sortBy)
    )
    if (primary !== 0) return primary * direction
    return naturalSortCompare(a.title, b.title)
  })
}
