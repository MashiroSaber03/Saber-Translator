import type { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, TagData } from '@/types/api'

type BookshelfStore = ReturnType<typeof useBookshelfStore>

export function setTestBooks(store: BookshelfStore, books: BookData[]): void {
  store.books = books
}

export function setTestTags(store: BookshelfStore, tags: TagData[]): void {
  store.tags = tags
}
