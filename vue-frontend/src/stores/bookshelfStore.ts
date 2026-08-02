import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import * as bookshelfApi from '@/api/bookshelf'
import type { BookData, ChapterData, TagData } from '@/types/api'

export type BookSortBy = 'title' | 'createdAt' | 'updatedAt'
export type SortOrder = 'asc' | 'desc'

interface BookUpdatePayload {
  title?: string
  cover?: File
  tags?: string[]
}

export const useBookshelfStore = defineStore('bookshelf', () => {
  const books = ref<BookData[]>([])
  const tags = ref<TagData[]>([])
  const searchKeyword = ref('')
  const selectedTagNames = ref<string[]>([])
  const sortBy = ref<BookSortBy>('updatedAt')
  const sortOrder = ref<SortOrder>('desc')
  const currentBookId = ref<string | null>(null)
  const currentBookDetail = ref<BookData | null>(null)
  const batchMode = ref(false)
  const selectedBookIds = ref<Set<string>>(new Set())
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const currentBook = computed(() => {
    if (!currentBookId.value) return null
    if (currentBookDetail.value?.id === currentBookId.value) {
      return currentBookDetail.value
    }
    return books.value.find(book => book.id === currentBookId.value) || null
  })
  const searchQuery = computed(() => searchKeyword.value)
  const isAllSelected = computed(
    () => books.value.length > 0 && books.value.every(book => selectedBookIds.value.has(book.id)),
  )

  function setBooks(bookList: BookData[]): void {
    books.value = bookList
    const detail = currentBookDetail.value
    if (!detail) return
    const summary = bookList.find(book => book.id === detail.id)
    if (!summary) return
    currentBookDetail.value = {
      ...detail,
      ...summary,
      chapters: summary.chapters ?? detail.chapters,
      translationConstraints:
        summary.translationConstraints ?? detail.translationConstraints,
    }
  }

  function upsertBook(book: BookData): void {
    const index = books.value.findIndex(item => item.id === book.id)
    if (index >= 0) {
      books.value[index] = book
    } else {
      books.value.unshift(book)
    }
    if (currentBookId.value === book.id) {
      currentBookDetail.value = book
    }
  }

  function eachBookProjection(
    bookId: string,
    update: (book: BookData) => void,
  ): void {
    const listBook = books.value.find(book => book.id === bookId)
    if (listBook) update(listBook)
    const detail = currentBookDetail.value
    if (detail?.id === bookId && detail !== listBook) update(detail)
  }

  function updateBook(bookId: string, updates: Partial<BookData>): void {
    eachBookProjection(bookId, book => Object.assign(book, updates))
  }

  function deleteBook(bookId: string): void {
    const index = books.value.findIndex(book => book.id === bookId)
    if (index >= 0) {
      books.value.splice(index, 1)
      if (currentBookId.value === bookId) {
        currentBookId.value = null
        currentBookDetail.value = null
      }
      if (selectedBookIds.value.delete(bookId)) {
        selectedBookIds.value = new Set(selectedBookIds.value)
      }
    }
  }

  function deleteBooks(bookIds: string[]): void {
    const ids = new Set(bookIds)
    books.value = books.value.filter(book => !ids.has(book.id))
    selectedBookIds.value = new Set([...selectedBookIds.value].filter(bookId => !ids.has(bookId)))
    if (currentBookId.value && ids.has(currentBookId.value)) {
      currentBookId.value = null
      currentBookDetail.value = null
    }
  }

  function getBookById(bookId: string): BookData | null {
    return books.value.find(book => book.id === bookId) || null
  }

  function addChapter(bookId: string, chapter: ChapterData): void {
    eachBookProjection(bookId, (book) => {
      book.chapters ??= []
      book.chapters.push(chapter)
      book.chapterCount = book.chapters.length
    })
  }

  function updateChapter(bookId: string, chapterId: string, updates: Partial<ChapterData>): void {
    eachBookProjection(bookId, (book) => {
      const index = book.chapters?.findIndex(item => item.id === chapterId) ?? -1
      if (book.chapters && index >= 0) {
        const chapter = book.chapters[index]
        if (chapter) {
          book.chapters[index] = { ...chapter, ...updates }
        }
      }
    })
  }

  function deleteChapter(bookId: string, chapterId: string): void {
    eachBookProjection(bookId, (book) => {
      const index = book.chapters?.findIndex(chapter => chapter.id === chapterId) ?? -1
      if (book.chapters && index >= 0) {
        book.chapters.splice(index, 1)
        book.chapterCount = book.chapters.length
      }
    })
  }

  function reorderChapters(bookId: string, chapterIds: string[]): void {
    eachBookProjection(bookId, (book) => {
      if (!book.chapters) return
      const reordered: ChapterData[] = []
      const orderedIds = new Set<string>()
      for (const chapterId of chapterIds) {
        if (orderedIds.has(chapterId)) continue
        const chapter = book.chapters.find(item => item.id === chapterId)
        if (chapter) {
          orderedIds.add(chapterId)
          reordered.push(chapter)
        }
      }
      for (const chapter of book.chapters) {
        if (!orderedIds.has(chapter.id)) reordered.push(chapter)
      }
      for (let index = 0; index < reordered.length; index += 1) {
        const chapter = reordered[index]
        if (chapter) chapter.order = index
      }
      book.chapters = reordered
    })
  }

  function setTags(tagList: TagData[]): void {
    tags.value = tagList
  }

  function addTag(tag: TagData): void {
    tags.value.push(tag)
  }

  function deleteTag(tagName: string): void {
    const index = tags.value.findIndex(tag => tag.name === tagName)
    if (index >= 0) {
      tags.value.splice(index, 1)
      const selectedIndex = selectedTagNames.value.indexOf(tagName)
      if (selectedIndex >= 0) {
        selectedTagNames.value.splice(selectedIndex, 1)
      }
    }
  }

  function toggleTagFilter(tagName: string): void {
    const index = selectedTagNames.value.indexOf(tagName)
    if (index >= 0) {
      selectedTagNames.value.splice(index, 1)
    } else {
      selectedTagNames.value.push(tagName)
    }
    void loadBooks()
  }

  function setSort(by: BookSortBy, order: SortOrder = 'desc'): void {
    sortBy.value = by
    sortOrder.value = order
    void loadBooks()
  }

  function enterBatchMode(): void {
    batchMode.value = true
  }

  function exitBatchMode(): void {
    batchMode.value = false
    selectedBookIds.value = new Set()
  }

  function toggleBookSelection(bookId: string): void {
    const next = new Set(selectedBookIds.value)
    if (next.has(bookId)) {
      next.delete(bookId)
    } else {
      next.add(bookId)
    }
    selectedBookIds.value = next
  }

  function toggleSelectAll(): void {
    if (isAllSelected.value) {
      selectedBookIds.value = new Set()
      return
    }

    selectedBookIds.value = new Set(books.value.map(book => book.id))
  }

  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  function setError(message: string | null): void {
    error.value = message
  }

  function setCurrentBook(bookId: string | null): void {
    currentBookId.value = bookId
    currentBookDetail.value = bookId
      ? books.value.find(book => book.id === bookId) || null
      : null
  }

  function setSearchQuery(query: string): void {
    searchKeyword.value = query
    void loadBooks()
  }

  async function loadBooks(): Promise<void> {
    setLoading(true)
    setError(null)
    try {
      const params: bookshelfApi.GetBooksParams = {}

      if (searchKeyword.value.trim()) {
        params.search = searchKeyword.value.trim()
      }
      if (selectedTagNames.value.length > 0) {
        params.tags = selectedTagNames.value
      }
      params.sortBy = sortBy.value
      params.sortOrder = sortOrder.value

      setBooks(await bookshelfApi.getBooks(params))
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载书籍失败')
    } finally {
      setLoading(false)
    }
  }

  async function loadBookDetail(bookId: string): Promise<BookData | null> {
    try {
      const book = await bookshelfApi.getBookDetail(bookId)
      upsertBook(book)
      return book
    } catch {
      return null
    }
  }

  async function loadTags(): Promise<void> {
    try {
      setTags(await bookshelfApi.getTags())
    } catch {
      return
    }
  }

  async function createBook(
    title: string,
    cover?: File,
    tags?: string[],
  ): Promise<BookData | null> {
    try {
      const book = await bookshelfApi.createBook(title, cover, tags)
      await Promise.all([
        loadBooks(),
        tags !== undefined ? loadTags() : Promise.resolve(),
      ])
      return book
    } catch {
      return null
    }
  }

  async function updateBookApi(bookId: string, data: BookUpdatePayload): Promise<boolean> {
    try {
      const book = await bookshelfApi.updateBook(bookId, data)
      updateBook(bookId, book)
      await Promise.all([
        loadBooks(),
        data.tags !== undefined ? loadTags() : Promise.resolve(),
      ])
      return true
    } catch {
      return false
    }
  }

  async function deleteBookApi(bookId: string): Promise<boolean> {
    await bookshelfApi.deleteBook(bookId)
    deleteBook(bookId)
    return true
  }

  async function batchDeleteBooksApi(
    bookIds: string[],
  ): Promise<bookshelfApi.BookBatchDeleteResult> {
    const result = await bookshelfApi.batchDeleteBooks(bookIds)
    deleteBooks(result.deleted)
    return result
  }

  async function batchUpdateTagsApi(
    bookIds: string[],
    tagNames: string[],
    action: 'add' | 'remove',
  ): Promise<number> {
    const result = await bookshelfApi.batchUpdateBookTags(bookIds, tagNames, action)
    await Promise.all([loadBooks(), loadTags()])
    return result.updated
  }

  async function createTag(name: string, color?: string): Promise<TagData | null> {
    try {
      const tag = await bookshelfApi.createTag(name, color)
      addTag(tag)
      return tag
    } catch {
      return null
    }
  }

  async function deleteTagApi(tagName: string): Promise<boolean> {
    try {
      await bookshelfApi.deleteTag(tagName)
      deleteTag(tagName)
      await loadBooks()
      return true
    } catch {
      return false
    }
  }

  async function updateTagApi(
    currentName: string,
    name: string,
    color: string,
  ): Promise<boolean> {
    try {
      await bookshelfApi.updateTag(currentName, name, color)
      await loadTags()
      await loadBooks()
      return true
    } catch {
      return false
    }
  }

  async function createChapterApi(bookId: string, title: string): Promise<ChapterData | null> {
    const chapter = await bookshelfApi.createChapter(bookId, title)
    addChapter(bookId, chapter)
    return chapter
  }

  async function updateChapterApi(
    bookId: string,
    chapterId: string,
    title: string,
  ): Promise<boolean> {
    await bookshelfApi.updateChapter(chapterId, title)
    updateChapter(bookId, chapterId, { title })
    return true
  }

  async function deleteChapterApi(bookId: string, chapterId: string): Promise<boolean> {
    await bookshelfApi.deleteChapter(chapterId)
    deleteChapter(bookId, chapterId)
    return true
  }

  async function reorderChaptersApi(bookId: string, chapterIds: string[]): Promise<boolean> {
    try {
      await bookshelfApi.reorderChapters(bookId, chapterIds)
      reorderChapters(bookId, chapterIds)
      return true
    } catch {
      return false
    }
  }

  function reset(): void {
    books.value = []
    tags.value = []
    searchKeyword.value = ''
    selectedTagNames.value = []
    batchMode.value = false
    selectedBookIds.value = new Set()
    sortBy.value = 'updatedAt'
    sortOrder.value = 'desc'
    currentBookId.value = null
    currentBookDetail.value = null
    isLoading.value = false
    error.value = null
  }

  return {
    books,
    tags,
    selectedTagNames,
    sortBy,
    sortOrder,
    currentBookId,
    batchMode,
    selectedBookIds,
    isLoading,
    error,

    currentBook,
    searchQuery,
    isAllSelected,

    updateBook,
    getBookById,

    setSearchQuery,
    toggleTagFilter,
    setSort,
    enterBatchMode,
    exitBatchMode,
    toggleBookSelection,
    toggleSelectAll,

    setCurrentBook,

    loadBooks,
    loadBookDetail,
    loadTags,
    createBook,
    updateBookApi,
    deleteBookApi,
    batchDeleteBooksApi,
    batchUpdateTagsApi,
    createTag,
    deleteTagApi,
    updateTagApi,
    createChapterApi,
    updateChapterApi,
    deleteChapterApi,
    reorderChaptersApi,

    reset,
  }
})
