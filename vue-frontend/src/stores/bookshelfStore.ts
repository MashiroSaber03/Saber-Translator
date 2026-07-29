import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import * as bookshelfApi from '@/api/bookshelf'
import type { BookData, ChapterData, TagData } from '@/types/api'
import { normalizeBookData, normalizeChapterData } from '@/utils/bookshelfModels'

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
  const expandedBookId = ref<string | null>(null)
  const currentBookId = ref<string | null>(null)
  const batchMode = ref(false)
  const selectedBookIds = ref<Set<string>>(new Set())
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const filteredBooks = computed(() => books.value)
  const bookCount = computed(() => books.value.length)
  const filteredBookCount = computed(() => filteredBooks.value.length)
  const expandedBook = computed(() => {
    if (!expandedBookId.value) return null
    return books.value.find(book => book.id === expandedBookId.value) || null
  })
  const currentBook = computed(() => {
    if (!currentBookId.value) return null
    return books.value.find(book => book.id === currentBookId.value) || null
  })
  const searchQuery = computed(() => searchKeyword.value)
  const isAllSelected = computed(
    () => books.value.length > 0 && books.value.every(book => selectedBookIds.value.has(book.id)),
  )

  function setBooks(bookList: BookData[]): void {
    books.value = bookList.map(normalizeBookData)
  }

  function addBook(book: BookData): void {
    books.value.unshift(normalizeBookData(book))
  }

  function upsertBook(book: BookData): void {
    const normalizedBook = normalizeBookData(book)
    const index = books.value.findIndex(item => item.id === normalizedBook.id)
    if (index >= 0) {
      books.value[index] = normalizedBook
      return
    }
    books.value.unshift(normalizedBook)
  }

  function updateBook(bookId: string, updates: Partial<BookData>): void {
    const index = books.value.findIndex(book => book.id === bookId)
    if (index >= 0) {
      const book = books.value[index]
      if (book) {
        books.value[index] = normalizeBookData({ ...book, ...updates })
      }
    }
  }

  function deleteBook(bookId: string): void {
    const index = books.value.findIndex(book => book.id === bookId)
    if (index >= 0) {
      books.value.splice(index, 1)
      if (expandedBookId.value === bookId) {
        expandedBookId.value = null
      }
      if (currentBookId.value === bookId) {
        currentBookId.value = null
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
    if (expandedBookId.value && ids.has(expandedBookId.value)) {
      expandedBookId.value = null
    }
    if (currentBookId.value && ids.has(currentBookId.value)) {
      currentBookId.value = null
    }
  }

  function getBookById(bookId: string): BookData | null {
    return books.value.find(book => book.id === bookId) || null
  }

  function addChapter(bookId: string, chapter: ChapterData): void {
    const book = books.value.find(item => item.id === bookId)
    if (book) {
      book.chapters ??= []
      book.chapters.push(normalizeChapterData(chapter))
      book.chapterCount = book.chapters.length
    }
  }

  function updateChapter(bookId: string, chapterId: string, updates: Partial<ChapterData>): void {
    const book = books.value.find(item => item.id === bookId)
    const index = book?.chapters?.findIndex(item => item.id === chapterId) ?? -1
    if (book?.chapters && index >= 0) {
      const chapter = book.chapters[index]
      if (chapter) {
        book.chapters[index] = normalizeChapterData({ ...chapter, ...updates })
      }
    }
  }

  function deleteChapter(bookId: string, chapterId: string): void {
    const book = books.value.find(item => item.id === bookId)
    const index = book?.chapters?.findIndex(chapter => chapter.id === chapterId) ?? -1
    if (book?.chapters && index >= 0) {
      book.chapters.splice(index, 1)
      book.chapterCount = book.chapters.length
    }
  }

  function reorderChapters(bookId: string, chapterIds: string[]): void {
    const book = books.value.find(item => item.id === bookId)
    if (!book?.chapters) {
      return
    }

    const reordered: ChapterData[] = []
    const orderedIds = new Set<string>()
    for (const chapterId of chapterIds) {
      if (orderedIds.has(chapterId)) {
        continue
      }
      const chapter = book.chapters.find(item => item.id === chapterId)
      if (chapter) {
        orderedIds.add(chapterId)
        reordered.push(chapter)
      }
    }

    for (const chapter of book.chapters) {
      if (!orderedIds.has(chapter.id)) {
        reordered.push(chapter)
      }
    }

    for (let index = 0; index < reordered.length; index += 1) {
      const chapter = reordered[index]
      if (chapter) {
        chapter.order = index
      }
    }
    book.chapters = reordered
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

  function addTagToBook(bookId: string, tagName: string): void {
    const book = books.value.find(item => item.id === bookId)
    if (!book) return

    const currentTags = book.tags ?? []
    if (!currentTags.includes(tagName)) {
      book.tags = [...currentTags, tagName]
    }
  }

  function removeTagFromBook(bookId: string, tagName: string): void {
    const book = books.value.find(item => item.id === bookId)
    if (!book?.tags) return

    book.tags = book.tags.filter(item => item !== tagName)
  }

  function batchAddTags(bookIds: string[], tagNames: string[]): void {
    for (const bookId of bookIds) {
      for (const tagName of tagNames) {
        addTagToBook(bookId, tagName)
      }
    }
  }

  function batchRemoveTags(bookIds: string[], tagNames: string[]): void {
    for (const bookId of bookIds) {
      for (const tagName of tagNames) {
        removeTagFromBook(bookId, tagName)
      }
    }
  }

  function setSearchKeyword(keyword: string): void {
    searchKeyword.value = keyword
  }

  function clearSearchKeyword(): void {
    searchKeyword.value = ''
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

  function setTagFilter(tagNames: string[]): void {
    selectedTagNames.value = tagNames
  }

  function clearTagFilter(): void {
    selectedTagNames.value = []
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

  function expandBook(bookId: string): void {
    expandedBookId.value = bookId
  }

  function collapseBook(): void {
    expandedBookId.value = null
  }

  function toggleBookExpand(bookId: string): void {
    expandedBookId.value = expandedBookId.value === bookId ? null : bookId
  }

  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  function setError(message: string | null): void {
    error.value = message
  }

  function setCurrentBook(bookId: string | null): void {
    currentBookId.value = bookId
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

      const response = await bookshelfApi.getBooks(params)
      if (response.success && response.books) {
        setBooks(response.books)
      } else {
        setError(response.error || '加载书籍失败')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载书籍失败')
    } finally {
      setLoading(false)
    }
  }

  async function loadBookDetail(bookId: string): Promise<BookData | null> {
    try {
      const response = await bookshelfApi.getBookDetail(bookId)
      if (response.success && response.book) {
        upsertBook(response.book)
        return response.book
      }
      return null
    } catch {
      return null
    }
  }

  async function loadTags(): Promise<void> {
    try {
      const response = await bookshelfApi.getTags()
      if (response.success && response.tags) {
        setTags(response.tags)
      }
    } catch {
      return
    }
  }

  async function createBook(
    title: string,
    description?: string,
    cover?: File,
    tags?: string[],
  ): Promise<BookData | null> {
    try {
      const response = await bookshelfApi.createBook(title, description, cover, tags)
      if (response.success && response.book) {
        addBook(response.book)
        return response.book
      }
      return null
    } catch {
      return null
    }
  }

  async function updateBookApi(bookId: string, data: BookUpdatePayload): Promise<boolean> {
    try {
      const response = await bookshelfApi.updateBook(bookId, data)
      if (response.success && response.book) {
        updateBook(bookId, response.book)
        return true
      }
      return false
    } catch {
      return false
    }
  }

  async function deleteBookApi(bookId: string): Promise<boolean> {
    const response = await bookshelfApi.deleteBook(bookId)
    if (response.success) {
      deleteBook(bookId)
      return true
    }
    return false
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
    if (action === 'add') batchAddTags(bookIds, tagNames)
    else batchRemoveTags(bookIds, tagNames)
    await loadBooks()
    return result.updated
  }

  async function createTag(name: string, color?: string): Promise<TagData | null> {
    try {
      const response = await bookshelfApi.createTag(name, color)
      if (response.success && response.tag) {
        addTag(response.tag)
        return response.tag
      }
      return null
    } catch {
      return null
    }
  }

  async function deleteTagApi(tagName: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.deleteTag(tagName)
      if (response.success) {
        deleteTag(tagName)
        return true
      }
      return false
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
      const response = await bookshelfApi.updateTag(currentName, name, color)
      if (response.success) {
        await loadTags()
        await loadBooks()
        return true
      }
      return false
    } catch {
      return false
    }
  }

  async function createChapterApi(bookId: string, title: string): Promise<ChapterData | null> {
    try {
      const response = await bookshelfApi.createChapter(bookId, title)
      if (response.success && response.chapter) {
        addChapter(bookId, response.chapter)
        return response.chapter
      }
      return null
    } catch {
      return null
    }
  }

  async function updateChapterApi(
    bookId: string,
    chapterId: string,
    title: string,
  ): Promise<boolean> {
    try {
      const response = await bookshelfApi.updateChapter(bookId, chapterId, title)
      if (response.success) {
        updateChapter(bookId, chapterId, { title })
        return true
      }
      return false
    } catch {
      return false
    }
  }

  async function deleteChapterApi(bookId: string, chapterId: string): Promise<boolean> {
    const response = await bookshelfApi.deleteChapter(bookId, chapterId)
    if (response.success) {
      deleteChapter(bookId, chapterId)
      return true
    }
    return false
  }

  async function reorderChaptersApi(bookId: string, chapterIds: string[]): Promise<boolean> {
    try {
      const response = await bookshelfApi.reorderChapters(bookId, chapterIds)
      if (response.success) {
        reorderChapters(bookId, chapterIds)
        return true
      }
      return false
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
    expandedBookId.value = null
    currentBookId.value = null
    isLoading.value = false
    error.value = null
  }

  return {
    books,
    tags,
    searchKeyword,
    selectedTagNames,
    sortBy,
    sortOrder,
    expandedBookId,
    currentBookId,
    batchMode,
    selectedBookIds,
    isLoading,
    error,

    filteredBooks,
    bookCount,
    filteredBookCount,
    expandedBook,
    currentBook,
    searchQuery,
    isAllSelected,

    setBooks,
    addBook,
    updateBook,
    deleteBook,
    deleteBooks,
    getBookById,

    addChapter,
    updateChapter,
    deleteChapter,
    reorderChapters,

    setTags,
    addTag,
    deleteTag,
    addTagToBook,
    removeTagFromBook,
    batchAddTags,
    batchRemoveTags,

    setSearchKeyword,
    clearSearchKeyword,
    setSearchQuery,
    toggleTagFilter,
    setTagFilter,
    clearTagFilter,
    setSort,
    enterBatchMode,
    exitBatchMode,
    toggleBookSelection,
    toggleSelectAll,

    expandBook,
    collapseBook,
    toggleBookExpand,

    setCurrentBook,

    setLoading,
    setError,

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
