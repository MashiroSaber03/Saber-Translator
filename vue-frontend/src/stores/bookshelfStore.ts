/**
 * 书架状态管理 Store
 * 管理书籍列表、搜索、标签筛选、批量操作
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { BookData, ChapterData, TagData } from '@/types/api'
import * as bookshelfApi from '@/api/bookshelf'

// ============================================================
// 类型定义
// ============================================================

/**
 * 书籍排序方式
 */
export type BookSortBy = 'title' | 'createdAt' | 'updatedAt'

/**
 * 排序方向
 */
export type SortOrder = 'asc' | 'desc'

/**
 * 批量操作类型
 */
export type BatchOperation = 'delete' | 'addTags' | 'removeTags'

// ============================================================
// Store 定义
// ============================================================

export const useBookshelfStore = defineStore('bookshelf', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 书籍列表 */
  const books = ref<BookData[]>([])

  /** 标签列表 */
  const tags = ref<TagData[]>([])

  /** 搜索关键词 */
  const searchKeyword = ref('')

  /** 选中的标签ID列表（用于筛选） */
  const selectedTagIds = ref<string[]>([])

  /** 排序方式 */
  const sortBy = ref<BookSortBy>('updatedAt')

  /** 排序方向 */
  const sortOrder = ref<SortOrder>('desc')

  /** 是否为批量操作模式 */
  const batchMode = ref(false)

  /** 批量选中的书籍ID列表 */
  const selectedBookIds = ref<Set<string>>(new Set())

  /** 当前展开的书籍ID（显示章节列表） */
  const expandedBookId = ref<string | null>(null)

  /** 当前选中的书籍ID（用于详情显示） */
  const currentBookId = ref<string | null>(null)

  /** 是否正在加载 */
  const isLoading = ref(false)

  /** 错误信息 */
  const error = ref<string | null>(null)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 过滤后的书籍列表（应用搜索和标签筛选） */
  const filteredBooks = computed(() => {
    let result = [...books.value]

    // 应用搜索过滤
    if (searchKeyword.value.trim()) {
      const keyword = searchKeyword.value.toLowerCase().trim()
      result = result.filter(book => {
        const titleMatch = book.title.toLowerCase().includes(keyword)
        const descMatch = book.description?.toLowerCase().includes(keyword) || false
        return titleMatch || descMatch
      })
    }

    // 应用标签筛选（selectedTagIds 实际存储的是标签名称）
    if (selectedTagIds.value.length > 0) {
      result = result.filter(book => {
        if (!book.tags || book.tags.length === 0) return false
        // 书籍必须包含所有选中的标签名称
        return selectedTagIds.value.every(tagName => book.tags?.includes(tagName))
      })
    }

    // 应用排序
    result.sort((a, b) => {
      let comparison = 0
      
      switch (sortBy.value) {
        case 'title':
          comparison = a.title.localeCompare(b.title)
          break
        case 'createdAt':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          break
        case 'updatedAt':
          comparison = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime()
          break
      }

      return sortOrder.value === 'asc' ? comparison : -comparison
    })

    return result
  })

  /** 书籍总数 */
  const bookCount = computed(() => books.value.length)

  /** 过滤后的书籍数量 */
  const filteredBookCount = computed(() => filteredBooks.value.length)

  /** 选中的书籍数量 */
  const selectedBookCount = computed(() => selectedBookIds.value.size)

  /** 是否全选 */
  const isAllSelected = computed(() => {
    if (filteredBooks.value.length === 0) return false
    return filteredBooks.value.every(book => selectedBookIds.value.has(book.id))
  })

  /** 当前展开的书籍 */
  const expandedBook = computed(() => {
    if (!expandedBookId.value) return null
    return books.value.find(book => book.id === expandedBookId.value) || null
  })

  /** 当前选中的书籍（用于详情显示） */
  const currentBook = computed(() => {
    if (!currentBookId.value) return null
    return books.value.find(book => book.id === currentBookId.value) || null
  })

  /** 选中的书籍集合（兼容旧API） */
  const selectedBooks = computed(() => selectedBookIds.value)

  /** 搜索查询（兼容旧API） */
  const searchQuery = computed(() => searchKeyword.value)

  // ============================================================
  // 书籍管理方法
  // ============================================================

  /**
   * 设置书籍列表
   * @param bookList - 书籍列表
   */
  function setBooks(bookList: BookData[]): void {
    books.value = bookList
    console.log(`书籍列表已设置，共 ${bookList.length} 本书`)
  }

  /**
   * 添加书籍
   * @param book - 书籍数据
   */
  function addBook(book: BookData): void {
    books.value.unshift(book)
    console.log(`已添加书籍: ${book.title}`)
  }

  /**
   * 更新书籍
   * @param bookId - 书籍ID
   * @param updates - 更新数据
   */
  function updateBook(bookId: string, updates: Partial<BookData>): void {
    const index = books.value.findIndex(b => b.id === bookId)
    if (index >= 0) {
      const book = books.value[index]
      if (book) {
        books.value[index] = { ...book, ...updates }
        console.log(`已更新书籍: ${bookId}`)
      }
    }
  }

  /**
   * 删除书籍
   * @param bookId - 书籍ID
   */
  function deleteBook(bookId: string): void {
    const index = books.value.findIndex(b => b.id === bookId)
    if (index >= 0) {
      books.value.splice(index, 1)
      selectedBookIds.value.delete(bookId)
      if (expandedBookId.value === bookId) {
        expandedBookId.value = null
      }
      console.log(`已删除书籍: ${bookId}`)
    }
  }

  /**
   * 批量删除书籍
   * @param bookIds - 书籍ID列表
   */
  function deleteBooks(bookIds: string[]): void {
    for (const bookId of bookIds) {
      deleteBook(bookId)
    }
    console.log(`已批量删除 ${bookIds.length} 本书`)
  }

  /**
   * 根据ID获取书籍
   * @param bookId - 书籍ID
   * @returns 书籍数据或 null
   */
  function getBookById(bookId: string): BookData | null {
    return books.value.find(b => b.id === bookId) || null
  }

  // ============================================================
  // 章节管理方法
  // ============================================================

  /**
   * 添加章节到书籍
   * @param bookId - 书籍ID
   * @param chapter - 章节数据
   */
  function addChapter(bookId: string, chapter: ChapterData): void {
    const book = books.value.find(b => b.id === bookId)
    if (book) {
      if (!book.chapters) {
        book.chapters = []
      }
      book.chapters.push(chapter)
      console.log(`已添加章节到书籍 ${bookId}: ${chapter.title}`)
    }
  }

  /**
   * 更新章节
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   * @param updates - 更新数据
   */
  function updateChapter(bookId: string, chapterId: string, updates: Partial<ChapterData>): void {
    const book = books.value.find(b => b.id === bookId)
    if (book && book.chapters) {
      const chapter = book.chapters.find(c => c.id === chapterId)
      if (chapter) {
        Object.assign(chapter, updates)
        console.log(`已更新章节: ${chapterId}`)
      }
    }
  }

  /**
   * 删除章节
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   */
  function deleteChapter(bookId: string, chapterId: string): void {
    const book = books.value.find(b => b.id === bookId)
    if (book && book.chapters) {
      const index = book.chapters.findIndex(c => c.id === chapterId)
      if (index >= 0) {
        book.chapters.splice(index, 1)
        console.log(`已删除章节: ${chapterId}`)
      }
    }
  }

  /**
   * 重新排序章节
   * @param bookId - 书籍ID
   * @param chapterIds - 新的章节ID顺序
   */
  function reorderChapters(bookId: string, chapterIds: string[]): void {
    const book = books.value.find(b => b.id === bookId)
    if (book && book.chapters) {
      const reordered: ChapterData[] = []
      for (let i = 0; i < chapterIds.length; i++) {
        const chapter = book.chapters.find(c => c.id === chapterIds[i])
        if (chapter) {
          chapter.order = i
          reordered.push(chapter)
        }
      }
      book.chapters = reordered
      console.log(`已重新排序章节: ${bookId}`)
    }
  }

  // ============================================================
  // 标签管理方法
  // ============================================================

  /**
   * 设置标签列表
   * @param tagList - 标签列表
   */
  function setTags(tagList: TagData[]): void {
    tags.value = tagList
    console.log(`标签列表已设置，共 ${tagList.length} 个标签`)
  }

  /**
   * 添加标签
   * @param tag - 标签数据
   */
  function addTag(tag: TagData): void {
    tags.value.push(tag)
    console.log(`已添加标签: ${tag.name}`)
  }

  /**
   * 删除标签
   * @param tagId - 标签ID
   */
  function deleteTag(tagId: string): void {
    const index = tags.value.findIndex(t => t.id === tagId)
    if (index >= 0) {
      tags.value.splice(index, 1)
      // 从选中列表中移除
      const selectedIndex = selectedTagIds.value.indexOf(tagId)
      if (selectedIndex >= 0) {
        selectedTagIds.value.splice(selectedIndex, 1)
      }
      console.log(`已删除标签: ${tagId}`)
    }
  }

  /**
   * 为书籍添加标签
   * @param bookId - 书籍ID
   * @param tagId - 标签ID
   */
  function addTagToBook(bookId: string, tagId: string): void {
    const book = books.value.find(b => b.id === bookId)
    if (book) {
      if (!book.tags) {
        book.tags = []
      }
      if (!book.tags.includes(tagId)) {
        book.tags.push(tagId)
        console.log(`已为书籍 ${bookId} 添加标签 ${tagId}`)
      }
    }
  }

  /**
   * 从书籍移除标签
   * @param bookId - 书籍ID
   * @param tagId - 标签ID
   */
  function removeTagFromBook(bookId: string, tagId: string): void {
    const book = books.value.find(b => b.id === bookId)
    if (book && book.tags) {
      const index = book.tags.indexOf(tagId)
      if (index >= 0) {
        book.tags.splice(index, 1)
        console.log(`已从书籍 ${bookId} 移除标签 ${tagId}`)
      }
    }
  }

  /**
   * 批量为书籍添加标签
   * @param bookIds - 书籍ID列表
   * @param tagIds - 标签ID列表
   */
  function batchAddTags(bookIds: string[], tagIds: string[]): void {
    for (const bookId of bookIds) {
      for (const tagId of tagIds) {
        addTagToBook(bookId, tagId)
      }
    }
    console.log(`已批量添加标签到 ${bookIds.length} 本书`)
  }

  /**
   * 批量从书籍移除标签
   * @param bookIds - 书籍ID列表
   * @param tagIds - 标签ID列表
   */
  function batchRemoveTags(bookIds: string[], tagIds: string[]): void {
    for (const bookId of bookIds) {
      for (const tagId of tagIds) {
        removeTagFromBook(bookId, tagId)
      }
    }
    console.log(`已批量从 ${bookIds.length} 本书移除标签`)
  }

  // ============================================================
  // 搜索和筛选方法
  // ============================================================

  /**
   * 设置搜索关键词
   * @param keyword - 搜索关键词
   */
  function setSearchKeyword(keyword: string): void {
    searchKeyword.value = keyword
  }

  /**
   * 清除搜索关键词
   */
  function clearSearchKeyword(): void {
    searchKeyword.value = ''
  }

  /**
   * 切换标签筛选
   * @param tagId - 标签ID
   */
  function toggleTagFilter(tagId: string): void {
    const index = selectedTagIds.value.indexOf(tagId)
    if (index >= 0) {
      selectedTagIds.value.splice(index, 1)
    } else {
      selectedTagIds.value.push(tagId)
    }
  }

  /**
   * 设置标签筛选
   * @param tagIds - 标签ID列表
   */
  function setTagFilter(tagIds: string[]): void {
    selectedTagIds.value = tagIds
  }

  /**
   * 清除标签筛选
   */
  function clearTagFilter(): void {
    selectedTagIds.value = []
  }

  /**
   * 设置排序方式
   * @param by - 排序字段
   * @param order - 排序方向
   */
  function setSort(by: BookSortBy, order: SortOrder = 'desc'): void {
    sortBy.value = by
    sortOrder.value = order
  }

  // ============================================================
  // 批量操作方法
  // ============================================================

  /**
   * 进入批量操作模式
   */
  function enterBatchMode(): void {
    batchMode.value = true
    selectedBookIds.value.clear()
    console.log('已进入批量操作模式')
  }

  /**
   * 退出批量操作模式
   */
  function exitBatchMode(): void {
    batchMode.value = false
    selectedBookIds.value.clear()
    console.log('已退出批量操作模式')
  }

  /**
   * 切换书籍选中状态
   * @param bookId - 书籍ID
   */
  function toggleBookSelection(bookId: string): void {
    if (selectedBookIds.value.has(bookId)) {
      selectedBookIds.value.delete(bookId)
    } else {
      selectedBookIds.value.add(bookId)
    }
  }

  /**
   * 全选/取消全选
   */
  function toggleSelectAll(): void {
    if (isAllSelected.value) {
      selectedBookIds.value.clear()
    } else {
      for (const book of filteredBooks.value) {
        selectedBookIds.value.add(book.id)
      }
    }
  }

  /**
   * 获取选中的书籍ID列表
   * @returns 书籍ID数组
   */
  function getSelectedBookIds(): string[] {
    return Array.from(selectedBookIds.value)
  }

  // ============================================================
  // 展开/折叠方法
  // ============================================================

  /**
   * 展开书籍（显示章节列表）
   * @param bookId - 书籍ID
   */
  function expandBook(bookId: string): void {
    expandedBookId.value = bookId
  }

  /**
   * 折叠书籍
   */
  function collapseBook(): void {
    expandedBookId.value = null
  }

  /**
   * 切换书籍展开状态
   * @param bookId - 书籍ID
   */
  function toggleBookExpand(bookId: string): void {
    if (expandedBookId.value === bookId) {
      expandedBookId.value = null
    } else {
      expandedBookId.value = bookId
    }
  }

  // ============================================================
  // 加载状态管理
  // ============================================================

  /**
   * 设置加载状态
   * @param loading - 是否正在加载
   */
  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  /**
   * 设置错误信息
   * @param message - 错误信息
   */
  function setError(message: string | null): void {
    error.value = message
  }

  // ============================================================
  // 当前书籍管理
  // ============================================================

  /**
   * 设置当前书籍（用于详情显示）
   * @param bookId - 书籍ID
   */
  function setCurrentBook(bookId: string | null): void {
    currentBookId.value = bookId
  }

  /**
   * 选中书籍（添加到选中集合）
   * @param bookId - 书籍ID
   */
  function selectBook(bookId: string): void {
    selectedBookIds.value.add(bookId)
  }

  /**
   * 取消选中书籍
   * @param bookId - 书籍ID
   */
  function deselectBook(bookId: string): void {
    selectedBookIds.value.delete(bookId)
  }

  /**
   * 清除所有选中
   */
  function clearSelection(): void {
    selectedBookIds.value.clear()
  }

  /**
   * 设置搜索查询（兼容旧API）
   * @param query - 搜索查询
   */
  function setSearchQuery(query: string): void {
    searchKeyword.value = query
  }

  // ============================================================
  // API 调用方法
  // ============================================================

  /**
   * 从服务器加载书籍列表
   */
  async function loadBooks(): Promise<void> {
    setLoading(true)
    setError(null)
    try {
      const response = await bookshelfApi.getBooks()
      if (response.success && response.books) {
        setBooks(response.books)
      } else {
        setError(response.error || '加载书籍失败')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载书籍失败')
      console.error('加载书籍失败:', err)
    } finally {
      setLoading(false)
    }
  }

  /**
   * 从服务器加载标签列表
   */
  async function loadTags(): Promise<void> {
    try {
      const response = await bookshelfApi.getTags()
      if (response.success && response.tags) {
        setTags(response.tags)
      }
    } catch (err) {
      console.error('加载标签失败:', err)
    }
  }

  /**
   * 创建新书籍
   * @param title - 书籍标题
   * @param description - 书籍描述
   * @param cover - 封面图片（Base64）
   * @param tags - 标签名称数组
   */
  async function createBook(title: string, description?: string, cover?: string, tags?: string[]): Promise<BookData | null> {
    try {
      const response = await bookshelfApi.createBook(title, description, cover, tags)
      if (response.success && response.book) {
        addBook(response.book)
        return response.book
      }
      return null
    } catch (err) {
      console.error('创建书籍失败:', err)
      return null
    }
  }

  /**
   * 更新书籍（调用API）
   * @param bookId - 书籍ID
   * @param data - 更新数据
   */
  async function updateBookApi(bookId: string, data: { title?: string; description?: string; cover?: string }): Promise<boolean> {
    try {
      const response = await bookshelfApi.updateBook(bookId, data)
      if (response.success && response.book) {
        updateBook(bookId, response.book)
        return true
      }
      return false
    } catch (err) {
      console.error('更新书籍失败:', err)
      return false
    }
  }

  /**
   * 删除书籍（调用API）
   * @param bookId - 书籍ID
   */
  async function deleteBookApi(bookId: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.deleteBook(bookId)
      if (response.success) {
        deleteBook(bookId)
        return true
      }
      return false
    } catch (err) {
      console.error('删除书籍失败:', err)
      return false
    }
  }

  /**
   * 批量删除书籍
   * @param bookIds - 书籍ID列表
   */
  async function batchDeleteBooks(bookIds: string[]): Promise<boolean> {
    try {
      // 逐个删除书籍
      for (const bookId of bookIds) {
        await bookshelfApi.deleteBook(bookId)
        deleteBook(bookId)
      }
      return true
    } catch (err) {
      console.error('批量删除书籍失败:', err)
      return false
    }
  }

  /**
   * 创建标签（调用API）
   * @param name - 标签名称
   * @param color - 标签颜色
   */
  async function createTag(name: string, color?: string): Promise<TagData | null> {
    try {
      const response = await bookshelfApi.createTag(name, color)
      if (response.success && response.tag) {
        addTag(response.tag)
        return response.tag
      }
      return null
    } catch (err) {
      console.error('创建标签失败:', err)
      return null
    }
  }

  /**
   * 删除标签（调用API）
   * @param tagId - 标签ID
   */
  async function deleteTagApi(tagId: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.deleteTag(tagId)
      if (response.success) {
        deleteTag(tagId)
        return true
      }
      return false
    } catch (err) {
      console.error('删除标签失败:', err)
      return false
    }
  }

  /**
   * 创建章节（调用API）
   * @param bookId - 书籍ID
   * @param title - 章节标题
   */
  async function createChapterApi(bookId: string, title: string): Promise<ChapterData | null> {
    try {
      const response = await bookshelfApi.createChapter(bookId, title)
      if (response.success && response.chapter) {
        addChapter(bookId, response.chapter)
        return response.chapter
      }
      return null
    } catch (err) {
      console.error('创建章节失败:', err)
      return null
    }
  }

  /**
   * 更新章节（调用API）
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   * @param title - 新标题
   */
  async function updateChapterApi(bookId: string, chapterId: string, title: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.updateChapter(bookId, chapterId, title)
      if (response.success) {
        updateChapter(bookId, chapterId, { title })
        return true
      }
      return false
    } catch (err) {
      console.error('更新章节失败:', err)
      return false
    }
  }

  /**
   * 删除章节（调用API）
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   */
  async function deleteChapterApi(bookId: string, chapterId: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.deleteChapter(bookId, chapterId)
      if (response.success) {
        deleteChapter(bookId, chapterId)
        return true
      }
      return false
    } catch (err) {
      console.error('删除章节失败:', err)
      return false
    }
  }

  /**
   * 重新排序章节（调用API）
   * @param bookId - 书籍ID
   * @param chapterIds - 新的章节ID顺序
   */
  async function reorderChaptersApi(bookId: string, chapterIds: string[]): Promise<boolean> {
    try {
      const response = await bookshelfApi.reorderChapters(bookId, chapterIds)
      if (response.success) {
        reorderChapters(bookId, chapterIds)
        return true
      }
      return false
    } catch (err) {
      console.error('重新排序章节失败:', err)
      return false
    }
  }

  /**
   * 为书籍添加标签（调用API）
   * @param bookId - 书籍ID
   * @param tagIds - 标签ID数组
   */
  async function addTagsToBookApi(bookId: string, tagIds: string[]): Promise<boolean> {
    try {
      const response = await bookshelfApi.addTagsToBook(bookId, tagIds)
      if (response.success) {
        // 更新本地状态 - 需要根据 tagId 找到 tagName
        for (const tagId of tagIds) {
          const tag = tags.value.find(t => t.id === tagId)
          if (tag) {
            addTagToBook(bookId, tag.name)
          }
        }
        return true
      }
      return false
    } catch (err) {
      console.error('添加标签失败:', err)
      return false
    }
  }

  /**
   * 从书籍移除标签（调用API）
   * @param bookId - 书籍ID
   * @param tagIds - 标签ID数组
   */
  async function removeTagsFromBookApi(bookId: string, tagIds: string[]): Promise<boolean> {
    try {
      const response = await bookshelfApi.removeTagsFromBook(bookId, tagIds)
      if (response.success) {
        // 更新本地状态 - 需要根据 tagId 找到 tagName
        for (const tagId of tagIds) {
          const tag = tags.value.find(t => t.id === tagId)
          if (tag) {
            removeTagFromBook(bookId, tag.name)
          }
        }
        return true
      }
      return false
    } catch (err) {
      console.error('移除标签失败:', err)
      return false
    }
  }

  // ============================================================
  // 重置方法
  // ============================================================

  /**
   * 重置所有状态
   */
  function reset(): void {
    books.value = []
    tags.value = []
    searchKeyword.value = ''
    selectedTagIds.value = []
    sortBy.value = 'updatedAt'
    sortOrder.value = 'desc'
    batchMode.value = false
    selectedBookIds.value.clear()
    expandedBookId.value = null
    isLoading.value = false
    error.value = null
    console.log('书架状态已重置')
  }

  // ============================================================
  // 返回 Store 接口
  // ============================================================

  return {
    // 状态
    books,
    tags,
    searchKeyword,
    selectedTagIds,
    sortBy,
    sortOrder,
    batchMode,
    selectedBookIds,
    expandedBookId,
    currentBookId,
    isLoading,
    error,

    // 计算属性
    filteredBooks,
    bookCount,
    filteredBookCount,
    selectedBookCount,
    isAllSelected,
    expandedBook,
    currentBook,
    selectedBooks,
    searchQuery,

    // 书籍管理（本地）
    setBooks,
    addBook,
    updateBook,
    deleteBook,
    deleteBooks,
    getBookById,

    // 章节管理（本地）
    addChapter,
    updateChapter,
    deleteChapter,
    reorderChapters,

    // 标签管理（本地）
    setTags,
    addTag,
    deleteTag,
    addTagToBook,
    removeTagFromBook,
    batchAddTags,
    batchRemoveTags,

    // 搜索和筛选
    setSearchKeyword,
    clearSearchKeyword,
    setSearchQuery,
    toggleTagFilter,
    setTagFilter,
    clearTagFilter,
    setSort,

    // 批量操作
    enterBatchMode,
    exitBatchMode,
    toggleBookSelection,
    toggleSelectAll,
    getSelectedBookIds,
    selectBook,
    deselectBook,
    clearSelection,

    // 展开/折叠
    expandBook,
    collapseBook,
    toggleBookExpand,

    // 当前书籍
    setCurrentBook,

    // 加载状态
    setLoading,
    setError,

    // API 调用方法
    loadBooks,
    fetchBooks: loadBooks,
    loadTags,
    createBook,
    updateBookApi,
    deleteBookApi,
    batchDeleteBooks,
    createTag,
    deleteTagApi,
    createChapterApi,
    updateChapterApi,
    deleteChapterApi,
    reorderChaptersApi,
    addTagsToBookApi,
    removeTagsFromBookApi,

    // 重置
    reset
  }
})
