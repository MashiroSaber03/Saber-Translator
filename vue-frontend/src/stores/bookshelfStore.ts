/**
 * 书架状态管理 Store
 * 管理书籍列表、搜索和标签筛选
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

  /** 【复刻原版】选中的标签名称列表(用于筛选) - 实际存储 name 而非 id */
  const selectedTagIds = ref<string[]>([])

  /** 排序方式 */
  const sortBy = ref<BookSortBy>('updatedAt')

  /** 排序方向 */
  const sortOrder = ref<SortOrder>('desc')

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

  /** 
   * 过滤后的书籍列表
   * 【复刻原版】搜索和标签筛选完全交给后端处理，前端直接返回后端结果
   */
  const filteredBooks = computed(() => books.value)

  /** 书籍总数 */
  const bookCount = computed(() => books.value.length)

  /** 过滤后的书籍数量 */
  const filteredBookCount = computed(() => filteredBooks.value.length)

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
      if (expandedBookId.value === bookId) {
        expandedBookId.value = null
      }
      console.log(`已删除书籍: ${bookId}`)
    }
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
   * 【复刻原版】使用 name 作为唯一标识
   * @param tagName - 标签名称(作为ID)
   */
  function deleteTag(tagName: string): void {
    const index = tags.value.findIndex(t => t.name === tagName)
    if (index >= 0) {
      tags.value.splice(index, 1)
      // 从选中列表中移除
      const selectedIndex = selectedTagIds.value.indexOf(tagName)
      if (selectedIndex >= 0) {
        selectedTagIds.value.splice(selectedIndex, 1)
      }
      console.log(`已删除标签: ${tagName}`)
    }
  }

  // 【已删除冗余函数】addTagToBook 和 removeTagFromBook
  // 原因: 现在所有标签操作都通过 updateBookApi 传递完整 tags 数组完成

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
   * 切换标签筛选并重新加载书籍
   * 与原版 bookshelf.js 的 toggleTagFilter 逻辑保持一致
   * @param tagId - 标签ID
   */
  function toggleTagFilter(tagId: string): void {
    const index = selectedTagIds.value.indexOf(tagId)
    if (index >= 0) {
      selectedTagIds.value.splice(index, 1)
    } else {
      selectedTagIds.value.push(tagId)
    }
    // 与原版保持一致：每次标签变化都从后端重新加载数据
    loadBooks()
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
   * 设置搜索查询并重新加载书籍
   * 与原版 bookshelf.js 的 handleSearch 逻辑保持一致
   * @param query - 搜索查询
   */
  function setSearchQuery(query: string): void {
    searchKeyword.value = query
    // 与原版保持一致：每次搜索变化都从后端重新加载数据
    loadBooks()
  }

  // ============================================================
  // API 调用方法
  // ============================================================

  /**
   * 从服务器加载书籍列表
   * 与原版 bookshelf.js 保持一致，将搜索和标签筛选参数传递给后端
   */
  async function loadBooks(): Promise<void> {
    setLoading(true)
    setError(null)
    try {
      // 构建请求参数，与原版逻辑保持一致
      const params: { search?: string; tags?: string[] } = {}

      if (searchKeyword.value.trim()) {
        params.search = searchKeyword.value.trim()
      }
      if (selectedTagIds.value.length > 0) {
        // selectedTagIds 实际存储的是标签名称
        params.tags = selectedTagIds.value
      }

      const response = await bookshelfApi.getBooks(params)
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
   * 更新书籍(调用API)
   * 【复刻原版 bookshelf.js saveBook】支持更新 title, cover, tags
   * @param bookId - 书籍ID
   * @param data - 更新数据
   */
  async function updateBookApi(bookId: string, data: {
    title?: string;
    description?: string;
    cover?: string;
    tags?: string[]  // 【复刻原版】支持更新 tags 数组
  }): Promise<boolean> {
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
   * 更新标签（调用API）
   * 【复刻原版 bookshelf.js editTag】
   * 更新成功后刷新标签列表和书籍列表
   * @param tagId - 标签ID（原标签名称）
   * @param name - 新标签名称
   * @param color - 新标签颜色
   */
  async function updateTagApi(tagId: string, name: string, color: string): Promise<boolean> {
    try {
      const response = await bookshelfApi.updateTag(tagId, name, color)
      if (response.success) {
        // 【复刻原版】更新成功后重新加载标签和书籍列表
        await loadTags()
        await loadBooks()
        return true
      }
      return false
    } catch (err) {
      console.error('更新标签失败:', err)
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

  // 【复刻原版】标签的增删改为通过 updateBookApi 完成,传递完整 tags 数组

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
    expandedBookId,
    currentBookId,
    isLoading,
    error,

    // 计算属性
    filteredBooks,
    bookCount,
    filteredBookCount,
    expandedBook,
    currentBook,
    searchQuery,

    // 书籍管理（本地）
    setBooks,
    addBook,
    updateBook,
    deleteBook,
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
    // 【已删除冗余】addTagToBook 和 removeTagFromBook - 现用 updateBookApi

    // 搜索和筛选
    setSearchKeyword,
    clearSearchKeyword,
    setSearchQuery,
    toggleTagFilter,
    setTagFilter,
    clearTagFilter,
    setSort,

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
    createTag,
    deleteTagApi,
    updateTagApi,
    createChapterApi,
    updateChapterApi,
    deleteChapterApi,
    reorderChaptersApi,
    // 【已删除】addTagsToBookApi 和 removeTagsFromBookApi,改用 updateBookApi 传递完整 tags 数组

    // 重置
    reset
  }
})
