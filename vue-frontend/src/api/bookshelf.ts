/**
 * 书架 API
 * 包含书籍 CRUD、章节管理、标签管理等功能
 */

import { apiClient } from './client'
import type { ApiResponse, BookData, ChapterData, TagData } from '@/types'

// ==================== 书籍 API ====================

/**
 * 书籍列表响应
 */
export interface BookListResponse {
  success: boolean
  books?: BookData[]
  error?: string
}

/**
 * 书籍详情响应
 */
export interface BookDetailResponse {
  success: boolean
  book?: BookData
  error?: string
}

/**
 * 获取书籍参数
 */
export interface GetBooksParams {
  /** 搜索关键词 */
  search?: string
  /** 标签名称数组（逗号分隔） */
  tags?: string[]
}

/**
 * 获取书籍列表
 * @param params 可选的搜索和标签筛选参数
 */
export async function getBooks(params?: GetBooksParams): Promise<BookListResponse> {
  // 构建查询参数，与原版 bookshelf.js 保持一致
  const queryParams = new URLSearchParams()

  if (params?.search) {
    queryParams.append('search', params.search)
  }
  if (params?.tags && params.tags.length > 0) {
    queryParams.append('tags', params.tags.join(','))
  }

  const queryString = queryParams.toString()
  const url = queryString ? `/api/bookshelf/books?${queryString}` : '/api/bookshelf/books'

  return apiClient.get<BookListResponse>(url)
}

/**
 * 获取书籍详情
 * @param bookId 书籍 ID
 */
export async function getBookDetail(bookId: string): Promise<BookDetailResponse> {
  return apiClient.get<BookDetailResponse>(`/api/bookshelf/books/${bookId}`)
}

/**
 * 创建书籍
 * @param title 书籍标题
 * @param description 书籍描述
 * @param cover 封面图片（Base64）
 * @param tags 标签名称数组
 */
export async function createBook(
  title: string,
  description?: string,
  cover?: string,
  tags?: string[]
): Promise<BookDetailResponse> {
  return apiClient.post<BookDetailResponse>('/api/bookshelf/books', {
    title,
    description,
    cover,
    tags,
  })
}

/**
 * 更新书籍
 * @param bookId 书籍 ID
 * @param data 更新数据
 */
export async function updateBook(
  bookId: string,
  data: {
    title?: string
    description?: string
    cover?: string
  }
): Promise<BookDetailResponse> {
  return apiClient.put<BookDetailResponse>(`/api/bookshelf/books/${bookId}`, data)
}

/**
 * 删除书籍
 * @param bookId 书籍 ID
 */
export async function deleteBook(bookId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/bookshelf/books/${bookId}`)
}

// ==================== 章节 API ====================

/**
 * 章节列表响应
 */
export interface ChapterListResponse {
  success: boolean
  chapters?: ChapterData[]
  error?: string
}

/**
 * 章节详情响应
 */
export interface ChapterDetailResponse {
  success: boolean
  chapter?: ChapterData
  error?: string
}

/**
 * 章节图片数据
 */
export interface ChapterImageData {
  /** 原图 URL 或 Base64 */
  original: string
  /** 翻译后图片 URL 或 Base64 */
  translated?: string
}

/**
 * 章节图片响应
 */
export interface ChapterImagesResponse {
  success: boolean
  images?: ChapterImageData[]
  error?: string
}

/**
 * 获取书籍章节列表
 * @param bookId 书籍 ID
 */
export async function getChapters(bookId: string): Promise<ChapterListResponse> {
  return apiClient.get<ChapterListResponse>(`/api/bookshelf/books/${bookId}/chapters`)
}

/**
 * 创建章节
 * @param bookId 书籍 ID
 * @param title 章节标题
 */
export async function createChapter(
  bookId: string,
  title: string
): Promise<ChapterDetailResponse> {
  return apiClient.post<ChapterDetailResponse>(`/api/bookshelf/books/${bookId}/chapters`, {
    title,
  })
}

/**
 * 更新章节
 * @param bookId 书籍 ID
 * @param chapterId 章节 ID
 * @param title 新标题
 */
export async function updateChapter(
  bookId: string,
  chapterId: string,
  title: string
): Promise<ChapterDetailResponse> {
  return apiClient.put<ChapterDetailResponse>(
    `/api/bookshelf/books/${bookId}/chapters/${chapterId}`,
    { title }
  )
}

/**
 * 删除章节
 * @param bookId 书籍 ID
 * @param chapterId 章节 ID
 */
export async function deleteChapter(bookId: string, chapterId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/bookshelf/books/${bookId}/chapters/${chapterId}`)
}

/**
 * 重新排序章节
 * @param bookId 书籍 ID
 * @param chapterIds 章节 ID 数组（按新顺序排列）
 */
export async function reorderChapters(
  bookId: string,
  chapterIds: string[]
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/bookshelf/books/${bookId}/chapters/reorder`, {
    chapter_ids: chapterIds,
  })
}

/**
 * 获取章节图片
 * @param bookId 书籍 ID
 * @param chapterId 章节 ID
 */
export async function getChapterImages(
  bookId: string,
  chapterId: string
): Promise<ChapterImagesResponse> {
  return apiClient.get<ChapterImagesResponse>(
    `/api/bookshelf/books/${bookId}/chapters/${chapterId}/images`
  )
}

// ==================== 标签 API ====================

/**
 * 标签列表响应
 */
export interface TagListResponse {
  success: boolean
  tags?: TagData[]
  error?: string
}

/**
 * 标签详情响应
 */
export interface TagDetailResponse {
  success: boolean
  tag?: TagData
  error?: string
}

/**
 * 获取所有标签
 */
export async function getTags(): Promise<TagListResponse> {
  return apiClient.get<TagListResponse>('/api/bookshelf/tags')
}

/**
 * 创建标签
 * @param name 标签名称
 * @param color 标签颜色
 */
export async function createTag(name: string, color?: string): Promise<TagDetailResponse> {
  return apiClient.post<TagDetailResponse>('/api/bookshelf/tags', { name, color })
}

/**
 * 删除标签
 * @param tagId 标签 ID（标签名称）
 */
export async function deleteTag(tagId: string): Promise<ApiResponse> {
  // 与原版 bookshelf.js 一致，使用 encodeURIComponent 编码名称
  return apiClient.delete<ApiResponse>(`/api/bookshelf/tags/${encodeURIComponent(tagId)}`)
}

/**
 * 更新标签
 * 【复刻原版 bookshelf.js editTag】
 * @param tagId 标签 ID（原标签名称）
 * @param name 新标签名称
 * @param color 新标签颜色
 */
export async function updateTag(
  tagId: string,
  name: string,
  color: string
): Promise<TagDetailResponse> {
  // 原版 API 使用标签名称作为 URL 路径参数
  return apiClient.put<TagDetailResponse>(
    `/api/bookshelf/tags/${encodeURIComponent(tagId)}`,
    { name, color }
  )
}

/**
 * 为书籍添加标签
 * @param bookId 书籍 ID
 * @param tagIds 标签 ID 数组
 */
export async function addTagsToBook(bookId: string, tagIds: string[]): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/bookshelf/books/${bookId}/tags`, {
    tag_ids: tagIds,
  })
}

/**
 * 从书籍移除标签
 * @param bookId 书籍 ID
 * @param tagIds 标签 ID 数组
 */
export async function removeTagsFromBook(
  bookId: string,
  tagIds: string[]
): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(`/api/bookshelf/books/${bookId}/tags`, {
    data: { tag_ids: tagIds },
  })
}

/**
 * 批量为书籍添加标签
 * @param bookIds 书籍 ID 数组
 * @param tagNames 标签名称数组
 */
export async function batchAddTags(bookIds: string[], tagNames: string[]): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/bookshelf/books/batch/add-tags', {
    book_ids: bookIds,
    tags: tagNames,
  })
}

/**
 * 批量从书籍移除标签
 * @param bookIds 书籍 ID 数组
 * @param tagNames 标签名称数组
 */
export async function batchRemoveTags(
  bookIds: string[],
  tagNames: string[]
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/bookshelf/books/batch/remove-tags', {
    book_ids: bookIds,
    tags: tagNames,
  })
}

/**
 * 批量删除响应
 */
export interface BatchDeleteResponse extends ApiResponse {
  /** 成功删除的数量 */
  success_count?: number
}

/**
 * 批量删除书籍
 * 【复刻原版 bookshelf.js batchDeleteBooks】使用单次请求删除多本书籍
 * @param bookIds 书籍 ID 数组
 */
export async function batchDeleteBooks(bookIds: string[]): Promise<BatchDeleteResponse> {
  return apiClient.post<BatchDeleteResponse>('/api/bookshelf/books/batch/delete', {
    book_ids: bookIds,
  })
}

