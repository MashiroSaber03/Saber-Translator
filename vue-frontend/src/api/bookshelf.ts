import { apiClient } from './client'
import type { ApiResponse, BookData, ChapterData, TagData } from '@/types'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

const BOOKS_ENDPOINT = '/api/bookshelf/books'
const TAGS_ENDPOINT = '/api/bookshelf/tags'

export interface BookListResponse {
  success: boolean
  books?: BookData[]
  error?: string
}

export interface BookDetailResponse {
  success: boolean
  book?: BookData
  error?: string
}

export interface GetBooksParams {
  search?: string
  tags?: string[]
}

export interface ChapterListResponse {
  success: boolean
  chapters?: ChapterData[]
  error?: string
}

export interface ChapterDetailResponse {
  success: boolean
  chapter?: ChapterData
  error?: string
}

export interface ChapterImageData {
  index: number
  original: string
  translated?: string
  fileName?: string
  relativePath?: string
}

export interface ChapterImagesResponse {
  success: boolean
  images?: ChapterImageData[]
  error?: string
}

export interface TagListResponse {
  success: boolean
  tags?: TagData[]
  error?: string
}

export interface TagDetailResponse {
  success: boolean
  tag?: TagData
  error?: string
}

function bookshelfPathSegment(value: string): string {
  return encodeURIComponent(value)
}

function bookPath(bookId: string): string {
  return `${BOOKS_ENDPOINT}/${bookshelfPathSegment(bookId)}`
}

function chapterPath(bookId: string, chapterId?: string): string {
  const basePath = `${bookPath(bookId)}/chapters`
  return chapterId ? `${basePath}/${bookshelfPathSegment(chapterId)}` : basePath
}

function tagPath(tagName: string): string {
  return `${TAGS_ENDPOINT}/${encodeURIComponent(tagName)}`
}

function buildBookListUrl(params?: GetBooksParams): string {
  const queryParams = new URLSearchParams()
  if (params?.search) {
    queryParams.append('search', params.search)
  }
  if (params?.tags && params.tags.length > 0) {
    queryParams.append('tags', params.tags.join(','))
  }
  const queryString = queryParams.toString()
  return queryString ? `${BOOKS_ENDPOINT}?${queryString}` : BOOKS_ENDPOINT
}

export async function getBooks(params?: GetBooksParams): Promise<BookListResponse> {
  return apiClient.get<BookListResponse>(buildBookListUrl(params))
}

export async function getBookDetail(bookId: string): Promise<BookDetailResponse> {
  return apiClient.get<BookDetailResponse>(bookPath(bookId))
}

export async function createBook(
  title: string,
  description?: string,
  cover?: string,
  tags?: string[],
  translation_constraints?: BookTranslationConstraints
): Promise<BookDetailResponse> {
  return apiClient.post<BookDetailResponse>(BOOKS_ENDPOINT, {
    title,
    description,
    cover,
    tags,
    translation_constraints,
  })
}

export async function updateBook(
  bookId: string,
  data: {
    title?: string
    description?: string
    cover?: string
    tags?: string[]
    translation_constraints?: BookTranslationConstraints
  }
): Promise<BookDetailResponse> {
  return apiClient.put<BookDetailResponse>(bookPath(bookId), data)
}

export async function deleteBook(bookId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(bookPath(bookId))
}

export async function getChapters(bookId: string): Promise<ChapterListResponse> {
  return apiClient.get<ChapterListResponse>(chapterPath(bookId))
}

export async function createChapter(bookId: string, title: string): Promise<ChapterDetailResponse> {
  return apiClient.post<ChapterDetailResponse>(chapterPath(bookId), { title })
}

export async function updateChapter(
  bookId: string,
  chapterId: string,
  title: string
): Promise<ChapterDetailResponse> {
  return apiClient.put<ChapterDetailResponse>(chapterPath(bookId, chapterId), { title })
}

export async function deleteChapter(bookId: string, chapterId: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(chapterPath(bookId, chapterId))
}

export async function reorderChapters(bookId: string, chapterIds: string[]): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`${chapterPath(bookId)}/reorder`, {
    chapter_ids: chapterIds,
  })
}

export async function getChapterImages(
  bookId: string,
  chapterId: string
): Promise<ChapterImagesResponse> {
  return apiClient.get<ChapterImagesResponse>(`${chapterPath(bookId, chapterId)}/images`)
}

export async function getTags(): Promise<TagListResponse> {
  return apiClient.get<TagListResponse>(TAGS_ENDPOINT)
}

export async function createTag(name: string, color?: string): Promise<TagDetailResponse> {
  return apiClient.post<TagDetailResponse>(TAGS_ENDPOINT, { name, color })
}

export async function deleteTag(tagName: string): Promise<ApiResponse> {
  return apiClient.delete<ApiResponse>(tagPath(tagName))
}

export async function updateTag(
  currentName: string,
  name: string,
  color: string
): Promise<TagDetailResponse> {
  return apiClient.put<TagDetailResponse>(tagPath(currentName), { name, color })
}
