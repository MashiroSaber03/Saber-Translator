import { apiClient } from './client'
import type { components } from '@/api/generated/v2'
import type { BookData, ChapterData, TagData } from '@/types'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import { newIdempotencyKey } from '@/api/v2/content'

type V2Book = components['schemas']['Book']
type V2BookList = components['schemas']['BookList']
type V2Chapter = components['schemas']['Chapter']
type V2ChapterTitleResult = components['schemas']['ChapterTitleResult']
type V2ConstraintDocument = components['schemas']['TranslationConstraintDocument']
type V2Tag = components['schemas']['Tag']
type V2TagList = components['schemas']['TagList']
type V2JobBatchAccepted = components['schemas']['JobBatchAccepted']

const BOOKS_ENDPOINT = '/api/v2/books'
const CHAPTERS_ENDPOINT = '/api/v2/chapters'
const TAGS_ENDPOINT = '/api/v2/tags'
const tagIdsByName = new Map<string, string>()
const chapterOrderRevisions = new Map<string, number>()

export interface GetBooksParams {
  search?: string
  tags?: string[]
  sortBy?: 'title' | 'createdAt' | 'updatedAt'
  sortOrder?: 'asc' | 'desc'
}

export type BookBatchDeleteResult = components['schemas']['BookBatchDeleteResult']

function bookPath(bookId: string, suffix = ''): string {
  return `${BOOKS_ENDPOINT}/${encodeURIComponent(bookId)}${suffix}`
}

function chapterPath(chapterId: string): string {
  return `${CHAPTERS_ENDPOINT}/${encodeURIComponent(chapterId)}`
}

function rememberTags(
  tags: Array<{ id: string; name: string }> | undefined,
): string[] {
  return (tags || []).map((tag) => {
    tagIdsByName.set(tag.name, tag.id)
    return tag.name
  })
}

function toChapter(chapter: V2Chapter): ChapterData {
  return {
    id: chapter.id,
    title: chapter.title,
    order: Math.max(0, chapter.ordinal - 1),
    imageCount: chapter.pageCount || 0,
    jobStatusSummary: chapter.jobStatusSummary,
  }
}

function toBook(
  book: V2Book & { chapters?: V2Chapter[] },
  constraints?: BookTranslationConstraints,
): BookData {
  chapterOrderRevisions.set(book.id, book.chapterOrderRevision)
  const chapters = book.chapters?.map(toChapter)
  const result: BookData = {
    id: book.id,
    title: book.title,
    cover: book.coverAssetUrl || undefined,
    tags: rememberTags(book.tags),
    chapterCount: book.chapterCount ?? chapters?.length ?? 0,
    totalPages: book.pageCount ?? chapters?.reduce(
      (total, chapter) => total + (chapter.imageCount || 0),
      0,
    ) ?? 0,
    createdAt: book.createdAt,
    updatedAt: book.updatedAt,
    jobStatusSummary: book.jobStatusSummary,
  }
  if (chapters !== undefined) {
    result.chapters = chapters
  }
  if (constraints !== undefined) {
    result.translationConstraints = constraints
  }
  return result
}

function rememberTag(tag: V2Tag): TagData {
  tagIdsByName.set(tag.name, tag.id)
  return tag
}

async function resolveTagIds(names: string[] | undefined): Promise<string[] | undefined> {
  if (names === undefined) return undefined
  const missing = names.filter(name => !tagIdsByName.has(name))
  if (missing.length > 0) await getTags()
  return names.map((name) => {
    const id = tagIdsByName.get(name)
    if (!id) throw new Error(`标签不存在：${name}`)
    return id
  })
}

export async function getBookTranslationConstraints(
  bookId: string,
): Promise<{ constraints: BookTranslationConstraints; revision: number }> {
  const document = await apiClient.get<V2ConstraintDocument>(
    bookPath(bookId, '/translation-constraints'),
  )
  return {
    constraints: document.payload,
    revision: document.revision,
  }
}

export async function updateBookTranslationConstraints(
  bookId: string,
  constraints: BookTranslationConstraints,
  baseRevision: number,
): Promise<{ constraints: BookTranslationConstraints; revision: number }> {
  const document = await apiClient.put<V2ConstraintDocument>(
    bookPath(bookId, '/translation-constraints'),
    {
      baseRevision,
      payload: constraints,
    },
  )
  return {
    constraints: document.payload,
    revision: document.revision,
  }
}

export async function getBooks(params?: GetBooksParams): Promise<BookData[]> {
  const query = new URLSearchParams()
  if (params?.search) query.set('search', params.search)
  const tagIds = await resolveTagIds(params?.tags)
  if (tagIds?.length) query.set('tagIds', tagIds.join(','))
  const sortMap = {
    title: 'title',
    createdAt: 'created_at',
    updatedAt: 'updated_at',
  } as const
  if (params?.sortBy) query.set('sort_by', sortMap[params.sortBy])
  if (params?.sortOrder) query.set('sort_order', params.sortOrder)
  const suffix = query.size ? `?${query}` : ''
  const result = await apiClient.get<V2BookList>(
    `${BOOKS_ENDPOINT}${suffix}`,
  )
  return result.items.map(book => toBook(book))
}

export async function getBookDetail(bookId: string): Promise<BookData> {
  const [book, constraintDocument] = await Promise.all([
    apiClient.get<V2Book & { chapters: V2Chapter[] }>(bookPath(bookId)),
    getBookTranslationConstraints(bookId),
  ])
  return toBook(book, constraintDocument.constraints)
}

function createBookFormData(title: string, tagIds: string[], cover: File): FormData {
  const body = new FormData()
  body.append('title', title)
  body.append('tagIds', JSON.stringify(tagIds))
  body.append('cover', cover, cover.name)
  return body
}

function updateBookFormData(
  data: { title?: string; cover: File },
  tagIds?: string[],
): FormData {
  const body = new FormData()
  if (data.title !== undefined) body.append('title', data.title)
  if (tagIds !== undefined) body.append('tagIds', JSON.stringify(tagIds))
  body.append('cover', data.cover, data.cover.name)
  return body
}

export async function createBook(
  title: string,
  cover?: File,
  tags?: string[],
): Promise<BookData> {
  const tagIds = await resolveTagIds(tags) || []
  const created = cover
    ? await apiClient.upload<V2Book>(
        BOOKS_ENDPOINT,
        createBookFormData(title, tagIds, cover),
      )
    : await apiClient.post<V2Book>(
        BOOKS_ENDPOINT,
        { title, tagIds },
      )
  return toBook(created)
}

export async function updateBook(
  bookId: string,
  data: {
    title?: string
    cover?: File
    tags?: string[]
  },
): Promise<BookData> {
  const tagIds = await resolveTagIds(data.tags)
  const updated = data.cover
    ? await apiClient.upload<V2Book & { chapters?: V2Chapter[] }>(
        bookPath(bookId),
        updateBookFormData({ title: data.title, cover: data.cover }, tagIds),
        undefined,
        'put',
      )
    : await apiClient.put<V2Book & { chapters?: V2Chapter[] }>(
        bookPath(bookId),
        {
          ...(data.title !== undefined ? { title: data.title } : {}),
          ...(tagIds !== undefined ? { tagIds } : {}),
        },
      )
  return toBook(updated)
}

export async function deleteBook(bookId: string): Promise<void> {
  await apiClient.delete(bookPath(bookId))
  chapterOrderRevisions.delete(bookId)
}

export function batchDeleteBooks(bookIds: string[]): Promise<BookBatchDeleteResult> {
  return apiClient.post<BookBatchDeleteResult>(
    `${BOOKS_ENDPOINT}/batch-delete`,
    { bookIds },
  )
}

export async function batchUpdateBookTags(
  bookIds: string[],
  tagNames: string[],
  action: 'add' | 'remove',
): Promise<{ updated: number }> {
  const tagIds = await resolveTagIds(tagNames) || []
  return apiClient.post(
    `${BOOKS_ENDPOINT}/batch-tags`,
    { bookIds, tagIds, action },
  )
}

export async function createChapter(
  bookId: string,
  title: string,
): Promise<ChapterData> {
  const chapter = await apiClient.post<V2Chapter>(
    bookPath(bookId, '/chapters'),
    { title },
  )
  return toChapter(chapter)
}

export async function updateChapter(
  chapterId: string,
  title: string,
): Promise<{ id: string; title: string }> {
  return apiClient.put<V2ChapterTitleResult>(
    chapterPath(chapterId),
    { title },
  )
}

export function createBooksExportJob(
  bookIds: string[],
  preserveOriginalFilenames: boolean,
): Promise<V2JobBatchAccepted> {
  return apiClient.post<V2JobBatchAccepted>(
    `${BOOKS_ENDPOINT}/export-jobs`,
    { bookIds, preserveOriginalFilenames },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createChaptersExportJob(
  chapterIds: string[],
  preserveOriginalFilenames: boolean,
): Promise<V2JobBatchAccepted> {
  return apiClient.post<V2JobBatchAccepted>(
    `${CHAPTERS_ENDPOINT}/export-jobs`,
    { chapterIds, preserveOriginalFilenames },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function deleteChapter(
  chapterId: string,
): Promise<void> {
  await apiClient.delete(chapterPath(chapterId))
}

export async function reorderChapters(
  bookId: string,
  chapterIds: string[],
): Promise<void> {
  let baseRevision = chapterOrderRevisions.get(bookId)
  if (!baseRevision) {
    await getBookDetail(bookId)
    baseRevision = chapterOrderRevisions.get(bookId)
  }
  const result = await apiClient.put<{ chapterOrderRevision: number }>(
    bookPath(bookId, '/chapters/order'),
    {
      baseRevision,
      orderedIds: chapterIds,
    },
  )
  chapterOrderRevisions.set(bookId, result.chapterOrderRevision)
}

export async function getTags(): Promise<TagData[]> {
  const result = await apiClient.get<V2TagList>(TAGS_ENDPOINT)
  return result.items.map(rememberTag)
}

export async function createTag(
  name: string,
  color: string,
): Promise<TagData> {
  const tag = await apiClient.post<V2Tag>(
    TAGS_ENDPOINT,
    { name, color },
  )
  return rememberTag(tag)
}

function requireTagId(name: string): string {
  const id = tagIdsByName.get(name)
  if (!id) throw new Error(`标签不存在：${name}`)
  return id
}

export async function deleteTag(tagName: string): Promise<void> {
  const tagId = requireTagId(tagName)
  await apiClient.delete(
    `${TAGS_ENDPOINT}/${encodeURIComponent(tagId)}`,
  )
  tagIdsByName.delete(tagName)
}

export async function updateTag(
  currentName: string,
  name: string,
  color: string,
): Promise<TagData> {
  const tagId = requireTagId(currentName)
  const tag = await apiClient.put<V2Tag>(
    `${TAGS_ENDPOINT}/${encodeURIComponent(tagId)}`,
    { name, color },
  )
  tagIdsByName.delete(currentName)
  return rememberTag(tag)
}
