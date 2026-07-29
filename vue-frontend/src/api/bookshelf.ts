import { apiClient } from './client'
import { newIdempotencyKey } from './v2/content'
import type { components } from '@/api/generated/v2'
import type { ApiResponse, BookData, ChapterData, TagData } from '@/types'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

type V2Book = components['schemas']['Book']
type V2Chapter = components['schemas']['Chapter']
type V2ConstraintDocument = components['schemas']['TranslationConstraintDocument']
type V2Tag = components['schemas']['Tag']

const BOOKS_ENDPOINT = '/api/v2/books'
const TAGS_ENDPOINT = '/api/v2/tags'
const tagIdsByName = new Map<string, string>()
const chapterOrderRevisions = new Map<string, number>()
const constraintRevisions = new Map<string, number>()

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
  sortBy?: 'title' | 'createdAt' | 'updatedAt'
  sortOrder?: 'asc' | 'desc'
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

function idempotencyConfig() {
  return {
    headers: {
      'Idempotency-Key': newIdempotencyKey(),
    },
  }
}

function bookPath(bookId: string, suffix = ''): string {
  return `${BOOKS_ENDPOINT}/${encodeURIComponent(bookId)}${suffix}`
}

function chapterPath(chapterId: string): string {
  return `/api/v2/chapters/${encodeURIComponent(chapterId)}`
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
    page_count: chapter.pageCount || 0,
    hasSession: true,
    ordinal: chapter.ordinal,
    pageOrderRevision: chapter.pageOrderRevision,
  }
}

function toBook(
  book: V2Book & { chapters?: V2Chapter[] },
  constraints?: BookTranslationConstraints,
): BookData {
  chapterOrderRevisions.set(book.id, book.chapterOrderRevision)
  const chapters = book.chapters?.map(toChapter)
  return {
    id: book.id,
    title: book.title,
    cover: book.coverAssetUrl || undefined,
    tags: rememberTags(book.tags),
    translation_constraints: constraints,
    chapters,
    chapterCount: book.chapterCount ?? chapters?.length ?? 0,
    totalPages: book.pageCount ?? chapters?.reduce(
      (total, chapter) => total + (chapter.imageCount || 0),
      0,
    ) ?? 0,
    createdAt: book.createdAt,
    updatedAt: book.updatedAt,
    chapterOrderRevision: book.chapterOrderRevision,
  }
}

function toTag(tag: V2Tag): TagData {
  tagIdsByName.set(tag.name, tag.id)
  return {
    id: tag.id,
    name: tag.name,
    color: tag.color,
    book_count: tag.bookCount || 0,
  }
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

function constraintsFromV2(document: V2ConstraintDocument): BookTranslationConstraints {
  constraintRevisions.set(document.bookId, document.revision)
  return {
    glossary: (document.payload.glossary || {}) as BookTranslationConstraints['glossary'],
    non_translate: (document.payload.nonTranslate || {}) as BookTranslationConstraints['non_translate'],
  }
}

async function getConstraints(bookId: string): Promise<BookTranslationConstraints> {
  const document = await apiClient.get<V2ConstraintDocument>(
    bookPath(bookId, '/translation-constraints'),
  )
  return constraintsFromV2(document)
}

async function saveConstraints(
  bookId: string,
  constraints: BookTranslationConstraints,
): Promise<BookTranslationConstraints> {
  let baseRevision = constraintRevisions.get(bookId)
  if (baseRevision === undefined) {
    await getConstraints(bookId)
    baseRevision = constraintRevisions.get(bookId)
  }
  const document = await apiClient.put<V2ConstraintDocument>(
    bookPath(bookId, '/translation-constraints'),
    {
      baseRevision,
      payload: {
        glossary: constraints.glossary,
        nonTranslate: constraints.non_translate,
      },
    },
    idempotencyConfig(),
  )
  return constraintsFromV2(document)
}

export async function getBooks(params?: GetBooksParams): Promise<BookListResponse> {
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
  const result = await apiClient.get<{ items: V2Book[] }>(
    `${BOOKS_ENDPOINT}${suffix}`,
  )
  return {
    success: true,
    books: result.items.map(book => toBook(book)),
  }
}

export async function getBookDetail(bookId: string): Promise<BookDetailResponse> {
  const [book, constraints] = await Promise.all([
    apiClient.get<V2Book & { chapters: V2Chapter[] }>(bookPath(bookId)),
    getConstraints(bookId),
  ])
  return { success: true, book: toBook(book, constraints) }
}

function bookFormData(
  title: string,
  tagIds: string[],
  cover?: File,
): FormData {
  const body = new FormData()
  body.append('title', title)
  body.append('tag_ids', JSON.stringify(tagIds))
  if (cover) body.append('cover', cover, cover.name)
  return body
}

export async function createBook(
  title: string,
  _description?: string,
  cover?: File,
  tags?: string[],
  translationConstraints?: BookTranslationConstraints,
): Promise<BookDetailResponse> {
  const tagIds = await resolveTagIds(tags) || []
  const created = cover
    ? await apiClient.upload<V2Book>(
        BOOKS_ENDPOINT,
        bookFormData(title, tagIds, cover),
        idempotencyConfig(),
      )
    : await apiClient.post<V2Book>(
        BOOKS_ENDPOINT,
        { title, tagIds },
        idempotencyConfig(),
      )
  if (translationConstraints) {
    await saveConstraints(created.id, translationConstraints)
  }
  return getBookDetail(created.id)
}

export async function updateBook(
  bookId: string,
  data: {
    title?: string
    description?: string
    cover?: File
    tags?: string[]
    translation_constraints?: BookTranslationConstraints
  },
): Promise<BookDetailResponse> {
  if (
    data.title !== undefined
    || data.tags !== undefined
    || data.cover !== undefined
  ) {
    const current = await apiClient.get<V2Book & { chapters: V2Chapter[] }>(
      bookPath(bookId),
    )
    const tagIds = await resolveTagIds(data.tags ?? rememberTags(current.tags)) || []
    const title = data.title ?? current.title
    if (data.cover) {
      await apiClient.upload(
        bookPath(bookId),
        bookFormData(title, tagIds, data.cover),
        idempotencyConfig(),
      )
    } else {
      await apiClient.put(
        bookPath(bookId),
        { title, tagIds },
        idempotencyConfig(),
      )
    }
  }
  if (data.translation_constraints) {
    await saveConstraints(bookId, data.translation_constraints)
  }
  return getBookDetail(bookId)
}

export async function deleteBook(bookId: string): Promise<ApiResponse> {
  await apiClient.delete(bookPath(bookId), idempotencyConfig())
  chapterOrderRevisions.delete(bookId)
  constraintRevisions.delete(bookId)
  return { success: true }
}

export async function getChapters(bookId: string): Promise<ChapterListResponse> {
  const result = await apiClient.get<{
    book: { chapterOrderRevision?: number }
    chapters: V2Chapter[]
  }>(bookPath(bookId, '/chapters'))
  if (result.book.chapterOrderRevision) {
    chapterOrderRevisions.set(bookId, result.book.chapterOrderRevision)
  }
  return {
    success: true,
    chapters: result.chapters.map(toChapter),
  }
}

export async function createChapter(
  bookId: string,
  title: string,
): Promise<ChapterDetailResponse> {
  const chapter = await apiClient.post<V2Chapter>(
    bookPath(bookId, '/chapters'),
    { title },
    idempotencyConfig(),
  )
  return { success: true, chapter: toChapter(chapter) }
}

export async function updateChapter(
  _bookId: string,
  chapterId: string,
  title: string,
): Promise<ChapterDetailResponse> {
  const chapter = await apiClient.put<V2Chapter>(
    chapterPath(chapterId),
    { title },
    idempotencyConfig(),
  )
  return { success: true, chapter: toChapter({
    ...chapter,
    ordinal: chapter.ordinal || 1,
    pageOrderRevision: chapter.pageOrderRevision || 1,
  }) }
}

export async function deleteChapter(
  _bookId: string,
  chapterId: string,
): Promise<ApiResponse> {
  await apiClient.delete(chapterPath(chapterId), idempotencyConfig())
  return { success: true }
}

export async function reorderChapters(
  bookId: string,
  chapterIds: string[],
): Promise<ApiResponse> {
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
    idempotencyConfig(),
  )
  chapterOrderRevisions.set(bookId, result.chapterOrderRevision)
  return { success: true }
}

export async function getTags(): Promise<TagListResponse> {
  const result = await apiClient.get<{ items: V2Tag[] }>(TAGS_ENDPOINT)
  return { success: true, tags: result.items.map(toTag) }
}

export async function createTag(
  name: string,
  color = '#808080',
): Promise<TagDetailResponse> {
  const tag = await apiClient.post<V2Tag>(
    TAGS_ENDPOINT,
    { name, color },
    idempotencyConfig(),
  )
  return { success: true, tag: toTag(tag) }
}

function requireTagId(name: string): string {
  const id = tagIdsByName.get(name)
  if (!id) throw new Error(`标签不存在：${name}`)
  return id
}

export async function deleteTag(tagName: string): Promise<ApiResponse> {
  const tagId = requireTagId(tagName)
  await apiClient.delete(
    `${TAGS_ENDPOINT}/${encodeURIComponent(tagId)}`,
    idempotencyConfig(),
  )
  tagIdsByName.delete(tagName)
  return { success: true }
}

export async function updateTag(
  currentName: string,
  name: string,
  color: string,
): Promise<TagDetailResponse> {
  const tagId = requireTagId(currentName)
  const tag = await apiClient.put<V2Tag>(
    `${TAGS_ENDPOINT}/${encodeURIComponent(tagId)}`,
    { name, color },
    idempotencyConfig(),
  )
  tagIdsByName.delete(currentName)
  return { success: true, tag: toTag(tag) }
}
