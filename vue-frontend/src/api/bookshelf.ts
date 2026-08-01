import { apiClient } from './client'
import { newIdempotencyKey } from './v2/content'
import type { components } from '@/api/generated/v2'
import type { BookData, ChapterData, TagData } from '@/types'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

type V2Book = components['schemas']['Book']
type V2BookList = components['schemas']['BookList']
type V2Chapter = components['schemas']['Chapter']
type V2ConstraintDocument = components['schemas']['TranslationConstraintDocument']
type V2Tag = components['schemas']['Tag']
type V2TagList = components['schemas']['TagList']

const BOOKS_ENDPOINT = '/api/v2/books'
const TAGS_ENDPOINT = '/api/v2/tags'
const tagIdsByName = new Map<string, string>()
const chapterOrderRevisions = new Map<string, number>()
const constraintRevisions = new Map<string, number>()

export interface GetBooksParams {
  search?: string
  tags?: string[]
  sortBy?: 'title' | 'createdAt' | 'updatedAt'
  sortOrder?: 'asc' | 'desc'
}

export type BookBatchDeleteResult = components['schemas']['BookBatchDeleteResult']

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
    ordinal: chapter.ordinal,
    pageOrderRevision: chapter.pageOrderRevision,
    jobStatusSummary: chapter.jobStatusSummary,
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
    translationConstraints: constraints,
    chapters,
    chapterCount: book.chapterCount ?? chapters?.length ?? 0,
    totalPages: book.pageCount ?? chapters?.reduce(
      (total, chapter) => total + (chapter.imageCount || 0),
      0,
    ) ?? 0,
    createdAt: book.createdAt,
    updatedAt: book.updatedAt,
    chapterOrderRevision: book.chapterOrderRevision,
    jobStatusSummary: book.jobStatusSummary,
  }
}

function toTag(tag: V2Tag): TagData {
  tagIdsByName.set(tag.name, tag.id)
  return {
    id: tag.id,
    name: tag.name,
    color: tag.color,
    bookCount: tag.bookCount || 0,
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
  const [book, constraints] = await Promise.all([
    apiClient.get<V2Book & { chapters: V2Chapter[] }>(bookPath(bookId)),
    getConstraints(bookId),
  ])
  return toBook(book, constraints)
}

function bookFormData(
  title: string,
  tagIds: string[],
  cover?: File,
): FormData {
  const body = new FormData()
  body.append('title', title)
  body.append('tagIds', JSON.stringify(tagIds))
  if (cover) body.append('cover', cover, cover.name)
  return body
}

export async function createBook(
  title: string,
  cover?: File,
  tags?: string[],
  translationConstraints?: BookTranslationConstraints,
): Promise<BookData> {
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
    cover?: File
    tags?: string[]
    translationConstraints?: BookTranslationConstraints
  },
): Promise<BookData> {
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
  if (data.translationConstraints) {
    await saveConstraints(bookId, data.translationConstraints)
  }
  return getBookDetail(bookId)
}

export async function deleteBook(bookId: string): Promise<void> {
  await apiClient.delete(bookPath(bookId), idempotencyConfig())
  chapterOrderRevisions.delete(bookId)
  constraintRevisions.delete(bookId)
}

export function batchDeleteBooks(bookIds: string[]): Promise<BookBatchDeleteResult> {
  return apiClient.post<BookBatchDeleteResult>(
    `${BOOKS_ENDPOINT}/batch-delete`,
    { bookIds },
    idempotencyConfig(),
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
    idempotencyConfig(),
  )
}

export async function createChapter(
  bookId: string,
  title: string,
): Promise<ChapterData> {
  const chapter = await apiClient.post<V2Chapter>(
    bookPath(bookId, '/chapters'),
    { title },
    idempotencyConfig(),
  )
  return toChapter(chapter)
}

export async function updateChapter(
  chapterId: string,
  title: string,
): Promise<ChapterData> {
  const chapter = await apiClient.put<V2Chapter>(
    chapterPath(chapterId),
    { title },
    idempotencyConfig(),
  )
  return toChapter(chapter)
}

export async function deleteChapter(
  chapterId: string,
): Promise<void> {
  await apiClient.delete(chapterPath(chapterId), idempotencyConfig())
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
    idempotencyConfig(),
  )
  chapterOrderRevisions.set(bookId, result.chapterOrderRevision)
}

export async function getTags(): Promise<TagData[]> {
  const result = await apiClient.get<V2TagList>(TAGS_ENDPOINT)
  return result.items.map(toTag)
}

export async function createTag(
  name: string,
  color = '#808080',
): Promise<TagData> {
  const tag = await apiClient.post<V2Tag>(
    TAGS_ENDPOINT,
    { name, color },
    idempotencyConfig(),
  )
  return toTag(tag)
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
    idempotencyConfig(),
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
    idempotencyConfig(),
  )
  tagIdsByName.delete(currentName)
  return toTag(tag)
}
