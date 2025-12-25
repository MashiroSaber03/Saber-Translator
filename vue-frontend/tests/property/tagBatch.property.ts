/**
 * 标签批量操作属性测试
 * Property 25: 标签批量操作一致性
 * Validates: Requirements 3.5
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, TagData } from '@/types/api'

describe('Property 25: 标签批量操作一致性', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成标签数据的 Arbitrary
   */
  const tagArbitrary = fc.record({
    id: fc.uuid(),
    name: fc.string({ minLength: 1, maxLength: 20 }),
    color: fc.option(fc.hexaString({ minLength: 6, maxLength: 6 }).map(s => `#${s}`), { nil: undefined }),
  }) as fc.Arbitrary<TagData>

  /**
   * 生成书籍数据的 Arbitrary
   */
  const bookArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(fc.uuid(), { maxLength: 3 }),
    chapters: fc.constant([]),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  it('批量添加标签后书籍标签正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 2, maxLength: 5 }),
        fc.array(tagArbitrary, { minLength: 1, maxLength: 3 }),
        (books, tags) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          store.setTags(tags)

          const bookIds = books.map(b => b.id)
          const tagIds = tags.map(t => t.id)

          // 批量添加标签
          store.batchAddTags(bookIds, tagIds)

          // 验证：每本书都包含所有添加的标签
          for (const bookId of bookIds) {
            const book = store.getBookById(bookId)
            for (const tagId of tagIds) {
              expect(book?.tags).toContain(tagId)
            }
          }
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('批量移除标签后书籍标签正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(tagArbitrary, { minLength: 2, maxLength: 4 }),
        (tags) => {
          const store = useBookshelfStore()
          
          // 创建带有标签的书籍
          const tagIds = tags.map(t => t.id)
          const books: BookData[] = [
            {
              id: 'book-1',
              title: '测试书籍1',
              tags: [...tagIds],
              chapters: [],
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
            {
              id: 'book-2',
              title: '测试书籍2',
              tags: [...tagIds],
              chapters: [],
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          ]
          
          store.setBooks(books)
          store.setTags(tags)

          // 选择要移除的标签（第一个）
          const tagsToRemove = [tagIds[0]]
          const bookIds = books.map(b => b.id)

          // 批量移除标签
          store.batchRemoveTags(bookIds, tagsToRemove)

          // 验证：每本书都不再包含移除的标签
          for (const bookId of bookIds) {
            const book = store.getBookById(bookId)
            expect(book?.tags).not.toContain(tagsToRemove[0])
          }
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('添加已存在的标签不会重复', () => {
    fc.assert(
      fc.property(
        tagArbitrary,
        (tag) => {
          const store = useBookshelfStore()
          
          // 创建已有标签的书籍
          const book: BookData = {
            id: 'book-1',
            title: '测试书籍',
            tags: [tag.id],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          }
          
          store.setBooks([book])
          store.setTags([tag])

          // 再次添加相同标签
          store.addTagToBook(book.id, tag.id)

          // 验证：标签不会重复
          const updatedBook = store.getBookById(book.id)
          const tagCount = updatedBook?.tags?.filter(t => t === tag.id).length || 0
          expect(tagCount).toBe(1)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('移除不存在的标签不影响书籍', () => {
    fc.assert(
      fc.property(
        fc.array(tagArbitrary, { minLength: 1, maxLength: 3 }),
        fc.uuid(),
        (tags, nonExistentTagId) => {
          // 确保标签ID不在列表中
          const existingIds = new Set(tags.map(t => t.id))
          if (existingIds.has(nonExistentTagId)) return true

          const store = useBookshelfStore()
          
          const tagIds = tags.map(t => t.id)
          const book: BookData = {
            id: 'book-1',
            title: '测试书籍',
            tags: [...tagIds],
            chapters: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          }
          
          store.setBooks([book])
          store.setTags(tags)

          const originalTags = [...(store.getBookById(book.id)?.tags || [])]

          // 尝试移除不存在的标签
          store.removeTagFromBook(book.id, nonExistentTagId)

          // 验证：标签列表不变
          const updatedBook = store.getBookById(book.id)
          expect(updatedBook?.tags).toEqual(originalTags)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('批量操作只影响指定的书籍', () => {
    fc.assert(
      fc.property(
        fc.array(tagArbitrary, { minLength: 1, maxLength: 2 }),
        (tags) => {
          const store = useBookshelfStore()
          
          const tagIds = tags.map(t => t.id)
          const books: BookData[] = [
            {
              id: 'book-1',
              title: '测试书籍1',
              tags: [],
              chapters: [],
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
            {
              id: 'book-2',
              title: '测试书籍2',
              tags: [],
              chapters: [],
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
            {
              id: 'book-3',
              title: '测试书籍3',
              tags: [],
              chapters: [],
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            },
          ]
          
          store.setBooks(books)
          store.setTags(tags)

          // 只对前两本书添加标签
          store.batchAddTags(['book-1', 'book-2'], tagIds)

          // 验证：第三本书不受影响
          const book3 = store.getBookById('book-3')
          expect(book3?.tags?.length || 0).toBe(0)
          
          // 验证：前两本书有标签
          const book1 = store.getBookById('book-1')
          const book2 = store.getBookById('book-2')
          expect(book1?.tags?.length).toBeGreaterThan(0)
          expect(book2?.tags?.length).toBeGreaterThan(0)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })
})
