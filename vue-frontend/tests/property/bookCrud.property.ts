/**
 * 书籍CRUD操作属性测试
 * Property 24: 书籍CRUD操作一致性
 * Validates: Requirements 3.2, 3.5
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData } from '@/types/api'

describe('Property 24: 书籍CRUD操作一致性', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成书籍数据的 Arbitrary
   */
  const bookArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(fc.uuid(), { maxLength: 5 }),
    chapters: fc.constant([]),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  it('创建书籍后列表正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 0, maxLength: 10 }),
        bookArbitrary,
        (existingBooks, newBook) => {
          const store = useBookshelfStore()
          
          // 设置初始书籍列表
          store.setBooks(existingBooks)
          const initialCount = store.bookCount

          // 添加新书籍
          store.addBook(newBook)

          // 验证：书籍数量增加1
          expect(store.bookCount).toBe(initialCount + 1)
          
          // 验证：新书籍在列表中
          const found = store.getBookById(newBook.id)
          expect(found).not.toBeNull()
          expect(found?.title).toBe(newBook.title)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('删除书籍后列表正确更新', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 1, maxLength: 10 }),
        (books) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          
          const initialCount = store.bookCount
          const bookToDelete = books[0]

          // 删除书籍
          store.deleteBook(bookToDelete.id)

          // 验证：书籍数量减少1
          expect(store.bookCount).toBe(initialCount - 1)
          
          // 验证：书籍不在列表中
          const found = store.getBookById(bookToDelete.id)
          expect(found).toBeNull()
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('批量删除操作正确性', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 3, maxLength: 10 }),
        fc.nat({ max: 2 }),
        (books, deleteCount) => {
          const store = useBookshelfStore()
          store.setBooks(books)
          
          const initialCount = store.bookCount
          const booksToDelete = books.slice(0, deleteCount + 1)
          const idsToDelete = booksToDelete.map(b => b.id)

          // 批量删除
          store.deleteBooks(idsToDelete)

          // 验证：书籍数量正确减少
          expect(store.bookCount).toBe(initialCount - idsToDelete.length)
          
          // 验证：所有删除的书籍都不在列表中
          for (const id of idsToDelete) {
            expect(store.getBookById(id)).toBeNull()
          }
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('更新书籍后数据正确更新', () => {
    fc.assert(
      fc.property(
        bookArbitrary,
        fc.string({ minLength: 1, maxLength: 100 }),
        (book, newTitle) => {
          const store = useBookshelfStore()
          store.setBooks([book])

          // 更新书籍标题
          store.updateBook(book.id, { title: newTitle })

          // 验证：标题已更新
          const updated = store.getBookById(book.id)
          expect(updated?.title).toBe(newTitle)
          
          // 验证：其他字段保持不变
          expect(updated?.id).toBe(book.id)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('删除不存在的书籍不影响列表', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 1, maxLength: 10 }),
        fc.uuid(),
        (books, nonExistentId) => {
          // 确保ID不在列表中
          const existingIds = new Set(books.map(b => b.id))
          if (existingIds.has(nonExistentId)) return true

          const store = useBookshelfStore()
          store.setBooks(books)
          
          const initialCount = store.bookCount

          // 尝试删除不存在的书籍
          store.deleteBook(nonExistentId)

          // 验证：书籍数量不变
          expect(store.bookCount).toBe(initialCount)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('新添加的书籍在列表最前面', () => {
    fc.assert(
      fc.property(
        fc.array(bookArbitrary, { minLength: 1, maxLength: 10 }),
        bookArbitrary,
        (existingBooks, newBook) => {
          const store = useBookshelfStore()
          store.setBooks(existingBooks)

          // 添加新书籍
          store.addBook(newBook)

          // 验证：新书籍在列表第一位
          expect(store.books[0].id).toBe(newBook.id)
          
          return true
        }
      ),
      { numRuns: 50 }
    )
  })
})
