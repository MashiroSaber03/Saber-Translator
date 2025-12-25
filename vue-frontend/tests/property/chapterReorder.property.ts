/**
 * 章节拖拽排序属性测试
 * Property 13: 章节排序一致性
 * Validates: Requirements 21.2
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, ChapterData } from '@/types/api'

describe('Property 13: 章节排序一致性', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * 生成章节数据的 Arbitrary
   */
  const chapterArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 50 }),
    order: fc.nat({ max: 100 }),
    imageCount: fc.nat({ max: 100 }),
    hasSession: fc.boolean(),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<ChapterData>

  /**
   * 生成书籍数据的 Arbitrary
   */
  const bookWithChaptersArbitrary = fc.record({
    id: fc.uuid(),
    title: fc.string({ minLength: 1, maxLength: 100 }),
    description: fc.option(fc.string({ maxLength: 200 }), { nil: undefined }),
    cover: fc.option(fc.string(), { nil: undefined }),
    tags: fc.array(fc.uuid(), { maxLength: 5 }),
    chapters: fc.array(chapterArbitrary, { minLength: 1, maxLength: 20 }),
    createdAt: fc.date().map(d => d.toISOString()),
    updatedAt: fc.date().map(d => d.toISOString()),
  }) as fc.Arbitrary<BookData>

  it('拖拽后章节顺序正确更新', () => {
    fc.assert(
      fc.property(
        bookWithChaptersArbitrary,
        fc.nat({ max: 100 }),
        (book, seed) => {
          const store = useBookshelfStore()
          store.setBooks([book])

          // 获取原始章节ID列表
          const originalIds = book.chapters!.map(c => c.id)
          if (originalIds.length < 2) return true // 至少需要2个章节才能测试排序

          // 生成新的随机顺序
          const shuffledIds = [...originalIds]
          // 使用 Fisher-Yates 洗牌算法
          for (let i = shuffledIds.length - 1; i > 0; i--) {
            const j = (seed + i) % (i + 1)
            ;[shuffledIds[i], shuffledIds[j]] = [shuffledIds[j], shuffledIds[i]]
          }

          // 执行重新排序
          store.reorderChapters(book.id, shuffledIds)

          // 验证：章节顺序应该与新顺序一致
          const reorderedBook = store.getBookById(book.id)
          const reorderedIds = reorderedBook?.chapters?.map(c => c.id) || []
          
          expect(reorderedIds).toEqual(shuffledIds)
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('排序后 order 字段连续性', () => {
    fc.assert(
      fc.property(
        bookWithChaptersArbitrary,
        (book) => {
          const store = useBookshelfStore()
          store.setBooks([book])

          // 获取章节ID列表并反转顺序
          const originalIds = book.chapters!.map(c => c.id)
          if (originalIds.length < 2) return true

          const reversedIds = [...originalIds].reverse()

          // 执行重新排序
          store.reorderChapters(book.id, reversedIds)

          // 验证：order 字段应该是连续的 0, 1, 2, ...
          const reorderedBook = store.getBookById(book.id)
          const orders = reorderedBook?.chapters?.map(c => c.order) || []
          
          for (let i = 0; i < orders.length; i++) {
            expect(orders[i]).toBe(i)
          }
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('排序不改变章节数量', () => {
    fc.assert(
      fc.property(
        bookWithChaptersArbitrary,
        (book) => {
          const store = useBookshelfStore()
          store.setBooks([book])

          const originalCount = book.chapters!.length
          const originalIds = book.chapters!.map(c => c.id)

          // 执行重新排序（反转顺序）
          store.reorderChapters(book.id, [...originalIds].reverse())

          // 验证：章节数量不变
          const reorderedBook = store.getBookById(book.id)
          expect(reorderedBook?.chapters?.length).toBe(originalCount)
          return true
        }
      ),
      { numRuns: 50 }
    )
  })

  it('排序保留所有章节内容', () => {
    fc.assert(
      fc.property(
        bookWithChaptersArbitrary,
        (book) => {
          const store = useBookshelfStore()
          store.setBooks([book])

          // 记录原始章节标题
          const originalTitles = new Set(book.chapters!.map(c => c.title))
          const originalIds = book.chapters!.map(c => c.id)

          // 执行重新排序
          store.reorderChapters(book.id, [...originalIds].reverse())

          // 验证：所有章节标题都保留
          const reorderedBook = store.getBookById(book.id)
          const reorderedTitles = new Set(reorderedBook?.chapters?.map(c => c.title) || [])
          
          expect(reorderedTitles).toEqual(originalTitles)
          return true
        }
      ),
      { numRuns: 50 }
    )
  })
})
