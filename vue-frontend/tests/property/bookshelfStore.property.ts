/**
 * 书架状态管理属性测试
 * 
 * Feature: vue-frontend-migration
 * Property 1: 书籍搜索过滤一致性
 * Property 2: 标签筛选一致性
 * 
 * Validates: Requirements 3.3, 3.4
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { BookData, TagData } from '@/types/api'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的书籍ID
 */
const bookIdArb = fc.uuid()

/**
 * 生成有效的标签ID
 */
const tagIdArb = fc.uuid()

/**
 * 生成有效的书籍标题
 */
const bookTitleArb = fc.string({ minLength: 1, maxLength: 100 })

/**
 * 生成有效的书籍描述
 */
const bookDescriptionArb = fc.option(fc.string({ maxLength: 500 }), { nil: undefined })

/**
 * 生成有效的日期字符串
 */
const dateStringArb = fc.date({ min: new Date('2020-01-01'), max: new Date('2025-12-31') })
  .map(d => d.toISOString())

/**
 * 生成有效的标签数据
 */
const tagDataArb: fc.Arbitrary<TagData> = fc.record({
  id: tagIdArb,
  name: fc.string({ minLength: 1, maxLength: 50 }),
  color: fc.option(fc.hexaString({ minLength: 6, maxLength: 6 }).map(h => `#${h}`), { nil: undefined })
})

/**
 * 生成有效的书籍数据
 */
const bookDataArb = (tagIds: string[]): fc.Arbitrary<BookData> => fc.record({
  id: bookIdArb,
  title: bookTitleArb,
  description: bookDescriptionArb,
  cover: fc.option(fc.string(), { nil: undefined }),
  tags: fc.option(
    fc.subarray(tagIds, { minLength: 0, maxLength: Math.min(tagIds.length, 5) }),
    { nil: undefined }
  ),
  chapters: fc.constant(undefined),
  createdAt: dateStringArb,
  updatedAt: dateStringArb
})

/**
 * 生成书籍列表和标签列表
 */
const bookshelfDataArb = fc.array(tagDataArb, { minLength: 0, maxLength: 10 })
  .chain(tags => {
    const tagIds = tags.map(t => t.id)
    return fc.tuple(
      fc.constant(tags),
      fc.array(bookDataArb(tagIds), { minLength: 0, maxLength: 20 })
    )
  })

// ============================================================
// 属性测试
// ============================================================

describe('书架状态管理属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 1: 书籍搜索过滤一致性
   * 
   * 对于任意书籍列表和搜索关键词，过滤后的结果应当仅包含标题或描述中包含该关键词的书籍，
   * 且不遗漏任何匹配项。
   * 
   * Feature: vue-frontend-migration, Property 1: 书籍搜索过滤一致性
   * Validates: Requirements 3.3
   */
  it('书籍搜索过滤一致性', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(fc.record({
            id: bookIdArb,
            title: bookTitleArb,
            description: bookDescriptionArb,
            createdAt: dateStringArb,
            updatedAt: dateStringArb
          }), { minLength: 0, maxLength: 20 }),
          fc.string({ maxLength: 20 })
        ),
        ([books, keyword]) => {
          const store = useBookshelfStore()
          
          // 设置书籍列表
          store.setBooks(books as BookData[])
          
          // 设置搜索关键词
          store.setSearchKeyword(keyword)
          
          const filtered = store.filteredBooks
          const keywordLower = keyword.toLowerCase().trim()
          
          if (keywordLower === '') {
            // 空关键词应返回所有书籍
            expect(filtered.length).toBe(books.length)
          } else {
            // 验证所有过滤结果都包含关键词
            for (const book of filtered) {
              const titleMatch = book.title.toLowerCase().includes(keywordLower)
              const descMatch = book.description?.toLowerCase().includes(keywordLower) || false
              expect(titleMatch || descMatch).toBe(true)
            }
            
            // 验证没有遗漏匹配项
            for (const book of books) {
              const titleMatch = book.title.toLowerCase().includes(keywordLower)
              const descMatch = book.description?.toLowerCase().includes(keywordLower) || false
              const shouldBeIncluded = titleMatch || descMatch
              const isIncluded = filtered.some(b => b.id === book.id)
              expect(isIncluded).toBe(shouldBeIncluded)
            }
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 2: 标签筛选一致性
   * 
   * 对于任意书籍列表和标签集合，筛选后的结果应当仅包含拥有所选标签的书籍，
   * 且不遗漏任何匹配项。
   * 
   * Feature: vue-frontend-migration, Property 2: 标签筛选一致性
   * Validates: Requirements 3.4
   */
  it('标签筛选一致性', () => {
    fc.assert(
      fc.property(bookshelfDataArb, ([tags, books]) => {
        const store = useBookshelfStore()
        
        // 设置标签和书籍列表
        store.setTags(tags)
        store.setBooks(books)
        
        // 随机选择一些标签进行筛选
        const tagIds = tags.map(t => t.id)
        const selectedTagCount = Math.min(tagIds.length, 3)
        const selectedTags = tagIds.slice(0, selectedTagCount)
        
        // 设置标签筛选
        store.setTagFilter(selectedTags)
        
        const filtered = store.filteredBooks
        
        if (selectedTags.length === 0) {
          // 没有选择标签时应返回所有书籍
          expect(filtered.length).toBe(books.length)
        } else {
          // 验证所有过滤结果都包含所有选中的标签
          for (const book of filtered) {
            for (const tagId of selectedTags) {
              expect(book.tags?.includes(tagId)).toBe(true)
            }
          }
          
          // 验证没有遗漏匹配项
          for (const book of books) {
            const hasAllTags = selectedTags.every(tagId => book.tags?.includes(tagId))
            const isIncluded = filtered.some(b => b.id === book.id)
            expect(isIncluded).toBe(hasAllTags)
          }
        }
        
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * 批量选择状态一致性测试
   */
  it('批量选择状态一致性', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(fc.record({
            id: bookIdArb,
            title: bookTitleArb,
            createdAt: dateStringArb,
            updatedAt: dateStringArb
          }), { minLength: 1, maxLength: 10 }),
          fc.array(fc.integer({ min: 0, max: 9 }), { minLength: 1, maxLength: 20 })
        ),
        ([books, toggleIndices]) => {
          const store = useBookshelfStore()
          
          // 设置书籍列表
          store.setBooks(books as BookData[])
          
          // 进入批量模式
          store.enterBatchMode()
          expect(store.batchMode).toBe(true)
          
          // 执行选择操作
          const expectedSelected = new Set<string>()
          
          for (const index of toggleIndices) {
            if (index < books.length) {
              const bookId = books[index]?.id
              if (bookId) {
                store.toggleBookSelection(bookId)
                
                if (expectedSelected.has(bookId)) {
                  expectedSelected.delete(bookId)
                } else {
                  expectedSelected.add(bookId)
                }
              }
            }
          }
          
          // 验证选中状态
          expect(store.selectedBookIds.size).toBe(expectedSelected.size)
          for (const bookId of expectedSelected) {
            expect(store.selectedBookIds.has(bookId)).toBe(true)
          }
          
          // 退出批量模式应清除选择
          store.exitBatchMode()
          expect(store.batchMode).toBe(false)
          expect(store.selectedBookIds.size).toBe(0)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * 全选/取消全选一致性测试
   */
  it('全选/取消全选一致性', () => {
    fc.assert(
      fc.property(
        fc.array(fc.record({
          id: bookIdArb,
          title: bookTitleArb,
          createdAt: dateStringArb,
          updatedAt: dateStringArb
        }), { minLength: 1, maxLength: 10 }),
        (books) => {
          const store = useBookshelfStore()
          
          // 设置书籍列表
          store.setBooks(books as BookData[])
          
          // 进入批量模式
          store.enterBatchMode()
          
          // 全选
          store.toggleSelectAll()
          expect(store.isAllSelected).toBe(true)
          expect(store.selectedBookIds.size).toBe(books.length)
          
          // 取消全选
          store.toggleSelectAll()
          expect(store.isAllSelected).toBe(false)
          expect(store.selectedBookIds.size).toBe(0)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * 标签切换筛选一致性测试
   */
  it('标签切换筛选一致性', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(tagDataArb, { minLength: 1, maxLength: 5 }),
          fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 10 })
        ),
        ([tags, toggleIndices]) => {
          const store = useBookshelfStore()
          
          // 设置标签列表
          store.setTags(tags)
          
          // 清除之前的筛选状态
          store.clearTagFilter()
          
          // 执行标签切换操作
          const expectedSelected = new Set<string>()
          
          for (const index of toggleIndices) {
            if (index < tags.length) {
              const tagId = tags[index]?.id
              if (tagId) {
                store.toggleTagFilter(tagId)
                
                if (expectedSelected.has(tagId)) {
                  expectedSelected.delete(tagId)
                } else {
                  expectedSelected.add(tagId)
                }
              }
            }
          }
          
          // 验证选中的标签数量
          const expectedArray = Array.from(expectedSelected)
          expect(store.selectedTagIds.length).toBe(expectedArray.length)
          
          // 验证每个期望的标签都在选中列表中
          for (const tagId of expectedArray) {
            expect(store.selectedTagIds.includes(tagId)).toBe(true)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * 书籍删除后选中状态更新测试
   */
  it('书籍删除后选中状态应正确更新', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(fc.record({
            id: bookIdArb,
            title: bookTitleArb,
            createdAt: dateStringArb,
            updatedAt: dateStringArb
          }), { minLength: 2, maxLength: 10 }),
          fc.integer({ min: 0, max: 9 })
        ),
        ([books, deleteIndex]) => {
          const store = useBookshelfStore()
          
          // 设置书籍列表
          store.setBooks(books as BookData[])
          
          // 进入批量模式并选中要删除的书籍
          store.enterBatchMode()
          
          const actualIndex = Math.min(deleteIndex, books.length - 1)
          const bookToDelete = books[actualIndex]
          
          if (bookToDelete) {
            store.toggleBookSelection(bookToDelete.id)
            expect(store.selectedBookIds.has(bookToDelete.id)).toBe(true)
            
            // 删除书籍
            store.deleteBook(bookToDelete.id)
            
            // 验证书籍已从列表中移除
            expect(store.books.find(b => b.id === bookToDelete.id)).toBeUndefined()
            
            // 验证选中状态已更新
            expect(store.selectedBookIds.has(bookToDelete.id)).toBe(false)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
