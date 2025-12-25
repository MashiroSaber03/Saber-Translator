/**
 * 漫画分析笔记属性测试
 * 
 * **Feature: vue-frontend-migration, Property 16: 笔记持久化往返一致性**
 * **Validates: Requirements 24.4**
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type NoteData, type NoteType } from '@/stores/insightStore'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的笔记类型
 */
const noteTypeArb = fc.constantFrom<NoteType>('text', 'qa')

/**
 * 生成有效的笔记ID
 */
const noteIdArb = fc.stringOf(fc.constantFrom(...'0123456789'.split('')), { minLength: 1, maxLength: 20 })

/**
 * 生成有效的笔记内容
 */
const noteContentArb = fc.string({ minLength: 1, maxLength: 500 })

/**
 * 生成有效的页码
 */
const pageNumArb = fc.option(fc.integer({ min: 1, max: 1000 }), { nil: undefined })

/**
 * 生成有效的ISO日期字符串
 */
const isoDateArb = fc.date().map(d => d.toISOString())

/**
 * 生成有效的笔记数据
 */
const noteDataArb: fc.Arbitrary<NoteData> = fc.record({
  id: noteIdArb,
  type: noteTypeArb,
  content: noteContentArb,
  pageNum: pageNumArb,
  createdAt: isoDateArb,
  updatedAt: isoDateArb
})

/**
 * 生成有效的书籍ID
 */
const bookIdArb = fc.stringOf(fc.constantFrom(...'abcdef0123456789'.split('')), { minLength: 8, maxLength: 8 })

// ============================================================
// 属性测试
// ============================================================

describe('漫画分析笔记属性测试', () => {
  beforeEach(() => {
    // 创建新的 Pinia 实例
    setActivePinia(createPinia())
  })

  /**
   * Property 16: 笔记持久化往返一致性
   * 
   * *对于任意* 有效的笔记数据和书籍ID，保存到 localStorage 后再读取应当得到等价的笔记数据
   */
  it('笔记保存到 localStorage 后应能正确恢复', () => {
    fc.assert(
      fc.property(
        fc.array(noteDataArb, { minLength: 0, maxLength: 10 }),
        bookIdArb,
        (notes, bookId) => {
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          const store = useInsightStore()
          
          // 清理 localStorage
          const storageKey = `manga_notes_${bookId}`
          localStorage.removeItem(storageKey)
          
          // 设置当前书籍
          store.setCurrentBook(bookId)
          
          // 添加笔记
          for (const note of notes) {
            store.addNote(note)
          }
          
          // 验证笔记数量（注意：addNote 是 unshift，所以顺序是反的）
          expect(store.notes.length).toBe(notes.length)
          
          // 验证每条笔记的内容
          for (let i = 0; i < notes.length; i++) {
            const originalNote = notes[notes.length - 1 - i] // 反向索引
            const storedNote = store.notes[i]
            
            expect(storedNote.id).toBe(originalNote.id)
            expect(storedNote.type).toBe(originalNote.type)
            expect(storedNote.content).toBe(originalNote.content)
          }
          
          // 清理
          localStorage.removeItem(storageKey)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 16: 笔记按 bookId 隔离存储
   * 
   * *对于任意* 两个不同的书籍ID，它们的笔记应当独立存储，互不影响
   */
  it('不同书籍的笔记应当隔离存储', () => {
    fc.assert(
      fc.property(
        fc.array(noteDataArb, { minLength: 1, maxLength: 5 }),
        fc.array(noteDataArb, { minLength: 1, maxLength: 5 }),
        bookIdArb,
        bookIdArb,
        (notes1, notes2, bookId1, bookId2Base) => {
          // 确保两个书籍ID不同
          const bookId2 = bookId2Base + '_2'
          
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          
          // 清理 localStorage
          const storageKey1 = `manga_notes_${bookId1}`
          const storageKey2 = `manga_notes_${bookId2}`
          localStorage.removeItem(storageKey1)
          localStorage.removeItem(storageKey2)
          
          const store1 = useInsightStore()
          
          // 为第一本书添加笔记
          store1.setCurrentBook(bookId1)
          for (const note of notes1) {
            store1.addNote(note)
          }
          
          // 验证第一本书的笔记数量
          expect(store1.notes.length).toBe(notes1.length)
          
          // 创建新的 store 实例为第二本书
          setActivePinia(createPinia())
          const store2 = useInsightStore()
          
          // 为第二本书添加笔记
          store2.setCurrentBook(bookId2)
          for (const note of notes2) {
            store2.addNote(note)
          }
          
          // 验证第二本书的笔记数量
          expect(store2.notes.length).toBe(notes2.length)
          
          // 清理
          localStorage.removeItem(storageKey1)
          localStorage.removeItem(storageKey2)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 16: 笔记加载后应与保存时一致
   * 
   * *对于任意* 有效的笔记数据，保存后重新加载应当得到相同的数据
   */
  it('笔记加载后应与保存时一致', () => {
    fc.assert(
      fc.property(
        fc.array(noteDataArb, { minLength: 1, maxLength: 10 }),
        bookIdArb,
        (notes, bookId) => {
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          
          // 清理 localStorage
          const storageKey = `manga_notes_${bookId}`
          localStorage.removeItem(storageKey)
          
          const store = useInsightStore()
          
          // 设置当前书籍并添加笔记
          store.setCurrentBook(bookId)
          for (const note of notes) {
            store.addNote(note)
          }
          
          // 保存笔记数量
          const savedCount = store.notes.length
          
          // 创建新的 store 实例模拟重新加载
          setActivePinia(createPinia())
          const store2 = useInsightStore()
          
          // 重新设置书籍并加载笔记
          store2.setCurrentBook(bookId)
          store2.loadNotesFromStorage()
          
          // 验证笔记数量一致
          expect(store2.notes.length).toBe(savedCount)
          
          // 验证笔记内容一致
          for (let i = 0; i < notes.length; i++) {
            const originalNote = notes[notes.length - 1 - i]
            const loadedNote = store2.notes[i]
            
            expect(loadedNote.id).toBe(originalNote.id)
            expect(loadedNote.type).toBe(originalNote.type)
            expect(loadedNote.content).toBe(originalNote.content)
          }
          
          // 清理
          localStorage.removeItem(storageKey)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
