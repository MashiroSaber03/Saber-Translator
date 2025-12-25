/**
 * 会话上下文管理属性测试
 * 
 * **Feature: vue-frontend-migration, Property 34: 会话保存加载往返一致性**
 * **Validates: Requirements 14.1, 14.2**
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'

// 生成有效的会话名称
const validSessionNameArb = fc.string({ minLength: 1, maxLength: 30 })
  .filter(s => /^[a-z0-9_-]+$/i.test(s))

describe('会话上下文管理属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 34.4: 会话上下文管理一致性
   */
  it('Property 34.4: 会话上下文管理一致性', () => {
    fc.assert(
      fc.property(
        fc.uuid(),
        fc.uuid(),
        (bookId, chapterId) => {
          const sessionStore = useSessionStore()
          
          expect(sessionStore.isBookshelfMode).toBe(false)
          
          sessionStore.setContext(bookId, chapterId)
          expect(sessionStore.isBookshelfMode).toBe(true)
          expect(sessionStore.currentBookId).toBe(bookId)
          expect(sessionStore.currentChapterId).toBe(chapterId)
          
          sessionStore.clearContext()
          expect(sessionStore.isBookshelfMode).toBe(false)
          expect(sessionStore.currentBookId).toBeNull()
          expect(sessionStore.currentChapterId).toBeNull()
        }
      ),
      { numRuns: 20 }
    )
  })

  /**
   * Property 34.5: URL 参数解析一致性
   */
  it('Property 34.5: URL 参数解析一致性', () => {
    fc.assert(
      fc.property(
        fc.uuid(),
        fc.uuid(),
        (bookId, chapterId) => {
          const sessionStore = useSessionStore()
          
          const searchParams = new URLSearchParams()
          searchParams.set('book', bookId)
          searchParams.set('chapter', chapterId)
          
          sessionStore.parseContextFromUrl(searchParams)
          
          expect(sessionStore.isBookshelfMode).toBe(true)
          expect(sessionStore.currentBookId).toBe(bookId)
          expect(sessionStore.currentChapterId).toBe(chapterId)
        }
      ),
      { numRuns: 20 }
    )
  })

  /**
   * Property 34.6: 会话名称管理一致性
   */
  it('Property 34.6: 会话名称管理一致性', () => {
    fc.assert(
      fc.property(
        validSessionNameArb,
        (sessionName) => {
          const sessionStore = useSessionStore()
          
          expect(sessionStore.currentSessionName).toBeNull()
          
          sessionStore.setSessionName(sessionName)
          expect(sessionStore.currentSessionName).toBe(sessionName)
          
          sessionStore.clearSessionName()
          expect(sessionStore.currentSessionName).toBeNull()
        }
      ),
      { numRuns: 20 }
    )
  })

  /**
   * Property 34.7: 图片 URL 转 Base64 格式一致性
   */
  it('Property 34.7: 图片 URL 转 Base64 格式一致性', async () => {
    const sessionStore = useSessionStore()
    
    const base64Data = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    const result = await sessionStore.imageUrlToBase64(base64Data)
    expect(result).toBe(base64Data)
  })

  /**
   * Property 34.8: 重置状态一致性
   */
  it('Property 34.8: 重置状态一致性', () => {
    fc.assert(
      fc.property(
        fc.uuid(),
        fc.uuid(),
        validSessionNameArb,
        (bookId, chapterId, sessionName) => {
          const sessionStore = useSessionStore()
          
          sessionStore.setContext(bookId, chapterId)
          sessionStore.setSessionName(sessionName)
          sessionStore.setSessionList([{ name: 'test', savedAt: new Date().toISOString(), imageCount: 1, version: '2.0' }])
          sessionStore.startBatchSave(10, 'test-session-id')
          sessionStore.setError('测试错误')
          
          expect(sessionStore.isBookshelfMode).toBe(true)
          expect(sessionStore.currentSessionName).toBe(sessionName)
          expect(sessionStore.sessionList.length).toBe(1)
          expect(sessionStore.batchSaveState.isInProgress).toBe(true)
          expect(sessionStore.error).toBe('测试错误')
          
          sessionStore.reset()
          
          expect(sessionStore.isBookshelfMode).toBe(false)
          expect(sessionStore.currentSessionName).toBeNull()
          expect(sessionStore.sessionList.length).toBe(0)
          expect(sessionStore.batchSaveState.isInProgress).toBe(false)
          expect(sessionStore.error).toBeNull()
        }
      ),
      { numRuns: 20 }
    )
  })
})
