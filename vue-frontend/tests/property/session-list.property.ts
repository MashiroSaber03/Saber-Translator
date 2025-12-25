/**
 * 会话列表管理属性测试
 * 
 * **Feature: vue-frontend-migration, Property 34: 会话保存加载往返一致性**
 * **Validates: Requirements 14.1, 14.2**
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'

describe('会话列表管理属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 34.2: 会话列表管理一致性
   */
  it('Property 34.2: 会话列表管理一致性', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 5 }),
        fc.uuid(),
        (count, uniqueId) => {
          // 每次迭代重新创建 Pinia 实例
          setActivePinia(createPinia())
          const sessionStore = useSessionStore()
          
          // 生成简单的会话列表
          const sessions = Array.from({ length: count }, (_, i) => ({
            name: `session_${uniqueId}_${i}`,
            savedAt: new Date().toISOString(),
            imageCount: i * 10,
            version: '2.0'
          }))
          
          sessionStore.setSessionList(sessions)
          const afterSetLength = sessionStore.sessionList.length
          
          const newSession = {
            name: `new_session_${uniqueId}`,
            savedAt: new Date().toISOString(),
            imageCount: 5,
            version: '2.0'
          }
          sessionStore.addToSessionList(newSession)
          const afterAddLength = sessionStore.sessionList.length
          
          // 验证添加后长度增加
          expect(afterAddLength).toBe(afterSetLength + 1)
          
          sessionStore.removeFromSessionList(newSession.name)
          expect(sessionStore.sessionList.length).toBe(afterSetLength)
        }
      ),
      { numRuns: 20 }
    )
  })
})
