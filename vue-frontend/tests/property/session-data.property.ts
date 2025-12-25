/**
 * 会话数据创建属性测试
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

describe('会话数据创建属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 34.1: 会话数据创建一致性
   */
  it('Property 34.1: 会话数据创建一致性', () => {
    fc.assert(
      fc.property(
        validSessionNameArb,
        fc.integer({ min: -1, max: 10 }),
        (name, currentIndex) => {
          const sessionStore = useSessionStore()
          
          const sessionData = sessionStore.createSessionData(
            name,
            [],
            currentIndex,
            { testSetting: 'value' }
          )
          
          expect(sessionData.name).toBe(name)
          expect(sessionData.version).toBe('2.0')
          expect(sessionData.imageCount).toBe(0)
          expect(sessionData.currentImageIndex).toBe(currentIndex)
          expect(sessionData.ui_settings).toEqual({ testSetting: 'value' })
          expect(sessionData.images.length).toBe(0)
          expect(new Date(sessionData.savedAt).toString()).not.toBe('Invalid Date')
        }
      ),
      { numRuns: 20 }
    )
  })
})
