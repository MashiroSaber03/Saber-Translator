/**
 * 漫画分析问答流式响应属性测试
 * 
 * **Feature: vue-frontend-migration, Property 32: 问答流式响应一致性**
 * **Validates: Requirements 6.3**
 */

import { describe, it, expect, beforeEach } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type QAMessage } from '@/stores/insightStore'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的消息ID
 */
const messageIdArb = fc.stringOf(fc.constantFrom(...'0123456789'.split('')), { minLength: 1, maxLength: 20 })

/**
 * 生成有效的消息角色
 */
const messageRoleArb = fc.constantFrom<'user' | 'assistant'>('user', 'assistant')

/**
 * 生成有效的消息内容
 */
const messageContentArb = fc.string({ minLength: 0, maxLength: 500 })

/**
 * 生成有效的ISO日期字符串
 */
const isoDateArb = fc.date().map(d => d.toISOString())

/**
 * 生成有效的问答消息
 */
const qaMessageArb: fc.Arbitrary<QAMessage> = fc.record({
  id: messageIdArb,
  role: messageRoleArb,
  content: messageContentArb,
  timestamp: isoDateArb
})

/**
 * 生成有效的流式响应块
 */
const streamChunkArb = fc.string({ minLength: 1, maxLength: 100 })

// ============================================================
// 属性测试
// ============================================================

describe('漫画分析问答流式响应属性测试', () => {
  beforeEach(() => {
    // 创建新的 Pinia 实例
    setActivePinia(createPinia())
  })

  /**
   * Property 32: 消息添加后队列正确更新
   * 
   * *对于任意* 有效的消息序列，添加后队列长度应当正确增加
   */
  it('消息添加后队列长度应当正确增加', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArb, { minLength: 0, maxLength: 20 }),
        (messages) => {
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          const store = useInsightStore()
          
          // 添加消息
          for (const message of messages) {
            store.addQAMessage(message)
          }
          
          // 验证队列长度
          expect(store.qaHistory.length).toBe(messages.length)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 消息顺序保持一致
   * 
   * *对于任意* 有效的消息序列，添加后消息顺序应当与添加顺序一致
   */
  it('消息顺序应当与添加顺序一致', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArb, { minLength: 1, maxLength: 20 }),
        (messages) => {
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          const store = useInsightStore()
          
          // 添加消息
          for (const message of messages) {
            store.addQAMessage(message)
          }
          
          // 验证消息顺序
          for (let i = 0; i < messages.length; i++) {
            expect(store.qaHistory[i].id).toBe(messages[i].id)
            expect(store.qaHistory[i].role).toBe(messages[i].role)
            expect(store.qaHistory[i].content).toBe(messages[i].content)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 流式响应内容累积正确
   * 
   * *对于任意* 流式响应块序列，最终内容应当是所有块的拼接
   */
  it('流式响应内容累积应当正确', () => {
    fc.assert(
      fc.property(
        fc.array(streamChunkArb, { minLength: 1, maxLength: 20 }),
        (chunks) => {
          const store = useInsightStore()
          
          // 添加一条助手消息
          store.addQAMessage({
            id: '1',
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString()
          })
          
          // 模拟流式响应
          let accumulatedContent = ''
          for (const chunk of chunks) {
            accumulatedContent += chunk
            store.updateLastAssistantMessage(accumulatedContent)
          }
          
          // 验证最终内容
          const lastMessage = store.qaHistory[store.qaHistory.length - 1]
          expect(lastMessage.content).toBe(accumulatedContent)
          expect(lastMessage.content).toBe(chunks.join(''))
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 更新最后一条助手消息不影响其他消息
   * 
   * *对于任意* 消息序列和更新内容，更新操作只影响最后一条助手消息
   */
  it('更新最后一条助手消息不应影响其他消息', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArb, { minLength: 2, maxLength: 10 }),
        messageContentArb,
        (messages, newContent) => {
          const store = useInsightStore()
          
          // 确保最后一条是助手消息
          const messagesWithAssistantLast = [...messages]
          messagesWithAssistantLast[messagesWithAssistantLast.length - 1] = {
            ...messagesWithAssistantLast[messagesWithAssistantLast.length - 1],
            role: 'assistant'
          }
          
          // 添加消息
          for (const message of messagesWithAssistantLast) {
            store.addQAMessage(message)
          }
          
          // 保存其他消息的内容
          const otherMessagesContent = store.qaHistory.slice(0, -1).map(m => m.content)
          
          // 更新最后一条助手消息
          store.updateLastAssistantMessage(newContent)
          
          // 验证其他消息未受影响
          for (let i = 0; i < otherMessagesContent.length; i++) {
            expect(store.qaHistory[i].content).toBe(otherMessagesContent[i])
          }
          
          // 验证最后一条消息已更新
          expect(store.qaHistory[store.qaHistory.length - 1].content).toBe(newContent)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 清除历史后队列为空
   * 
   * *对于任意* 消息序列，清除后队列应当为空
   */
  it('清除历史后队列应当为空', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArb, { minLength: 0, maxLength: 20 }),
        (messages) => {
          const store = useInsightStore()
          
          // 添加消息
          for (const message of messages) {
            store.addQAMessage(message)
          }
          
          // 清除历史
          store.clearQAHistory()
          
          // 验证队列为空
          expect(store.qaHistory.length).toBe(0)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 流式状态正确切换
   * 
   * *对于任意* 流式状态切换序列，状态应当正确反映最后设置的值
   */
  it('流式状态切换应当正确', () => {
    fc.assert(
      fc.property(
        fc.array(fc.boolean(), { minLength: 1, maxLength: 20 }),
        (streamingSequence) => {
          const store = useInsightStore()
          
          // 依次设置流式状态
          for (const streaming of streamingSequence) {
            store.setStreaming(streaming)
          }
          
          // 验证最终状态
          const lastStreaming = streamingSequence[streamingSequence.length - 1]
          expect(store.isStreaming).toBe(lastStreaming)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 32: 用户和助手消息交替添加
   * 
   * *对于任意* 问答对数量，用户消息和助手消息应当正确交替
   */
  it('用户和助手消息应当正确交替', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        messageContentArb,
        messageContentArb,
        (pairCount, userContent, assistantContent) => {
          // 每次测试创建新的 Pinia 实例
          setActivePinia(createPinia())
          const store = useInsightStore()
          
          // 添加问答对
          for (let i = 0; i < pairCount; i++) {
            store.addQAMessage({
              id: `user-${i}`,
              role: 'user',
              content: userContent,
              timestamp: new Date().toISOString()
            })
            store.addQAMessage({
              id: `assistant-${i}`,
              role: 'assistant',
              content: assistantContent,
              timestamp: new Date().toISOString()
            })
          }
          
          // 验证消息数量
          expect(store.qaHistory.length).toBe(pairCount * 2)
          
          // 验证交替模式
          for (let i = 0; i < pairCount; i++) {
            expect(store.qaHistory[i * 2].role).toBe('user')
            expect(store.qaHistory[i * 2 + 1].role).toBe('assistant')
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
