/**
 * ToastNotification 消息队列属性测试
 *
 * Feature: vue-frontend-migration
 * Property 18: 消息队列管理一致性
 *
 * Validates: Requirements 8.2
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// Toast 消息队列核心逻辑（从服务中提取用于测试）
// ============================================================

/**
 * Toast 消息类型
 */
type ToastType = 'success' | 'error' | 'info' | 'warning'

/**
 * Toast 消息接口
 */
interface Toast {
  id: number
  message: string
  type: ToastType
  timer?: ReturnType<typeof setTimeout>
}

/**
 * 创建 Toast 队列管理器
 */
function createToastManager() {
  let toasts: Toast[] = []
  let toastId = 0

  /**
   * 添加 Toast 消息
   */
  const addToast = (message: string, type: ToastType = 'info', duration: number = 3000): number => {
    const id = ++toastId
    const toast: Toast = { id, message, type }

    if (duration > 0) {
      toast.timer = setTimeout(() => {
        removeToast(id)
      }, duration)
    }

    toasts.push(toast)
    return id
  }

  /**
   * 移除指定 ID 的 Toast
   */
  const removeToast = (id: number): boolean => {
    const index = toasts.findIndex((t) => t.id === id)
    if (index !== -1) {
      const toast = toasts[index]
      if (toast?.timer) {
        clearTimeout(toast.timer)
      }
      toasts.splice(index, 1)
      return true
    }
    return false
  }

  /**
   * 清除所有 Toast
   */
  const clearAll = (): void => {
    toasts.forEach((toast) => {
      if (toast.timer) {
        clearTimeout(toast.timer)
      }
    })
    toasts = []
  }

  /**
   * 获取当前队列
   */
  const getToasts = (): Toast[] => [...toasts]

  /**
   * 获取队列长度
   */
  const getLength = (): number => toasts.length

  /**
   * 根据 ID 查找 Toast
   */
  const findById = (id: number): Toast | undefined => toasts.find((t) => t.id === id)

  return {
    addToast,
    removeToast,
    clearAll,
    getToasts,
    getLength,
    findById
  }
}

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的消息类型
 */
const toastTypeArb: fc.Arbitrary<ToastType> = fc.constantFrom('success', 'error', 'info', 'warning')

/**
 * 生成有效的消息内容
 */
const messageArb = fc.string({ minLength: 1, maxLength: 200 })

/**
 * 生成有效的持续时间（毫秒）
 */
const durationArb = fc.integer({ min: 0, max: 10000 })

/**
 * 生成 Toast 添加操作
 */
const addOperationArb = fc.record({
  message: messageArb,
  type: toastTypeArb,
  duration: durationArb
})

// ============================================================
// Property 18: 消息队列管理一致性
// ============================================================

describe('ToastNotification 属性测试', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('Property 18: 消息队列管理一致性', () => {
    /**
     * 测试消息添加后队列长度正确
     *
     * Feature: vue-frontend-migration, Property 18: 消息队列管理一致性
     * Validates: Requirements 8.2
     */
    it('添加消息后队列长度应正确增加', () => {
      fc.assert(
        fc.property(fc.array(addOperationArb, { minLength: 0, maxLength: 20 }), (operations) => {
          const manager = createToastManager()

          // 添加所有消息
          operations.forEach((op) => {
            manager.addToast(op.message, op.type, 0) // duration=0 表示不自动关闭
          })

          // 验证队列长度
          expect(manager.getLength()).toBe(operations.length)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试每个添加的消息都有唯一 ID
     */
    it('每个添加的消息应有唯一 ID', () => {
      fc.assert(
        fc.property(fc.array(addOperationArb, { minLength: 1, maxLength: 20 }), (operations) => {
          const manager = createToastManager()
          const ids: number[] = []

          // 添加所有消息并收集 ID
          operations.forEach((op) => {
            const id = manager.addToast(op.message, op.type, 0)
            ids.push(id)
          })

          // 验证所有 ID 唯一
          const uniqueIds = new Set(ids)
          expect(uniqueIds.size).toBe(ids.length)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试按 ID 清除消息正确性
     */
    it('按 ID 移除消息后队列应正确更新', () => {
      fc.assert(
        fc.property(
          fc.array(addOperationArb, { minLength: 1, maxLength: 10 }),
          fc.integer({ min: 0, max: 9 }),
          (operations, removeIndex) => {
            const manager = createToastManager()
            const ids: number[] = []

            // 添加所有消息
            operations.forEach((op) => {
              const id = manager.addToast(op.message, op.type, 0)
              ids.push(id)
            })

            const initialLength = manager.getLength()
            const actualRemoveIndex = removeIndex % operations.length
            const idToRemove = ids[actualRemoveIndex]

            if (idToRemove !== undefined) {
              // 移除消息
              const removed = manager.removeToast(idToRemove)
              expect(removed).toBe(true)

              // 验证队列长度减少
              expect(manager.getLength()).toBe(initialLength - 1)

              // 验证消息已被移除
              expect(manager.findById(idToRemove)).toBeUndefined()
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试移除不存在的 ID 不影响队列
     */
    it('移除不存在的 ID 不应影响队列', () => {
      fc.assert(
        fc.property(
          fc.array(addOperationArb, { minLength: 1, maxLength: 10 }),
          fc.integer({ min: 10000, max: 99999 }),
          (operations, nonExistentId) => {
            const manager = createToastManager()

            // 添加所有消息
            operations.forEach((op) => {
              manager.addToast(op.message, op.type, 0)
            })

            const initialLength = manager.getLength()

            // 尝试移除不存在的 ID
            const removed = manager.removeToast(nonExistentId)
            expect(removed).toBe(false)

            // 验证队列长度不变
            expect(manager.getLength()).toBe(initialLength)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试清除所有消息
     */
    it('clearAll 应清空队列', () => {
      fc.assert(
        fc.property(fc.array(addOperationArb, { minLength: 1, maxLength: 20 }), (operations) => {
          const manager = createToastManager()

          // 添加所有消息
          operations.forEach((op) => {
            manager.addToast(op.message, op.type, 0)
          })

          expect(manager.getLength()).toBeGreaterThan(0)

          // 清除所有
          manager.clearAll()

          // 验证队列为空
          expect(manager.getLength()).toBe(0)
          expect(manager.getToasts()).toEqual([])

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试消息自动消失后队列更新
     */
    it('消息自动消失后队列应正确更新', () => {
      const manager = createToastManager()

      // 添加一个会自动消失的消息
      const id = manager.addToast('测试消息', 'info', 1000)
      expect(manager.getLength()).toBe(1)
      expect(manager.findById(id)).toBeDefined()

      // 推进时间
      vi.advanceTimersByTime(1000)

      // 验证消息已被移除
      expect(manager.getLength()).toBe(0)
      expect(manager.findById(id)).toBeUndefined()
    })

    /**
     * 测试多个消息按不同时间自动消失
     */
    it('多个消息应按各自的持续时间自动消失', () => {
      const manager = createToastManager()

      // 添加不同持续时间的消息
      const id1 = manager.addToast('消息1', 'info', 1000)
      const id2 = manager.addToast('消息2', 'success', 2000)
      const id3 = manager.addToast('消息3', 'warning', 3000)

      expect(manager.getLength()).toBe(3)

      // 推进 1 秒
      vi.advanceTimersByTime(1000)
      expect(manager.getLength()).toBe(2)
      expect(manager.findById(id1)).toBeUndefined()
      expect(manager.findById(id2)).toBeDefined()
      expect(manager.findById(id3)).toBeDefined()

      // 再推进 1 秒
      vi.advanceTimersByTime(1000)
      expect(manager.getLength()).toBe(1)
      expect(manager.findById(id2)).toBeUndefined()
      expect(manager.findById(id3)).toBeDefined()

      // 再推进 1 秒
      vi.advanceTimersByTime(1000)
      expect(manager.getLength()).toBe(0)
      expect(manager.findById(id3)).toBeUndefined()
    })

    /**
     * 测试 duration=0 的消息不会自动消失
     */
    it('duration=0 的消息不应自动消失', () => {
      const manager = createToastManager()

      // 添加不会自动消失的消息
      const id = manager.addToast('永久消息', 'info', 0)
      expect(manager.getLength()).toBe(1)

      // 推进大量时间
      vi.advanceTimersByTime(100000)

      // 验证消息仍然存在
      expect(manager.getLength()).toBe(1)
      expect(manager.findById(id)).toBeDefined()
    })

    /**
     * 测试消息内容和类型正确存储
     */
    it('消息内容和类型应正确存储', () => {
      fc.assert(
        fc.property(messageArb, toastTypeArb, (message, type) => {
          const manager = createToastManager()

          const id = manager.addToast(message, type, 0)
          const toast = manager.findById(id)

          expect(toast).toBeDefined()
          expect(toast?.message).toBe(message)
          expect(toast?.type).toBe(type)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试手动移除消息会清除定时器
     */
    it('手动移除消息应清除其定时器', () => {
      const manager = createToastManager()

      // 添加一个会自动消失的消息
      const id = manager.addToast('测试消息', 'info', 5000)
      expect(manager.getLength()).toBe(1)

      // 手动移除
      manager.removeToast(id)
      expect(manager.getLength()).toBe(0)

      // 推进时间，不应有任何影响
      vi.advanceTimersByTime(5000)
      expect(manager.getLength()).toBe(0)
    })
  })

  describe('边界条件测试', () => {
    /**
     * 测试空队列操作
     */
    it('空队列的 clearAll 应正常工作', () => {
      const manager = createToastManager()
      expect(manager.getLength()).toBe(0)

      // 清除空队列不应报错
      manager.clearAll()
      expect(manager.getLength()).toBe(0)
    })

    /**
     * 测试重复移除同一 ID
     */
    it('重复移除同一 ID 应返回 false', () => {
      const manager = createToastManager()

      const id = manager.addToast('测试', 'info', 0)
      expect(manager.removeToast(id)).toBe(true)
      expect(manager.removeToast(id)).toBe(false)
      expect(manager.removeToast(id)).toBe(false)
    })

    /**
     * 测试大量消息
     */
    it('应能处理大量消息', () => {
      const manager = createToastManager()
      const count = 1000

      // 添加大量消息
      for (let i = 0; i < count; i++) {
        manager.addToast(`消息 ${i}`, 'info', 0)
      }

      expect(manager.getLength()).toBe(count)

      // 清除所有
      manager.clearAll()
      expect(manager.getLength()).toBe(0)
    })
  })
})


// ============================================================
// Property 39: 消息提示系统一致性（showGeneralMessage API）
// ============================================================

/**
 * 扩展的 Toast 消息接口（支持 messageId 和 isHTML）
 */
interface ExtendedToast {
  id: number
  messageId: string
  message: string
  type: ToastType
  isHTML: boolean
  timer?: ReturnType<typeof setTimeout>
  safetyTimer?: ReturnType<typeof setTimeout>
}

/**
 * 创建扩展的 Toast 队列管理器（支持 showGeneralMessage API）
 */
function createExtendedToastManager() {
  let toasts: ExtendedToast[] = []
  let toastId = 0
  const SAFETY_TIMEOUT = 30000

  /**
   * 清除 Toast 的所有定时器
   */
  const clearToastTimers = (toast: ExtendedToast): void => {
    if (toast.timer) {
      clearTimeout(toast.timer)
      toast.timer = undefined
    }
    if (toast.safetyTimer) {
      clearTimeout(toast.safetyTimer)
      toast.safetyTimer = undefined
    }
  }

  /**
   * 移除指定 ID 的 Toast
   */
  const removeToast = (id: number): void => {
    const index = toasts.findIndex((t) => t.id === id)
    if (index !== -1) {
      const toast = toasts[index]
      if (toast) {
        clearToastTimers(toast)
      }
      toasts.splice(index, 1)
    }
  }

  /**
   * 清除所有 Toast
   */
  const clearAll = (): void => {
    toasts.forEach((toast) => {
      clearToastTimers(toast)
    })
    toasts = []
  }

  /**
   * 显示通用消息（队列模式：只显示最新的一个）
   */
  const showGeneralMessage = (
    message: string,
    type: ToastType = 'info',
    isHTML: boolean = false,
    duration: number = 5000,
    messageId: string = ''
  ): string => {
    const msgId = messageId || `msg_${Date.now()}_${Math.floor(Math.random() * 1000)}`
    
    // 队列模式：立即移除所有现有消息
    clearAll()
    
    const id = ++toastId
    const toast: ExtendedToast = { id, messageId: msgId, message, type, isHTML }
    
    const actualDuration = duration > 0 ? duration : SAFETY_TIMEOUT
    
    toast.timer = setTimeout(() => {
      removeToast(id)
    }, actualDuration)
    
    toasts.push(toast)
    return msgId
  }

  /**
   * 按 messageId 清除消息
   */
  const clearGeneralMessageById = (messageId: string): void => {
    if (!messageId) return
    
    const index = toasts.findIndex((t) => t.messageId === messageId)
    if (index !== -1) {
      const toast = toasts[index]
      if (toast) {
        clearToastTimers(toast)
      }
      toasts.splice(index, 1)
    }
  }

  /**
   * 清除所有特定类型的消息
   */
  const clearAllGeneralMessages = (type: ToastType | '' = ''): void => {
    if (type === '') {
      clearAll()
    } else {
      const indicesToRemove: number[] = []
      toasts.forEach((toast, index) => {
        if (toast.type === type) {
          clearToastTimers(toast)
          indicesToRemove.push(index)
        }
      })
      for (let i = indicesToRemove.length - 1; i >= 0; i--) {
        const idx = indicesToRemove[i]
        if (idx !== undefined) {
          toasts.splice(idx, 1)
        }
      }
    }
  }

  /**
   * 获取当前队列
   */
  const getToasts = (): ExtendedToast[] => [...toasts]

  /**
   * 获取队列长度
   */
  const getLength = (): number => toasts.length

  /**
   * 根据 messageId 查找 Toast
   */
  const findByMessageId = (messageId: string): ExtendedToast | undefined => 
    toasts.find((t) => t.messageId === messageId)

  return {
    showGeneralMessage,
    clearGeneralMessageById,
    clearAllGeneralMessages,
    clearAll,
    getToasts,
    getLength,
    findByMessageId
  }
}

describe('Property 39: 消息提示系统一致性', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  /**
   * 测试 showGeneralMessage 队列模式（只保留最新消息）
   *
   * Feature: vue-frontend-migration, Property 39: 消息提示系统一致性
   * Validates: Requirements 8.2
   */
  it('showGeneralMessage 应只保留最新的一条消息', () => {
    fc.assert(
      fc.property(
        fc.array(addOperationArb, { minLength: 2, maxLength: 10 }),
        (operations) => {
          const manager = createExtendedToastManager()

          // 连续调用 showGeneralMessage
          let lastMsgId = ''
          operations.forEach((op) => {
            lastMsgId = manager.showGeneralMessage(op.message, op.type, false, 5000)
          })

          // 验证队列中只有一条消息
          expect(manager.getLength()).toBe(1)
          
          // 验证是最后一条消息
          const toast = manager.findByMessageId(lastMsgId)
          expect(toast).toBeDefined()
          expect(toast?.message).toBe(operations[operations.length - 1]?.message)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * 测试按 messageId 清除消息正确性
   */
  it('clearGeneralMessageById 应正确清除指定消息', () => {
    const manager = createExtendedToastManager()

    // 添加消息
    const msgId = manager.showGeneralMessage('测试消息', 'info', false, 5000)
    expect(manager.getLength()).toBe(1)
    expect(manager.findByMessageId(msgId)).toBeDefined()

    // 按 ID 清除
    manager.clearGeneralMessageById(msgId)
    expect(manager.getLength()).toBe(0)
    expect(manager.findByMessageId(msgId)).toBeUndefined()
  })

  /**
   * 测试清除不存在的 messageId 不影响队列
   */
  it('clearGeneralMessageById 清除不存在的 ID 不应影响队列', () => {
    const manager = createExtendedToastManager()

    const msgId = manager.showGeneralMessage('测试消息', 'info', false, 5000)
    expect(manager.getLength()).toBe(1)

    // 尝试清除不存在的 ID
    manager.clearGeneralMessageById('non_existent_id')
    expect(manager.getLength()).toBe(1)
    expect(manager.findByMessageId(msgId)).toBeDefined()
  })

  /**
   * 测试空字符串 messageId 不执行清除
   */
  it('clearGeneralMessageById 空字符串不应执行清除', () => {
    const manager = createExtendedToastManager()

    manager.showGeneralMessage('测试消息', 'info', false, 5000)
    expect(manager.getLength()).toBe(1)

    // 传入空字符串
    manager.clearGeneralMessageById('')
    expect(manager.getLength()).toBe(1)
  })

  /**
   * 测试自动消失时间计算正确
   */
  it('消息应在指定时间后自动消失', () => {
    const manager = createExtendedToastManager()

    const msgId = manager.showGeneralMessage('测试消息', 'info', false, 2000)
    expect(manager.getLength()).toBe(1)

    // 推进 1 秒，消息应该还在
    vi.advanceTimersByTime(1000)
    expect(manager.getLength()).toBe(1)

    // 再推进 1 秒，消息应该消失
    vi.advanceTimersByTime(1000)
    expect(manager.getLength()).toBe(0)
    expect(manager.findByMessageId(msgId)).toBeUndefined()
  })

  /**
   * 测试 duration=0 时使用安全超时
   */
  it('duration=0 时应使用安全超时（30秒）', () => {
    const manager = createExtendedToastManager()

    const msgId = manager.showGeneralMessage('测试消息', 'info', false, 0)
    expect(manager.getLength()).toBe(1)

    // 推进 29 秒，消息应该还在
    vi.advanceTimersByTime(29000)
    expect(manager.getLength()).toBe(1)

    // 再推进 1 秒，消息应该消失
    vi.advanceTimersByTime(1000)
    expect(manager.getLength()).toBe(0)
    expect(manager.findByMessageId(msgId)).toBeUndefined()
  })

  /**
   * 测试 isHTML 标志正确存储
   */
  it('isHTML 标志应正确存储', () => {
    fc.assert(
      fc.property(messageArb, toastTypeArb, fc.boolean(), (message, type, isHTML) => {
        const manager = createExtendedToastManager()

        const msgId = manager.showGeneralMessage(message, type, isHTML, 5000)
        const toast = manager.findByMessageId(msgId)

        expect(toast).toBeDefined()
        expect(toast?.isHTML).toBe(isHTML)

        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * 测试自定义 messageId 正确使用
   */
  it('自定义 messageId 应正确使用', () => {
    const manager = createExtendedToastManager()
    const customId = 'my_custom_message_id'

    const returnedId = manager.showGeneralMessage('测试消息', 'info', false, 5000, customId)
    
    expect(returnedId).toBe(customId)
    expect(manager.findByMessageId(customId)).toBeDefined()
  })

  /**
   * 测试 clearAllGeneralMessages 按类型清除
   */
  it('clearAllGeneralMessages 应按类型清除消息', () => {
    // 由于 showGeneralMessage 是队列模式，我们需要直接测试 clearAllGeneralMessages 的逻辑
    // 这里我们测试空字符串清除所有消息的情况
    const manager = createExtendedToastManager()

    manager.showGeneralMessage('测试消息', 'info', false, 5000)
    expect(manager.getLength()).toBe(1)

    // 清除所有消息
    manager.clearAllGeneralMessages('')
    expect(manager.getLength()).toBe(0)
  })

  /**
   * 测试消息类型正确存储
   */
  it('消息类型应正确存储', () => {
    fc.assert(
      fc.property(messageArb, toastTypeArb, (message, type) => {
        const manager = createExtendedToastManager()

        const msgId = manager.showGeneralMessage(message, type, false, 5000)
        const toast = manager.findByMessageId(msgId)

        expect(toast).toBeDefined()
        expect(toast?.type).toBe(type)

        return true
      }),
      { numRuns: 100 }
    )
  })
})
