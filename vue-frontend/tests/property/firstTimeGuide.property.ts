/**
 * 首次使用引导状态属性测试
 * 
 * Property 46: 首次使用引导状态一致性
 * - 测试"不再显示"状态正确持久化到 localStorage
 * - 测试引导弹窗显示条件判断正确
 * 
 * **Validates: Requirements 31.1**
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 常量定义
// ============================================================

/** localStorage 存储键 */
const GUIDE_SHOWN_KEY = 'first_time_guide_shown'
const DISMISS_SETUP_REMINDER_KEY = 'saber_translator_dismiss_setup_reminder'

// ============================================================
// 模拟 localStorage
// ============================================================

/**
 * 创建模拟的 localStorage
 */
function createMockLocalStorage(): Storage {
  let store: Record<string, string> = {}
  
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
    get length() { return Object.keys(store).length },
    key: (index: number) => Object.keys(store)[index] ?? null
  }
}

// ============================================================
// 引导状态管理逻辑（从组件中提取）
// ============================================================

/**
 * 首次使用引导状态管理器
 */
class FirstTimeGuideManager {
  private storage: Storage
  
  constructor(storage: Storage) {
    this.storage = storage
  }
  
  /**
   * 检查是否应该显示引导
   * @returns 是否应该显示引导弹窗
   */
  shouldShowGuide(): boolean {
    const hasShown = this.storage.getItem(GUIDE_SHOWN_KEY)
    return !hasShown
  }
  
  /**
   * 标记引导已显示
   */
  markGuideShown(): void {
    this.storage.setItem(GUIDE_SHOWN_KEY, 'true')
  }
  
  /**
   * 重置引导状态
   */
  resetGuideState(): void {
    this.storage.removeItem(GUIDE_SHOWN_KEY)
  }
  
  /**
   * 检查设置提醒是否已关闭
   * @returns 是否已关闭设置提醒
   */
  isSetupReminderDismissed(): boolean {
    return this.storage.getItem(DISMISS_SETUP_REMINDER_KEY) === 'true'
  }
  
  /**
   * 关闭设置提醒
   * @param permanent 是否永久关闭
   */
  dismissSetupReminder(permanent: boolean): void {
    if (permanent) {
      this.storage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
    }
  }
  
  /**
   * 重置设置提醒状态
   */
  resetSetupReminder(): void {
    this.storage.removeItem(DISMISS_SETUP_REMINDER_KEY)
  }
}

// ============================================================
// 属性测试
// ============================================================

describe('Property 46: 首次使用引导状态一致性', () => {
  let mockStorage: Storage
  let manager: FirstTimeGuideManager
  
  beforeEach(() => {
    mockStorage = createMockLocalStorage()
    manager = new FirstTimeGuideManager(mockStorage)
  })
  
  afterEach(() => {
    mockStorage.clear()
  })
  
  describe('引导弹窗显示条件', () => {
    it('首次访问时应该显示引导弹窗', () => {
      // 首次访问，localStorage 为空
      expect(manager.shouldShowGuide()).toBe(true)
    })
    
    it('标记已显示后不应该再显示引导弹窗', () => {
      // 标记已显示
      manager.markGuideShown()
      
      // 不应该再显示
      expect(manager.shouldShowGuide()).toBe(false)
    })
    
    it('重置状态后应该再次显示引导弹窗', () => {
      // 标记已显示
      manager.markGuideShown()
      expect(manager.shouldShowGuide()).toBe(false)
      
      // 重置状态
      manager.resetGuideState()
      
      // 应该再次显示
      expect(manager.shouldShowGuide()).toBe(true)
    })
  })
  
  describe('"不再显示"状态持久化', () => {
    it('Property: 对于任意布尔值，永久关闭设置应该正确持久化', () => {
      fc.assert(
        fc.property(
          fc.boolean(),
          (permanent) => {
            // 重置状态
            manager.resetSetupReminder()
            
            // 关闭设置提醒
            manager.dismissSetupReminder(permanent)
            
            // 验证持久化状态
            if (permanent) {
              expect(manager.isSetupReminderDismissed()).toBe(true)
              expect(mockStorage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBe('true')
            } else {
              expect(manager.isSetupReminderDismissed()).toBe(false)
              expect(mockStorage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBeNull()
            }
          }
        ),
        { numRuns: 100 }
      )
    })
    
    it('Property: 重置后状态应该恢复为未关闭', () => {
      fc.assert(
        fc.property(
          fc.boolean(),
          (permanent) => {
            // 先关闭
            manager.dismissSetupReminder(permanent)
            
            // 重置
            manager.resetSetupReminder()
            
            // 验证状态恢复
            expect(manager.isSetupReminderDismissed()).toBe(false)
            expect(mockStorage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBeNull()
          }
        ),
        { numRuns: 100 }
      )
    })
  })
  
  describe('引导状态往返一致性', () => {
    it('Property: 标记显示后重置应该恢复初始状态', () => {
      fc.assert(
        fc.property(
          fc.nat(10), // 重复次数
          (repeatCount) => {
            // 初始状态应该显示引导
            expect(manager.shouldShowGuide()).toBe(true)
            
            // 重复标记和重置
            for (let i = 0; i < repeatCount; i++) {
              manager.markGuideShown()
              expect(manager.shouldShowGuide()).toBe(false)
              
              manager.resetGuideState()
              expect(manager.shouldShowGuide()).toBe(true)
            }
          }
        ),
        { numRuns: 50 }
      )
    })
  })
  
  describe('多次操作一致性', () => {
    it('Property: 多次标记显示应该是幂等的', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 20 }), // 至少执行1次，确保操作被执行
          (times) => {
            // 重置状态，确保每次测试从初始状态开始
            manager.resetGuideState()
            
            // 多次标记显示
            for (let i = 0; i < times; i++) {
              manager.markGuideShown()
            }
            
            // 状态应该一致
            expect(manager.shouldShowGuide()).toBe(false)
            expect(mockStorage.getItem(GUIDE_SHOWN_KEY)).toBe('true')
          }
        ),
        { numRuns: 50 }
      )
    })
    
    it('Property: 多次永久关闭应该是幂等的', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 20 }), // 至少执行1次，确保操作被执行
          (times) => {
            // 重置状态，确保每次测试从初始状态开始
            manager.resetSetupReminder()
            
            // 多次永久关闭
            for (let i = 0; i < times; i++) {
              manager.dismissSetupReminder(true)
            }
            
            // 状态应该一致
            expect(manager.isSetupReminderDismissed()).toBe(true)
            expect(mockStorage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBe('true')
          }
        ),
        { numRuns: 50 }
      )
    })
  })
  
  describe('边界情况', () => {
    it('localStorage 中存在无效值时应该正确处理', () => {
      // 设置无效值
      mockStorage.setItem(GUIDE_SHOWN_KEY, 'invalid')
      
      // 任何非空值都应该被视为已显示
      expect(manager.shouldShowGuide()).toBe(false)
    })
    
    it('设置提醒状态只有 "true" 才被视为已关闭', () => {
      // 设置各种值
      const testValues = ['false', '0', 'yes', 'no', '']
      
      for (const value of testValues) {
        mockStorage.setItem(DISMISS_SETUP_REMINDER_KEY, value)
        expect(manager.isSetupReminderDismissed()).toBe(false)
      }
      
      // 只有 'true' 才被视为已关闭
      mockStorage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
      expect(manager.isSetupReminderDismissed()).toBe(true)
    })
  })
})

describe('引导弹窗显示逻辑', () => {
  it('Property: 显示条件与 localStorage 状态一致', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        (hasShownBefore) => {
          const mockStorage = createMockLocalStorage()
          const manager = new FirstTimeGuideManager(mockStorage)
          
          if (hasShownBefore) {
            mockStorage.setItem(GUIDE_SHOWN_KEY, 'true')
          }
          
          // 验证显示条件
          expect(manager.shouldShowGuide()).toBe(!hasShownBefore)
        }
      ),
      { numRuns: 100 }
    )
  })
})
