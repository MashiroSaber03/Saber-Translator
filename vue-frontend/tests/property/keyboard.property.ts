/**
 * 快捷键系统属性测试
 * 
 * Feature: vue-frontend-migration
 * Property 10: 快捷键事件处理一致性
 * 
 * Validates: Requirements 19.1-19.7
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 类型定义
// ============================================================

interface KeyboardHandler {
  key: string | string[]
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  handler: (event: KeyboardEvent) => void
  description?: string
  preventDefault?: boolean
  allowInInput?: boolean
}

interface MockKeyboardEvent {
  key: string
  ctrlKey: boolean
  shiftKey: boolean
  altKey: boolean
  target: { tagName: string; isContentEditable: boolean }
  preventDefault: () => void
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 检查是否在输入框中
 */
function isInInputElement(event: MockKeyboardEvent): boolean {
  return (
    event.target.tagName === 'INPUT' ||
    event.target.tagName === 'TEXTAREA' ||
    event.target.isContentEditable
  )
}

/**
 * 检查按键是否匹配
 */
function matchKey(handler: KeyboardHandler, event: MockKeyboardEvent): boolean {
  // 检查修饰键
  if (handler.ctrl && !event.ctrlKey) return false
  if (handler.shift && !event.shiftKey) return false
  if (handler.alt && !event.altKey) return false

  // 如果没有指定修饰键，但按下了修饰键，则不匹配
  if (!handler.ctrl && event.ctrlKey) return false
  if (!handler.shift && event.shiftKey) return false
  if (!handler.alt && event.altKey) return false

  // 检查按键
  const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
  return keys.some(k => k.toLowerCase() === event.key.toLowerCase())
}

/**
 * 处理键盘事件
 */
function handleKeyboardEvent(
  handlers: KeyboardHandler[],
  event: MockKeyboardEvent,
  enabled: boolean
): { handled: boolean; handlerIndex: number } {
  if (!enabled) {
    return { handled: false, handlerIndex: -1 }
  }

  const inInput = isInInputElement(event)

  for (let i = 0; i < handlers.length; i++) {
    const handler = handlers[i]
    
    // 如果在输入框中且不允许，跳过
    if (inInput && !handler.allowInInput) continue

    // 检查是否匹配
    if (matchKey(handler, event)) {
      handler.handler(event as unknown as KeyboardEvent)
      
      if (handler.preventDefault !== false) {
        event.preventDefault()
      }
      
      return { handled: true, handlerIndex: i }
    }
  }

  return { handled: false, handlerIndex: -1 }
}

/**
 * 格式化快捷键组合
 */
function formatKeyCombo(handler: KeyboardHandler): string {
  const parts: string[] = []
  if (handler.ctrl) parts.push('Ctrl')
  if (handler.shift) parts.push('Shift')
  if (handler.alt) parts.push('Alt')
  
  const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
  parts.push(keys.join('/'))
  
  return parts.join('+')
}

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成按键名称
 */
const keyNameArb = fc.constantFrom(
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'Enter', 'Escape', 'Delete', 'Backspace', 'Tab', 'Space',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'PageUp', 'PageDown', 'Home', 'End',
  '+', '-', '=', '[', ']', '\\', ';', "'", ',', '.', '/'
)

/**
 * 生成修饰键组合
 */
const modifiersArb = fc.record({
  ctrl: fc.boolean(),
  shift: fc.boolean(),
  alt: fc.boolean()
})

/**
 * 生成目标元素类型
 */
const targetElementArb = fc.constantFrom(
  { tagName: 'DIV', isContentEditable: false },
  { tagName: 'BUTTON', isContentEditable: false },
  { tagName: 'INPUT', isContentEditable: false },
  { tagName: 'TEXTAREA', isContentEditable: false },
  { tagName: 'DIV', isContentEditable: true }
)

/**
 * 生成模拟键盘事件
 */
const mockKeyboardEventArb = fc.record({
  key: keyNameArb,
  ctrlKey: fc.boolean(),
  shiftKey: fc.boolean(),
  altKey: fc.boolean(),
  target: targetElementArb
}).map(data => ({
  ...data,
  preventDefault: vi.fn()
}))

/**
 * 生成快捷键处理器
 */
const keyboardHandlerArb = fc.record({
  key: fc.oneof(keyNameArb, fc.array(keyNameArb, { minLength: 1, maxLength: 3 })),
  ctrl: fc.option(fc.boolean(), { nil: undefined }),
  shift: fc.option(fc.boolean(), { nil: undefined }),
  alt: fc.option(fc.boolean(), { nil: undefined }),
  description: fc.option(fc.string(), { nil: undefined }),
  preventDefault: fc.option(fc.boolean(), { nil: undefined }),
  allowInInput: fc.option(fc.boolean(), { nil: undefined })
}).map(data => ({
  ...data,
  handler: vi.fn()
}))

// ============================================================
// 属性测试
// ============================================================

describe('快捷键系统属性测试', () => {
  /**
   * Property 10: 快捷键事件处理一致性
   * 
   * 对于任意快捷键配置和键盘事件，处理结果应当一致且可预测。
   * 
   * Feature: vue-frontend-migration, Property 10: 快捷键事件处理一致性
   * Validates: Requirements 19.1-19.7
   */
  describe('Property 10: 快捷键事件处理一致性', () => {
    it('匹配的快捷键应当触发对应的处理器', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, (handler) => {
          // 创建匹配的事件
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          const event: MockKeyboardEvent = {
            key: keys[0],
            ctrlKey: handler.ctrl || false,
            shiftKey: handler.shift || false,
            altKey: handler.alt || false,
            target: { tagName: 'DIV', isContentEditable: false },
            preventDefault: vi.fn()
          }

          const result = handleKeyboardEvent([handler], event, true)

          // 验证处理器被调用
          expect(result.handled).toBe(true)
          expect(result.handlerIndex).toBe(0)
          expect(handler.handler).toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('不匹配的快捷键不应触发处理器', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, keyNameArb, (handler, differentKey) => {
          // 确保按键不同
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          if (keys.some(k => k.toLowerCase() === differentKey.toLowerCase())) {
            return true // 跳过相同按键的情况
          }

          // 创建不匹配的事件
          const event: MockKeyboardEvent = {
            key: differentKey,
            ctrlKey: handler.ctrl || false,
            shiftKey: handler.shift || false,
            altKey: handler.alt || false,
            target: { tagName: 'DIV', isContentEditable: false },
            preventDefault: vi.fn()
          }

          // 重置 mock
          handler.handler.mockClear()

          const result = handleKeyboardEvent([handler], event, true)

          // 验证处理器未被调用
          expect(result.handled).toBe(false)
          expect(handler.handler).not.toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('修饰键不匹配时不应触发处理器', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, modifiersArb, (handler, differentModifiers) => {
          // 确保修饰键不同
          const handlerCtrl = handler.ctrl || false
          const handlerShift = handler.shift || false
          const handlerAlt = handler.alt || false

          if (
            handlerCtrl === differentModifiers.ctrl &&
            handlerShift === differentModifiers.shift &&
            handlerAlt === differentModifiers.alt
          ) {
            return true // 跳过相同修饰键的情况
          }

          // 创建修饰键不匹配的事件
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          const event: MockKeyboardEvent = {
            key: keys[0],
            ctrlKey: differentModifiers.ctrl,
            shiftKey: differentModifiers.shift,
            altKey: differentModifiers.alt,
            target: { tagName: 'DIV', isContentEditable: false },
            preventDefault: vi.fn()
          }

          // 重置 mock
          handler.handler.mockClear()

          const result = handleKeyboardEvent([handler], event, true)

          // 验证处理器未被调用
          expect(result.handled).toBe(false)
          expect(handler.handler).not.toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('禁用状态下不应触发任何处理器', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, (handler) => {
          // 创建匹配的事件
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          const event: MockKeyboardEvent = {
            key: keys[0],
            ctrlKey: handler.ctrl || false,
            shiftKey: handler.shift || false,
            altKey: handler.alt || false,
            target: { tagName: 'DIV', isContentEditable: false },
            preventDefault: vi.fn()
          }

          // 重置 mock
          handler.handler.mockClear()

          // 禁用状态
          const result = handleKeyboardEvent([handler], event, false)

          // 验证处理器未被调用
          expect(result.handled).toBe(false)
          expect(handler.handler).not.toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('输入框中默认不触发快捷键', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, (handler) => {
          // 确保 allowInInput 为 false 或 undefined
          handler.allowInInput = false

          // 创建匹配的事件，但在输入框中
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          const event: MockKeyboardEvent = {
            key: keys[0],
            ctrlKey: handler.ctrl || false,
            shiftKey: handler.shift || false,
            altKey: handler.alt || false,
            target: { tagName: 'INPUT', isContentEditable: false },
            preventDefault: vi.fn()
          }

          // 重置 mock
          handler.handler.mockClear()

          const result = handleKeyboardEvent([handler], event, true)

          // 验证处理器未被调用
          expect(result.handled).toBe(false)
          expect(handler.handler).not.toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('allowInInput 为 true 时在输入框中也触发', () => {
      fc.assert(
        fc.property(keyboardHandlerArb, (handler) => {
          // 设置 allowInInput 为 true
          handler.allowInInput = true

          // 创建匹配的事件，在输入框中
          const keys = Array.isArray(handler.key) ? handler.key : [handler.key]
          const event: MockKeyboardEvent = {
            key: keys[0],
            ctrlKey: handler.ctrl || false,
            shiftKey: handler.shift || false,
            altKey: handler.alt || false,
            target: { tagName: 'INPUT', isContentEditable: false },
            preventDefault: vi.fn()
          }

          // 重置 mock
          handler.handler.mockClear()

          const result = handleKeyboardEvent([handler], event, true)

          // 验证处理器被调用
          expect(result.handled).toBe(true)
          expect(handler.handler).toHaveBeenCalled()

          return true
        }),
        { numRuns: 50 }
      )
    })

    it('多个处理器时只触发第一个匹配的', () => {
      fc.assert(
        fc.property(
          fc.array(keyboardHandlerArb, { minLength: 2, maxLength: 5 }),
          (handlers) => {
            // 让所有处理器使用相同的按键
            const commonKey = 'a'
            handlers.forEach(h => {
              h.key = commonKey
              h.ctrl = false
              h.shift = false
              h.alt = false
              h.handler.mockClear()
            })

            const event: MockKeyboardEvent = {
              key: commonKey,
              ctrlKey: false,
              shiftKey: false,
              altKey: false,
              target: { tagName: 'DIV', isContentEditable: false },
              preventDefault: vi.fn()
            }

            const result = handleKeyboardEvent(handlers, event, true)

            // 验证只有第一个处理器被调用
            expect(result.handled).toBe(true)
            expect(result.handlerIndex).toBe(0)
            expect(handlers[0].handler).toHaveBeenCalled()
            
            // 其他处理器不应被调用
            for (let i = 1; i < handlers.length; i++) {
              expect(handlers[i].handler).not.toHaveBeenCalled()
            }

            return true
          }
        ),
        { numRuns: 30 }
      )
    })

    it('preventDefault 默认为 true', () => {
      const handler: KeyboardHandler = {
        key: 'a',
        handler: vi.fn()
        // 不设置 preventDefault
      }

      const event: MockKeyboardEvent = {
        key: 'a',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }

      handleKeyboardEvent([handler], event, true)

      // 验证 preventDefault 被调用
      expect(event.preventDefault).toHaveBeenCalled()
    })

    it('preventDefault 为 false 时不阻止默认行为', () => {
      const handler: KeyboardHandler = {
        key: 'a',
        handler: vi.fn(),
        preventDefault: false
      }

      const event: MockKeyboardEvent = {
        key: 'a',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }

      handleKeyboardEvent([handler], event, true)

      // 验证 preventDefault 未被调用
      expect(event.preventDefault).not.toHaveBeenCalled()
    })
  })

  describe('快捷键格式化测试', () => {
    it('格式化单键快捷键', () => {
      const handler: KeyboardHandler = {
        key: 'a',
        handler: vi.fn()
      }
      expect(formatKeyCombo(handler)).toBe('a')
    })

    it('格式化带修饰键的快捷键', () => {
      const handler: KeyboardHandler = {
        key: 'Enter',
        ctrl: true,
        handler: vi.fn()
      }
      expect(formatKeyCombo(handler)).toBe('Ctrl+Enter')
    })

    it('格式化多修饰键快捷键', () => {
      const handler: KeyboardHandler = {
        key: 's',
        ctrl: true,
        shift: true,
        handler: vi.fn()
      }
      expect(formatKeyCombo(handler)).toBe('Ctrl+Shift+s')
    })

    it('格式化多按键快捷键', () => {
      const handler: KeyboardHandler = {
        key: ['a', 'PageUp'],
        handler: vi.fn()
      }
      expect(formatKeyCombo(handler)).toBe('a/PageUp')
    })
  })

  describe('编辑模式快捷键测试', () => {
    it('A/D 键应当切换图片', () => {
      const onPrevious = vi.fn()
      const onNext = vi.fn()

      const handlers: KeyboardHandler[] = [
        { key: ['a', 'PageUp'], handler: onPrevious },
        { key: ['d', 'PageDown'], handler: onNext }
      ]

      // 测试 A 键
      handleKeyboardEvent(handlers, {
        key: 'a',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onPrevious).toHaveBeenCalled()

      // 测试 D 键
      handleKeyboardEvent(handlers, {
        key: 'd',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onNext).toHaveBeenCalled()
    })

    it('方向键应当切换气泡', () => {
      const onPrevious = vi.fn()
      const onNext = vi.fn()

      const handlers: KeyboardHandler[] = [
        { key: 'ArrowLeft', handler: onPrevious },
        { key: 'ArrowRight', handler: onNext }
      ]

      // 测试左箭头
      handleKeyboardEvent(handlers, {
        key: 'ArrowLeft',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onPrevious).toHaveBeenCalled()

      // 测试右箭头
      handleKeyboardEvent(handlers, {
        key: 'ArrowRight',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onNext).toHaveBeenCalled()
    })

    it('Delete/Backspace 应当删除气泡', () => {
      const onDelete = vi.fn()

      const handlers: KeyboardHandler[] = [
        { key: ['Delete', 'Backspace'], handler: onDelete }
      ]

      // 测试 Delete 键
      handleKeyboardEvent(handlers, {
        key: 'Delete',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onDelete).toHaveBeenCalledTimes(1)

      // 测试 Backspace 键
      handleKeyboardEvent(handlers, {
        key: 'Backspace',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onDelete).toHaveBeenCalledTimes(2)
    })

    it('Ctrl+Enter 应当应用并下一张', () => {
      const onApplyAndNext = vi.fn()

      const handlers: KeyboardHandler[] = [
        { key: 'Enter', ctrl: true, handler: onApplyAndNext }
      ]

      // 测试 Ctrl+Enter
      handleKeyboardEvent(handlers, {
        key: 'Enter',
        ctrlKey: true,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onApplyAndNext).toHaveBeenCalled()

      // 单独 Enter 不应触发
      onApplyAndNext.mockClear()
      handleKeyboardEvent(handlers, {
        key: 'Enter',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onApplyAndNext).not.toHaveBeenCalled()
    })

    it('Escape 应当退出编辑模式', () => {
      const onExit = vi.fn()

      const handlers: KeyboardHandler[] = [
        { key: 'Escape', handler: onExit }
      ]

      handleKeyboardEvent(handlers, {
        key: 'Escape',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        target: { tagName: 'DIV', isContentEditable: false },
        preventDefault: vi.fn()
      }, true)
      expect(onExit).toHaveBeenCalled()
    })
  })
})
