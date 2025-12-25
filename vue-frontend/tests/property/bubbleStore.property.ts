/**
 * 气泡状态管理属性测试
 * 
 * Feature: vue-frontend-migration
 * Property 6: 气泡状态序列化往返一致性
 * Property 7: 气泡多选状态一致性
 * 
 * Validates: Requirements 30.4, 37.1, 37.2
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import {
  useBubbleStore,
  createBubbleState,
  cloneBubbleStates,
  isValidBubbleState
} from '@/stores/bubbleStore'
import type { BubbleState, BubbleCoords, TextDirection, InpaintMethod } from '@/types/bubble'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的气泡坐标
 */
const bubbleCoordsArb = fc.tuple(
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 }),
  fc.integer({ min: 0, max: 1000 })
).map(([x1, y1, x2, y2]): BubbleCoords => {
  // 确保 x2 > x1 且 y2 > y1
  const minX = Math.min(x1, x2)
  const maxX = Math.max(x1, x2) + 1
  const minY = Math.min(y1, y2)
  const maxY = Math.max(y1, y2) + 1
  return [minX, minY, maxX, maxY]
})

/**
 * 生成有效的文本方向
 */
const textDirectionArb: fc.Arbitrary<TextDirection> = fc.constantFrom(
  'vertical',
  'horizontal',
  'auto'
)

/**
 * 生成有效的修复方式
 */
const inpaintMethodArb: fc.Arbitrary<InpaintMethod> = fc.constantFrom(
  'solid',
  'lama_mpe',
  'litelama'
)

/**
 * 生成有效的颜色值（十六进制）
 */
const colorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map(hex => `#${hex}`)

/**
 * 生成有效的气泡状态
 */
const bubbleStateArb: fc.Arbitrary<BubbleState> = fc.record({
  originalText: fc.string({ maxLength: 500 }),
  translatedText: fc.string({ maxLength: 500 }),
  textboxText: fc.string({ maxLength: 500 }),
  coords: bubbleCoordsArb,
  polygon: fc.array(fc.tuple(fc.integer(), fc.integer()).map(([x, y]) => [x, y]), { maxLength: 10 }),
  fontSize: fc.integer({ min: 8, max: 72 }),
  fontFamily: fc.constantFrom('fonts/STSONG.TTF', 'fonts/msyh.ttc', 'Arial'),
  textDirection: textDirectionArb,
  autoTextDirection: fc.constantFrom('vertical', 'horizontal') as fc.Arbitrary<TextDirection>,
  textColor: colorArb,
  fillColor: colorArb,
  rotationAngle: fc.integer({ min: -180, max: 180 }),
  position: fc.record({
    x: fc.integer({ min: -100, max: 100 }),
    y: fc.integer({ min: -100, max: 100 })
  }),
  strokeEnabled: fc.boolean(),
  strokeColor: colorArb,
  strokeWidth: fc.integer({ min: 1, max: 10 }),
  inpaintMethod: inpaintMethodArb
})

/**
 * 生成气泡状态数组
 */
const bubbleStatesArb = fc.array(bubbleStateArb, { minLength: 0, maxLength: 20 })

// ============================================================
// 属性测试
// ============================================================

describe('气泡状态管理属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 6: 气泡状态序列化往返一致性
   * 
   * 对于任意有效的气泡状态数组，序列化后再反序列化应当得到等价的气泡状态数组。
   * 
   * Feature: vue-frontend-migration, Property 6: 气泡状态序列化往返一致性
   * Validates: Requirements 30.4
   */
  it('气泡状态序列化往返一致性', () => {
    fc.assert(
      fc.property(bubbleStatesArb, (states) => {
        const store = useBubbleStore()
        
        // 设置初始状态
        store.setBubbles(states)
        
        // 序列化
        const serialized = store.serialize()
        
        // 清空状态
        store.clearBubbles()
        expect(store.bubbles.length).toBe(0)
        
        // 反序列化
        const success = store.deserialize(serialized)
        expect(success).toBe(true)
        
        // 验证数量一致
        expect(store.bubbles.length).toBe(states.length)
        
        // 验证每个气泡的关键属性一致
        for (let i = 0; i < states.length; i++) {
          const original = states[i]
          const restored = store.bubbles[i]
          
          if (!original || !restored) continue
          
          expect(restored.originalText).toBe(original.originalText)
          expect(restored.translatedText).toBe(original.translatedText)
          expect(restored.textboxText).toBe(original.textboxText)
          expect(restored.coords).toEqual(original.coords)
          expect(restored.fontSize).toBe(original.fontSize)
          expect(restored.fontFamily).toBe(original.fontFamily)
          expect(restored.textDirection).toBe(original.textDirection)
          expect(restored.textColor).toBe(original.textColor)
          expect(restored.fillColor).toBe(original.fillColor)
          expect(restored.rotationAngle).toBe(original.rotationAngle)
          expect(restored.strokeEnabled).toBe(original.strokeEnabled)
          expect(restored.strokeColor).toBe(original.strokeColor)
          expect(restored.strokeWidth).toBe(original.strokeWidth)
          expect(restored.inpaintMethod).toBe(original.inpaintMethod)
        }
        
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * Property 7: 气泡多选状态一致性
   * 
   * 对于任意气泡列表和选择操作序列，selectedIndices 数组应当准确反映所有被选中的气泡索引，
   * 且不包含重复项。
   * 
   * Feature: vue-frontend-migration, Property 7: 气泡多选状态一致性
   * Validates: Requirements 37.1, 37.2
   */
  it('气泡多选状态一致性', () => {
    fc.assert(
      fc.property(
        // 生成气泡数组和选择操作序列
        fc.tuple(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 10 }),
          fc.array(fc.integer({ min: 0, max: 9 }), { minLength: 1, maxLength: 20 })
        ),
        ([coordsList, selectOperations]) => {
          const store = useBubbleStore()
          
          // 创建气泡
          const bubbles = coordsList.map(coords => createBubbleState(coords))
          store.setBubbles(bubbles)
          
          // 执行多选操作
          const expectedSelected = new Set<number>()
          
          for (const index of selectOperations) {
            if (index < store.bubbles.length) {
              store.toggleMultiSelect(index)
              
              // 更新期望的选中集合
              if (expectedSelected.has(index)) {
                expectedSelected.delete(index)
              } else {
                expectedSelected.add(index)
              }
            }
          }
          
          // 验证 selectedIndices 不包含重复项
          const uniqueIndices = new Set(store.selectedIndices)
          expect(uniqueIndices.size).toBe(store.selectedIndices.length)
          
          // 验证 selectedIndices 与期望一致
          expect(new Set(store.selectedIndices)).toEqual(expectedSelected)
          
          // 验证所有索引都在有效范围内
          for (const idx of store.selectedIndices) {
            expect(idx).toBeGreaterThanOrEqual(0)
            expect(idx).toBeLessThan(store.bubbles.length)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * 深拷贝一致性测试
   * 
   * 验证 cloneBubbleStates 函数能正确深拷贝气泡状态数组
   */
  it('深拷贝应当创建独立的副本', () => {
    fc.assert(
      fc.property(bubbleStatesArb, (states) => {
        const cloned = cloneBubbleStates(states)
        
        // 验证数量一致
        expect(cloned.length).toBe(states.length)
        
        // 验证内容一致但引用不同
        for (let i = 0; i < states.length; i++) {
          const original = states[i]
          const copy = cloned[i]
          
          if (!original || !copy) continue
          
          // 内容应该相等
          expect(copy.originalText).toBe(original.originalText)
          expect(copy.coords).toEqual(original.coords)
          
          // 但引用应该不同
          expect(copy).not.toBe(original)
          expect(copy.coords).not.toBe(original.coords)
          expect(copy.position).not.toBe(original.position)
        }
        
        // 修改克隆不应影响原始数据
        if (cloned.length > 0 && cloned[0]) {
          cloned[0].originalText = '修改后的文本'
          if (states[0]) {
            expect(states[0].originalText).not.toBe('修改后的文本')
          }
        }
        
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * 气泡状态验证函数测试
   */
  it('isValidBubbleState 应当正确验证气泡状态', () => {
    fc.assert(
      fc.property(bubbleStateArb, (state) => {
        // 有效的气泡状态应该通过验证
        expect(isValidBubbleState(state)).toBe(true)
        
        return true
      }),
      { numRuns: 100 }
    )
  })

  /**
   * 无效数据应当被拒绝
   */
  it('无效数据应当被 isValidBubbleState 拒绝', () => {
    // null 和 undefined
    expect(isValidBubbleState(null)).toBe(false)
    expect(isValidBubbleState(undefined)).toBe(false)
    
    // 非对象
    expect(isValidBubbleState('string')).toBe(false)
    expect(isValidBubbleState(123)).toBe(false)
    expect(isValidBubbleState([])).toBe(false)
    
    // 缺少必要字段
    expect(isValidBubbleState({})).toBe(false)
    expect(isValidBubbleState({ coords: [0, 0, 100, 100] })).toBe(false)
    expect(isValidBubbleState({ originalText: 'test' })).toBe(false)
    
    // coords 格式错误
    expect(isValidBubbleState({
      coords: [0, 0, 100], // 只有3个元素
      originalText: 'test',
      translatedText: 'test'
    })).toBe(false)
  })

  /**
   * 删除气泡后索引调整测试
   */
  it('删除气泡后选中索引应当正确调整', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.array(bubbleCoordsArb, { minLength: 3, maxLength: 10 }),
          fc.integer({ min: 0, max: 9 })
        ),
        ([coordsList, deleteIndex]) => {
          const store = useBubbleStore()
          
          // 创建气泡
          const bubbles = coordsList.map(coords => createBubbleState(coords))
          store.setBubbles(bubbles)
          
          const originalLength = store.bubbles.length
          
          // 选择一个气泡
          const selectIndex = Math.min(deleteIndex + 1, originalLength - 1)
          if (selectIndex < originalLength) {
            store.selectBubble(selectIndex)
          }
          
          // 删除一个气泡
          const actualDeleteIndex = Math.min(deleteIndex, originalLength - 1)
          if (actualDeleteIndex < originalLength) {
            store.deleteBubble(actualDeleteIndex)
            
            // 验证长度减少
            expect(store.bubbles.length).toBe(originalLength - 1)
            
            // 验证选中索引在有效范围内
            if (store.selectedIndex >= 0) {
              expect(store.selectedIndex).toBeLessThan(store.bubbles.length)
            }
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
