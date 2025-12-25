/**
 * 气泡工厂函数属性测试
 *
 * Feature: vue-frontend-migration
 * Property 11: 气泡状态创建一致性
 *
 * Validates: Requirements 30.1
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import {
  createBubbleState,
  detectTextDirection,
  isValidBubbleState,
  createBubbleStatesFromResponse,
  bubbleStatesToApiRequest,
  updateBubbleState,
  updateAllBubbleStates,
  cloneBubbleStates,
  cloneBubbleState,
  getBubbleCenter,
  getBubbleSize,
  isPointInBubble,
  isPointInPolygon,
  isPointInBubbleArea,
  getDefaultBubbleSettings,
  initBubbleStates
} from '@/utils/bubbleFactory'
import {
  DEFAULT_FILL_COLOR,
  DEFAULT_STROKE_ENABLED,
  DEFAULT_STROKE_COLOR,
  DEFAULT_STROKE_WIDTH
} from '@/constants'
import type { BubbleCoords, TextDirection, InpaintMethod, BubbleState } from '@/types/bubble'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的气泡坐标
 */
const bubbleCoordsArb = fc
  .tuple(
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 }),
    fc.integer({ min: 0, max: 1000 })
  )
  .map(([x1, y1, x2, y2]): BubbleCoords => {
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
const textDirectionArb: fc.Arbitrary<TextDirection> = fc.constantFrom('vertical', 'horizontal', 'auto')

/**
 * 生成有效的修复方式
 */
const inpaintMethodArb: fc.Arbitrary<InpaintMethod> = fc.constantFrom('solid', 'lama_mpe', 'litelama')

/**
 * 生成有效的颜色值（十六进制）
 */
const colorArb = fc.hexaString({ minLength: 6, maxLength: 6 }).map((hex) => `#${hex}`)

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
// Property 11: 气泡状态创建一致性
// ============================================================

describe('气泡工厂函数属性测试', () => {
  describe('Property 11: 气泡状态创建一致性', () => {
    /**
     * 测试 createBubbleState 默认值正确性
     *
     * Feature: vue-frontend-migration, Property 11: 气泡状态创建一致性
     * Validates: Requirements 30.1
     */
    it('createBubbleState 无参数时应返回正确的默认值', () => {
      const state = createBubbleState()

      // 验证文本内容默认值
      expect(state.originalText).toBe('')
      expect(state.translatedText).toBe('')
      expect(state.textboxText).toBe('')

      // 验证坐标默认值
      expect(state.coords).toEqual([0, 0, 100, 100])
      expect(state.polygon).toEqual([])

      // 验证渲染参数默认值
      expect(state.fontSize).toBe(24)
      expect(state.fontFamily).toBe('')
      expect(state.textDirection).toBe('auto')
      expect(state.autoTextDirection).toBe('vertical')
      expect(state.textColor).toBe('#000000')
      expect(state.fillColor).toBe(DEFAULT_FILL_COLOR)
      expect(state.rotationAngle).toBe(0)
      expect(state.position).toEqual({ x: 0, y: 0 })

      // 验证描边参数默认值
      expect(state.strokeEnabled).toBe(DEFAULT_STROKE_ENABLED)
      expect(state.strokeColor).toBe(DEFAULT_STROKE_COLOR)
      expect(state.strokeWidth).toBe(DEFAULT_STROKE_WIDTH)

      // 验证修复参数默认值
      expect(state.inpaintMethod).toBe('solid')
    })

    /**
     * 测试 createBubbleState 覆盖参数正确性
     *
     * Feature: vue-frontend-migration, Property 11: 气泡状态创建一致性
     * Validates: Requirements 30.1
     */
    it('createBubbleState 应正确应用覆盖参数', () => {
      fc.assert(
        fc.property(
          fc.record({
            originalText: fc.string(),
            translatedText: fc.string(),
            coords: bubbleCoordsArb,
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArb,
            strokeEnabled: fc.boolean()
          }),
          (overrides) => {
            const state = createBubbleState(overrides)

            // 验证覆盖的值
            expect(state.originalText).toBe(overrides.originalText)
            expect(state.translatedText).toBe(overrides.translatedText)
            expect(state.coords).toEqual(overrides.coords)
            expect(state.fontSize).toBe(overrides.fontSize)
            expect(state.textColor).toBe(overrides.textColor)
            expect(state.strokeEnabled).toBe(overrides.strokeEnabled)

            // 验证未覆盖的值保持默认
            expect(state.textboxText).toBe('')
            expect(state.fontFamily).toBe('')
            expect(state.inpaintMethod).toBe('solid')

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 createBubbleState 返回的对象是独立的
     *
     * Feature: vue-frontend-migration, Property 11: 气泡状态创建一致性
     * Validates: Requirements 30.1
     */
    it('createBubbleState 每次调用应返回独立的对象', () => {
      const state1 = createBubbleState()
      const state2 = createBubbleState()

      // 验证是不同的对象引用
      expect(state1).not.toBe(state2)
      expect(state1.coords).not.toBe(state2.coords)
      expect(state1.position).not.toBe(state2.position)

      // 修改一个不应影响另一个
      state1.originalText = '修改后的文本'
      expect(state2.originalText).toBe('')
    })
  })

  describe('detectTextDirection 宽高比判断测试', () => {
    /**
     * 测试 detectTextDirection 宽高比判断正确性
     *
     * Feature: vue-frontend-migration, Property 11: 气泡状态创建一致性
     * Validates: Requirements 30.1
     */
    it('高度大于宽度时应返回 vertical', () => {
      fc.assert(
        fc.property(
          fc.tuple(
            fc.integer({ min: 0, max: 500 }),
            fc.integer({ min: 0, max: 500 }),
            fc.integer({ min: 1, max: 200 }) // 宽度
          ),
          ([x1, y1, width]) => {
            // 高度 = 宽度 + 1，确保高度大于宽度
            const height = width + 1
            const coords: BubbleCoords = [x1, y1, x1 + width, y1 + height]
            const direction = detectTextDirection(coords)
            expect(direction).toBe('vertical')
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试宽度大于等于高度时应返回 horizontal
     */
    it('宽度大于等于高度时应返回 horizontal', () => {
      fc.assert(
        fc.property(
          fc.tuple(
            fc.integer({ min: 0, max: 500 }),
            fc.integer({ min: 0, max: 500 }),
            fc.integer({ min: 1, max: 200 }) // 高度
          ),
          ([x1, y1, height]) => {
            // 宽度 = 高度，确保宽度大于等于高度
            const width = height
            const coords: BubbleCoords = [x1, y1, x1 + width, y1 + height]
            const direction = detectTextDirection(coords)
            expect(direction).toBe('horizontal')
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试坐标顺序不影响结果
     */
    it('坐标顺序不应影响方向判断', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const [x1, y1, x2, y2] = coords
          // 交换坐标顺序
          const reversedCoords: BubbleCoords = [x2, y2, x1, y1]
          const direction1 = detectTextDirection(coords)
          const direction2 = detectTextDirection(reversedCoords)
          expect(direction1).toBe(direction2)
          return true
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('isValidBubbleState 验证逻辑测试', () => {
    /**
     * 测试 isValidBubbleState 对有效状态的验证
     *
     * Feature: vue-frontend-migration, Property 11: 气泡状态创建一致性
     * Validates: Requirements 30.1
     */
    it('有效的气泡状态应通过验证', () => {
      fc.assert(
        fc.property(bubbleStateArb, (state) => {
          expect(isValidBubbleState(state)).toBe(true)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 createBubbleState 创建的状态应通过验证
     */
    it('createBubbleState 创建的状态应通过验证', () => {
      fc.assert(
        fc.property(
          fc.record({
            coords: bubbleCoordsArb,
            fontSize: fc.integer({ min: 1, max: 100 }),
            textDirection: textDirectionArb,
            inpaintMethod: inpaintMethodArb
          }),
          (overrides) => {
            const state = createBubbleState(overrides)
            expect(isValidBubbleState(state)).toBe(true)
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试无效数据应被拒绝
     */
    it('无效数据应被拒绝', () => {
      // null 和 undefined
      expect(isValidBubbleState(null)).toBe(false)
      expect(isValidBubbleState(undefined)).toBe(false)

      // 非对象
      expect(isValidBubbleState('string')).toBe(false)
      expect(isValidBubbleState(123)).toBe(false)
      expect(isValidBubbleState([])).toBe(false)

      // 缺少必要字段
      expect(isValidBubbleState({})).toBe(false)

      // coords 格式错误
      expect(
        isValidBubbleState({
          coords: [0, 0, 100], // 只有3个元素
          fontSize: 24,
          textDirection: 'auto',
          inpaintMethod: 'solid'
        })
      ).toBe(false)

      // coords 包含非数字
      expect(
        isValidBubbleState({
          coords: [0, 0, 'invalid', 100],
          fontSize: 24,
          textDirection: 'auto',
          inpaintMethod: 'solid'
        })
      ).toBe(false)

      // fontSize 为负数
      expect(
        isValidBubbleState({
          coords: [0, 0, 100, 100],
          fontSize: -1,
          textDirection: 'auto',
          inpaintMethod: 'solid'
        })
      ).toBe(false)

      // fontSize 为0
      expect(
        isValidBubbleState({
          coords: [0, 0, 100, 100],
          fontSize: 0,
          textDirection: 'auto',
          inpaintMethod: 'solid'
        })
      ).toBe(false)

      // textDirection 无效
      expect(
        isValidBubbleState({
          coords: [0, 0, 100, 100],
          fontSize: 24,
          textDirection: 'invalid',
          inpaintMethod: 'solid'
        })
      ).toBe(false)

      // inpaintMethod 无效
      expect(
        isValidBubbleState({
          coords: [0, 0, 100, 100],
          fontSize: 24,
          textDirection: 'auto',
          inpaintMethod: 'invalid'
        })
      ).toBe(false)
    })
  })

  describe('其他工厂函数测试', () => {
    /**
     * 测试 updateBubbleState 更新逻辑
     */
    it('updateBubbleState 应正确更新气泡状态', () => {
      fc.assert(
        fc.property(
          bubbleStateArb,
          fc.record({
            // 使用非空字符串以避免与原始状态相同
            originalText: fc.string({ minLength: 1 }).map((s) => `updated_${s}`),
            fontSize: fc.integer({ min: 8, max: 72 })
          }),
          (state, updates) => {
            // 确保原始状态的 originalText 与更新值不同
            const originalText = state.originalText
            const updated = updateBubbleState(state, updates)

            // 验证更新的值
            expect(updated.originalText).toBe(updates.originalText)
            expect(updated.fontSize).toBe(updates.fontSize)

            // 验证未更新的值保持不变
            expect(updated.translatedText).toBe(state.translatedText)
            expect(updated.coords).toEqual(state.coords)

            // 验证返回的是新对象
            expect(updated).not.toBe(state)

            // 验证原状态未被修改（通过检查原始值是否保持不变）
            expect(state.originalText).toBe(originalText)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 updateAllBubbleStates 批量更新逻辑
     */
    it('updateAllBubbleStates 应正确批量更新所有气泡', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleStateArb, { minLength: 1, maxLength: 10 }),
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArb
          }),
          (states, updates) => {
            const updatedStates = updateAllBubbleStates(states, updates)

            // 验证数量不变
            expect(updatedStates.length).toBe(states.length)

            // 验证所有气泡都应用了更新
            for (const updated of updatedStates) {
              expect(updated.fontSize).toBe(updates.fontSize)
              expect(updated.textColor).toBe(updates.textColor)
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 cloneBubbleState 深拷贝逻辑
     */
    it('cloneBubbleState 应创建独立的深拷贝', () => {
      fc.assert(
        fc.property(bubbleStateArb, (state) => {
          const cloned = cloneBubbleState(state)

          // 验证内容相等
          expect(cloned.originalText).toBe(state.originalText)
          expect(cloned.coords).toEqual(state.coords)
          expect(cloned.position).toEqual(state.position)

          // 验证引用不同
          expect(cloned).not.toBe(state)
          expect(cloned.coords).not.toBe(state.coords)
          expect(cloned.position).not.toBe(state.position)
          expect(cloned.polygon).not.toBe(state.polygon)

          // 修改克隆不应影响原始
          cloned.originalText = '修改后'
          expect(state.originalText).not.toBe('修改后')

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 cloneBubbleStates 批量深拷贝逻辑
     */
    it('cloneBubbleStates 应创建独立的深拷贝数组', () => {
      fc.assert(
        fc.property(bubbleStatesArb, (states) => {
          const cloned = cloneBubbleStates(states)

          // 验证数量一致
          expect(cloned.length).toBe(states.length)

          // 验证每个元素都是独立的
          for (let i = 0; i < states.length; i++) {
            const original = states[i]
            const copy = cloned[i]
            if (original && copy) {
              expect(copy).not.toBe(original)
              expect(copy.coords).not.toBe(original.coords)
            }
          }

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 getBubbleCenter 中心点计算
     */
    it('getBubbleCenter 应正确计算中心点', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const state = createBubbleState({ coords })
          const center = getBubbleCenter(state)

          const [x1, y1, x2, y2] = coords
          expect(center.x).toBe((x1 + x2) / 2)
          expect(center.y).toBe((y1 + y2) / 2)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 getBubbleSize 尺寸计算
     */
    it('getBubbleSize 应正确计算宽高', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const state = createBubbleState({ coords })
          const size = getBubbleSize(state)

          const [x1, y1, x2, y2] = coords
          expect(size.width).toBe(Math.abs(x2 - x1))
          expect(size.height).toBe(Math.abs(y2 - y1))

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 isPointInBubble 点在矩形内判断
     */
    it('isPointInBubble 应正确判断点是否在矩形内', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const state = createBubbleState({ coords })
          const center = getBubbleCenter(state)

          // 中心点应该在矩形内
          expect(isPointInBubble(state, center.x, center.y)).toBe(true)

          // 远离矩形的点应该在矩形外
          const [x1, y1, x2, y2] = coords
          const farX = Math.max(x1, x2) + 1000
          const farY = Math.max(y1, y2) + 1000
          expect(isPointInBubble(state, farX, farY)).toBe(false)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 createBubbleStatesFromResponse 从响应创建状态
     */
    it('createBubbleStatesFromResponse 应正确从响应创建状态', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 5 }),
          fc.array(fc.string(), { minLength: 1, maxLength: 5 }),
          (coords, texts) => {
            const response = {
              bubble_coords: coords,
              original_texts: texts.slice(0, coords.length),
              bubble_texts: texts.slice(0, coords.length)
            }

            const states = createBubbleStatesFromResponse(response)

            // 验证数量与坐标数量一致
            expect(states.length).toBe(coords.length)

            // 验证每个状态的坐标正确
            for (let i = 0; i < coords.length; i++) {
              expect(states[i]?.coords).toEqual(coords[i])
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 bubbleStatesToApiRequest 转换为API请求格式
     */
    it('bubbleStatesToApiRequest 应正确转换为API请求格式', () => {
      fc.assert(
        fc.property(fc.array(bubbleStateArb, { minLength: 1, maxLength: 5 }), (states) => {
          const request = bubbleStatesToApiRequest(states)

          // 验证数量一致
          expect(request.bubble_coords.length).toBe(states.length)
          expect(request.bubble_states.length).toBe(states.length)
          expect(request.original_texts.length).toBe(states.length)
          expect(request.translated_texts.length).toBe(states.length)
          expect(request.textbox_texts.length).toBe(states.length)

          // 验证内容正确
          for (let i = 0; i < states.length; i++) {
            const state = states[i]
            if (state) {
              expect(request.bubble_coords[i]).toEqual(state.coords)
              expect(request.original_texts[i]).toBe(state.originalText)
              expect(request.translated_texts[i]).toBe(state.translatedText)
              expect(request.textbox_texts[i]).toBe(state.textboxText)
            }
          }

          return true
        }),
        { numRuns: 100 }
      )
    })
  })

  // ============================================================
  // Property 35: 多边形点击检测一致性
  // ============================================================

  describe('Property 35: 多边形点击检测一致性', () => {
    /**
     * 测试 isPointInPolygon 空多边形
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 空多边形应返回 false', () => {
      expect(isPointInPolygon([], 50, 50)).toBe(false)
      expect(isPointInPolygon([[0, 0]], 50, 50)).toBe(false)
      expect(isPointInPolygon([[0, 0], [100, 0]], 50, 50)).toBe(false)
    })

    /**
     * 测试 isPointInPolygon 简单矩形多边形
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 应正确判断点在矩形多边形内', () => {
      const rectangle = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ]

      // 中心点应该在内部
      expect(isPointInPolygon(rectangle, 50, 50)).toBe(true)

      // 外部点应该在外部
      expect(isPointInPolygon(rectangle, 150, 50)).toBe(false)
      expect(isPointInPolygon(rectangle, -50, 50)).toBe(false)
    })

    /**
     * 测试 isPointInPolygon 三角形多边形
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 应正确判断点在三角形内', () => {
      // 等腰三角形：顶点在 (50, 0)，底边在 (0, 100) 和 (100, 100)
      const triangle = [
        [50, 0],
        [100, 100],
        [0, 100]
      ]

      // 三角形重心应该在内部
      expect(isPointInPolygon(triangle, 50, 66)).toBe(true)

      // 三角形外部的点
      expect(isPointInPolygon(triangle, 10, 10)).toBe(false)
      expect(isPointInPolygon(triangle, 90, 10)).toBe(false)
    })

    /**
     * 测试 isPointInPolygon 凹多边形（L形）
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 应正确处理凹多边形', () => {
      // L形多边形
      const lShape = [
        [0, 0],
        [50, 0],
        [50, 50],
        [100, 50],
        [100, 100],
        [0, 100]
      ]

      // L形内部的点
      expect(isPointInPolygon(lShape, 25, 25)).toBe(true)
      expect(isPointInPolygon(lShape, 25, 75)).toBe(true)
      expect(isPointInPolygon(lShape, 75, 75)).toBe(true)

      // L形凹陷处（外部）
      expect(isPointInPolygon(lShape, 75, 25)).toBe(false)
    })

    /**
     * 测试 isPointInPolygon 属性：多边形内部点的随机测试
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 矩形多边形内部点应返回 true', () => {
      fc.assert(
        fc.property(
          // 生成矩形的左上角和尺寸
          fc.integer({ min: 0, max: 500 }),
          fc.integer({ min: 0, max: 500 }),
          fc.integer({ min: 10, max: 200 }),
          fc.integer({ min: 10, max: 200 }),
          // 生成内部点的相对位置（10%-90%之间，使用整数百分比）
          fc.integer({ min: 10, max: 90 }),
          fc.integer({ min: 10, max: 90 }),
          (x, y, width, height, relXPercent, relYPercent) => {
            // 构建矩形多边形
            const polygon = [
              [x, y],
              [x + width, y],
              [x + width, y + height],
              [x, y + height]
            ]

            // 计算内部点坐标（使用百分比）
            const pointX = x + width * (relXPercent / 100)
            const pointY = y + height * (relYPercent / 100)

            // 内部点应该返回 true
            expect(isPointInPolygon(polygon, pointX, pointY)).toBe(true)
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 isPointInPolygon 属性：多边形外部点的随机测试
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 矩形多边形外部点应返回 false', () => {
      fc.assert(
        fc.property(
          // 生成矩形的左上角和尺寸
          fc.integer({ min: 100, max: 500 }),
          fc.integer({ min: 100, max: 500 }),
          fc.integer({ min: 50, max: 200 }),
          fc.integer({ min: 50, max: 200 }),
          // 生成外部偏移量
          fc.integer({ min: 10, max: 100 }),
          (x, y, width, height, offset) => {
            // 构建矩形多边形
            const polygon = [
              [x, y],
              [x + width, y],
              [x + width, y + height],
              [x, y + height]
            ]

            // 测试四个方向的外部点
            // 左侧外部
            expect(isPointInPolygon(polygon, x - offset, y + height / 2)).toBe(false)
            // 右侧外部
            expect(isPointInPolygon(polygon, x + width + offset, y + height / 2)).toBe(false)
            // 上方外部
            expect(isPointInPolygon(polygon, x + width / 2, y - offset)).toBe(false)
            // 下方外部
            expect(isPointInPolygon(polygon, x + width / 2, y + height + offset)).toBe(false)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 isPointInBubbleArea 优先使用多边形
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInBubbleArea 有多边形时应优先使用多边形检测', () => {
      const state = createBubbleState({
        coords: [0, 0, 100, 100],
        polygon: [
          [25, 25],
          [75, 25],
          [75, 75],
          [25, 75]
        ]
      })

      // 在多边形内但在矩形内的点
      expect(isPointInBubbleArea(state, 50, 50)).toBe(true)

      // 在矩形内但在多边形外的点
      expect(isPointInBubbleArea(state, 10, 10)).toBe(false)
    })

    /**
     * 测试 isPointInBubbleArea 无多边形时使用矩形
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInBubbleArea 无多边形时应使用矩形检测', () => {
      const state = createBubbleState({
        coords: [0, 0, 100, 100],
        polygon: []
      })

      // 在矩形内的点
      expect(isPointInBubbleArea(state, 50, 50)).toBe(true)
      expect(isPointInBubbleArea(state, 10, 10)).toBe(true)

      // 在矩形外的点
      expect(isPointInBubbleArea(state, 150, 50)).toBe(false)
    })

    /**
     * 测试 isPointInBubbleArea 属性：多边形优先级
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInBubbleArea 多边形应优先于矩形坐标', () => {
      fc.assert(
        fc.property(
          // 生成矩形坐标
          bubbleCoordsArb,
          // 生成内部多边形的缩放比例（20%-40%，使用整数百分比）
          fc.integer({ min: 20, max: 40 }),
          (coords, scalePercent) => {
            const [x1, y1, x2, y2] = coords
            const width = x2 - x1
            const height = y2 - y1
            const centerX = (x1 + x2) / 2
            const centerY = (y1 + y2) / 2
            const scale = scalePercent / 100

            // 创建一个比矩形小的内部多边形
            const innerWidth = width * scale
            const innerHeight = height * scale
            const polygon = [
              [centerX - innerWidth / 2, centerY - innerHeight / 2],
              [centerX + innerWidth / 2, centerY - innerHeight / 2],
              [centerX + innerWidth / 2, centerY + innerHeight / 2],
              [centerX - innerWidth / 2, centerY + innerHeight / 2]
            ]

            const state = createBubbleState({ coords, polygon })

            // 中心点应该在多边形内
            expect(isPointInBubbleArea(state, centerX, centerY)).toBe(true)

            // 矩形角落（多边形外）应该返回 false
            // 使用一个在矩形内但在多边形外的点
            const cornerX = x1 + width * 0.1
            const cornerY = y1 + height * 0.1
            expect(isPointInBubbleArea(state, cornerX, cornerY)).toBe(false)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 isPointInPolygon 边界点处理
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 边界点处理应一致', () => {
      // 简单正方形
      const square = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ]

      // 测试边界附近的点（略微在内部）
      expect(isPointInPolygon(square, 1, 50)).toBe(true)
      expect(isPointInPolygon(square, 99, 50)).toBe(true)
      expect(isPointInPolygon(square, 50, 1)).toBe(true)
      expect(isPointInPolygon(square, 50, 99)).toBe(true)

      // 测试边界附近的点（略微在外部）
      expect(isPointInPolygon(square, -1, 50)).toBe(false)
      expect(isPointInPolygon(square, 101, 50)).toBe(false)
      expect(isPointInPolygon(square, 50, -1)).toBe(false)
      expect(isPointInPolygon(square, 50, 101)).toBe(false)
    })

    /**
     * 测试 isPointInPolygon 复杂凹多边形（星形）
     *
     * Feature: vue-frontend-migration, Property 35: 多边形点击检测一致性
     * Validates: Requirements 17.2
     */
    it('isPointInPolygon 应正确处理复杂凹多边形', () => {
      // 简化的星形（5个点）
      const star = [
        [50, 0],    // 顶点
        [60, 35],   // 右上内凹
        [100, 35],  // 右上外点
        [70, 60],   // 右下内凹
        [80, 100],  // 右下外点
        [50, 75],   // 底部内凹
        [20, 100],  // 左下外点
        [30, 60],   // 左下内凹
        [0, 35],    // 左上外点
        [40, 35]    // 左上内凹
      ]

      // 中心点应该在内部
      expect(isPointInPolygon(star, 50, 50)).toBe(true)

      // 星形凹陷处（外部）- 右上角凹陷
      expect(isPointInPolygon(star, 75, 20)).toBe(false)
    })
  })

  // ============================================================
  // Property 42: 气泡状态初始化一致性
  // ============================================================

  describe('Property 42: 气泡状态初始化一致性', () => {
    /**
     * 测试 getDefaultBubbleSettings 无参数时返回默认值
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1, 29.2
     */
    it('getDefaultBubbleSettings 无参数时应返回正确的默认值', () => {
      const defaults = getDefaultBubbleSettings()

      // 验证默认值存在
      expect(defaults.fontSize).toBe(24)
      expect(defaults.fontFamily).toBe('')
      expect(defaults.textDirection).toBe('auto')
      expect(defaults.textColor).toBe('#000000')
      expect(defaults.fillColor).toBe(DEFAULT_FILL_COLOR)
      expect(defaults.strokeEnabled).toBe(DEFAULT_STROKE_ENABLED)
      expect(defaults.strokeColor).toBe(DEFAULT_STROKE_COLOR)
      expect(defaults.strokeWidth).toBe(DEFAULT_STROKE_WIDTH)
      expect(defaults.inpaintMethod).toBe('solid')
    })

    /**
     * 测试 getDefaultBubbleSettings 应正确应用全局设置
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1
     */
    it('getDefaultBubbleSettings 应正确应用全局设置', () => {
      fc.assert(
        fc.property(
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            fontFamily: fc.constantFrom('fonts/STSONG.TTF', 'fonts/msyh.ttc', 'Arial'),
            layoutDirection: fc.constantFrom('auto', 'vertical', 'horizontal') as fc.Arbitrary<
              'auto' | 'vertical' | 'horizontal'
            >,
            textColor: colorArb,
            fillColor: colorArb,
            strokeEnabled: fc.boolean(),
            strokeColor: colorArb,
            strokeWidth: fc.integer({ min: 1, max: 10 }),
            inpaintMethod: inpaintMethodArb
          }),
          (globalSettings) => {
            const defaults = getDefaultBubbleSettings(globalSettings)

            // 验证全局设置被正确应用
            expect(defaults.fontSize).toBe(globalSettings.fontSize)
            expect(defaults.fontFamily).toBe(globalSettings.fontFamily)
            expect(defaults.textDirection).toBe(globalSettings.layoutDirection)
            expect(defaults.textColor).toBe(globalSettings.textColor)
            expect(defaults.fillColor).toBe(globalSettings.fillColor)
            expect(defaults.strokeEnabled).toBe(globalSettings.strokeEnabled)
            expect(defaults.strokeColor).toBe(globalSettings.strokeColor)
            expect(defaults.strokeWidth).toBe(globalSettings.strokeWidth)
            expect(defaults.inpaintMethod).toBe(globalSettings.inpaintMethod)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 initBubbleStates 有保存状态时直接使用
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1
     */
    it('initBubbleStates 有保存状态时应直接使用', () => {
      fc.assert(
        fc.property(bubbleStatesArb.filter((s) => s.length > 0), (savedStates) => {
          const result = initBubbleStates(savedStates, undefined, undefined)

          // 验证数量一致
          expect(result.length).toBe(savedStates.length)

          // 验证内容一致（深拷贝）
          for (let i = 0; i < savedStates.length; i++) {
            const saved = savedStates[i]
            const initialized = result[i]
            if (saved && initialized) {
              expect(initialized.originalText).toBe(saved.originalText)
              expect(initialized.coords).toEqual(saved.coords)
              // 验证是独立的副本
              expect(initialized).not.toBe(saved)
              expect(initialized.coords).not.toBe(saved.coords)
            }
          }

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 initBubbleStates 无保存状态时根据坐标创建
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1
     */
    it('initBubbleStates 无保存状态时应根据坐标创建', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 10 }),
          (coords) => {
            const result = initBubbleStates(undefined, coords, undefined)

            // 验证数量与坐标数量一致
            expect(result.length).toBe(coords.length)

            // 验证每个状态的坐标正确
            for (let i = 0; i < coords.length; i++) {
              expect(result[i]?.coords).toEqual(coords[i])
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 initBubbleStates 无坐标时返回空数组
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1
     */
    it('initBubbleStates 无坐标时应返回空数组', () => {
      const result1 = initBubbleStates(undefined, undefined, undefined)
      expect(result1).toEqual([])

      const result2 = initBubbleStates(undefined, [], undefined)
      expect(result2).toEqual([])
    })

    /**
     * 测试 initBubbleStates 自动排版方向检测
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 29.2
     */
    it('initBubbleStates 应正确设置自动排版方向', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 5 }),
          (coords) => {
            const result = initBubbleStates(undefined, coords, undefined)

            // 验证每个气泡的自动排版方向与 detectTextDirection 一致
            for (let i = 0; i < coords.length; i++) {
              const coord = coords[i]
              const state = result[i]
              if (coord && state) {
                const expectedDirection = detectTextDirection(coord)
                expect(state.autoTextDirection).toBe(expectedDirection)
              }
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试 initBubbleStates 应用全局默认设置
     *
     * Feature: vue-frontend-migration, Property 42: 气泡状态初始化一致性
     * Validates: Requirements 30.1
     */
    it('initBubbleStates 应正确应用全局默认设置', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 5 }),
          fc.record({
            fontSize: fc.integer({ min: 8, max: 72 }),
            textColor: colorArb
          }),
          (coords, globalDefaults) => {
            const result = initBubbleStates(undefined, coords, globalDefaults)

            // 验证全局默认设置被应用
            for (const state of result) {
              expect(state.fontSize).toBe(globalDefaults.fontSize)
              expect(state.textColor).toBe(globalDefaults.textColor)
            }

            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})
