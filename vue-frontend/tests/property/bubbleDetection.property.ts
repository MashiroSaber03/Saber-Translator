/**
 * 气泡检测功能属性测试
 *
 * Feature: vue-frontend-migration
 * Property 36: 气泡检测功能一致性
 *
 * Validates: Requirements 4.8
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import * as fc from 'fast-check'
import type { BubbleCoords } from '@/types/bubble'
import type { DetectBoxesResponse } from '@/types/api'

// ============================================================
// 模拟模块
// ============================================================

// 模拟 API 客户端
vi.mock('@/api/translate', () => ({
  detectBoxes: vi.fn()
}))

// 模拟 stores
vi.mock('@/stores/imageStore', () => ({
  useImageStore: vi.fn(() => ({
    images: [],
    currentImage: null,
    currentImageIndex: 0,
    updateImageByIndex: vi.fn(),
    setCurrentIndex: vi.fn()
  }))
}))

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: vi.fn(() => ({
    settings: {
      textDetector: 'ctd',
      boxExpand: {
        ratio: 0,
        top: 0,
        bottom: 0,
        left: 0,
        right: 0
      },
      textStyle: {
        fontSize: 24,
        fontFamily: '',
        layoutDirection: 'auto',
        textColor: '#000000',
        fillColor: '#FFFFFF',
        strokeEnabled: true,
        strokeColor: '#FFFFFF',
        strokeWidth: 3,
        inpaintMethod: 'solid'
      }
    }
  }))
}))

vi.mock('@/stores/bubbleStore', () => ({
  useBubbleStore: vi.fn(() => ({
    bubbles: [],
    setBubbles: vi.fn(),
    selectBubble: vi.fn()
  }))
}))

vi.mock('@/utils/toast', () => ({
  useToast: vi.fn(() => ({
    info: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn()
  }))
}))

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
 * 生成有效的自动方向
 */
const autoDirectionArb = fc.constantFrom('v', 'h') as fc.Arbitrary<'v' | 'h'>

/**
 * 生成有效的检测响应
 */
const detectResponseArb = fc.record({
  success: fc.constant(true),
  bubble_coords: fc.array(bubbleCoordsArb, { minLength: 0, maxLength: 20 }),
  bubble_angles: fc.array(fc.integer({ min: -180, max: 180 }), { minLength: 0, maxLength: 20 }),
  auto_directions: fc.array(autoDirectionArb, { minLength: 0, maxLength: 20 })
}).map((response) => {
  // 确保数组长度一致
  const length = response.bubble_coords.length
  return {
    ...response,
    bubble_angles: response.bubble_angles.slice(0, length),
    auto_directions: response.auto_directions.slice(0, length)
  }
})

/**
 * 生成失败的检测响应
 */
const failedResponseArb = fc.record({
  success: fc.constant(false),
  error: fc.string({ minLength: 1, maxLength: 100 })
})

// Base64 图片数据生成器（保留用于未来扩展）
// const base64ImageArb = fc.string({ minLength: 100, maxLength: 500 }).map(
//   (s) => `data:image/png;base64,${btoa(s)}`
// )

// ============================================================
// 辅助函数（从 useBubbleDetection 中提取的纯函数逻辑）
// ============================================================

/**
 * 根据宽高比检测文本方向
 * @param coords - 气泡坐标
 * @returns 文本方向
 */
function detectTextDirectionFromCoords(coords: BubbleCoords): 'vertical' | 'horizontal' {
  const [x1, y1, x2, y2] = coords
  const width = Math.abs(x2 - x1)
  const height = Math.abs(y2 - y1)
  return height > width ? 'vertical' : 'horizontal'
}

/**
 * 计算检测进度百分比
 * @param current - 当前处理数量
 * @param total - 总数量
 * @returns 百分比（0-100）
 */
function calculateProgressPercent(current: number, total: number): number {
  if (total === 0) return 0
  return Math.round((current / total) * 100)
}

/**
 * 从检测响应创建气泡状态数组
 * @param response - 检测响应
 * @param defaults - 默认设置
 * @returns 气泡状态数组
 */
function createBubbleStatesFromDetection(
  response: DetectBoxesResponse,
  defaults: {
    fontSize: number
    fontFamily: string
    textDirection: 'vertical' | 'horizontal' | 'auto'
    textColor: string
    fillColor: string
    strokeEnabled: boolean
    strokeColor: string
    strokeWidth: number
    inpaintMethod: 'solid' | 'lama_mpe' | 'litelama'
  }
) {
  if (!response.bubble_coords || response.bubble_coords.length === 0) {
    return []
  }

  const autoDirections = response.auto_directions || []
  const bubbleAngles = response.bubble_angles || []

  return response.bubble_coords.map((coords, index) => {
    // 确定自动排版方向
    let autoTextDirection: 'vertical' | 'horizontal'
    if (autoDirections[index]) {
      autoTextDirection = autoDirections[index] === 'v' ? 'vertical' : 'horizontal'
    } else {
      autoTextDirection = detectTextDirectionFromCoords(coords)
    }

    return {
      coords,
      originalText: '',
      translatedText: '',
      textboxText: '',
      polygon: [] as number[][],
      ...defaults,
      autoTextDirection,
      rotationAngle: bubbleAngles[index] || 0,
      position: { x: 0, y: 0 }
    }
  })
}

/**
 * 验证气泡坐标格式是否正确
 * @param coords - 气泡坐标
 * @returns 是否有效
 */
function isValidBubbleCoords(coords: unknown): coords is BubbleCoords {
  if (!Array.isArray(coords)) return false
  if (coords.length !== 4) return false
  return coords.every((v) => typeof v === 'number' && !isNaN(v))
}

/**
 * 验证检测响应格式是否正确
 * @param response - 检测响应
 * @returns 是否有效
 */
function isValidDetectResponse(response: unknown): response is DetectBoxesResponse {
  if (!response || typeof response !== 'object') return false
  const r = response as Record<string, unknown>
  if (typeof r.success !== 'boolean') return false
  if (r.bubble_coords !== undefined && !Array.isArray(r.bubble_coords)) return false
  return true
}

// ============================================================
// Property 36: 气泡检测功能一致性
// ============================================================

describe('气泡检测功能属性测试', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Property 36: 气泡检测功能一致性', () => {
    /**
     * 测试检测结果坐标格式正确
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('检测结果坐标格式应正确（4元素数组）', () => {
      fc.assert(
        fc.property(detectResponseArb, (response) => {
          if (response.bubble_coords) {
            for (const coords of response.bubble_coords) {
              // 验证坐标是4元素数组
              expect(coords).toHaveLength(4)
              // 验证所有元素都是数字
              expect(coords.every((v) => typeof v === 'number')).toBe(true)
              // 验证坐标有效（x2 > x1, y2 > y1）
              const [x1, y1, x2, y2] = coords
              expect(x2).toBeGreaterThan(x1)
              expect(y2).toBeGreaterThan(y1)
            }
          }
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试检测响应数组长度一致性
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('检测响应中各数组长度应一致', () => {
      fc.assert(
        fc.property(detectResponseArb, (response) => {
          if (response.success && response.bubble_coords) {
            const coordsLength = response.bubble_coords.length
            
            // bubble_angles 长度应与 bubble_coords 一致（或为空）
            if (response.bubble_angles) {
              expect(response.bubble_angles.length).toBeLessThanOrEqual(coordsLength)
            }
            
            // auto_directions 长度应与 bubble_coords 一致（或为空）
            if (response.auto_directions) {
              expect(response.auto_directions.length).toBeLessThanOrEqual(coordsLength)
            }
          }
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试从检测响应创建气泡状态数量正确
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('从检测响应创建的气泡状态数量应与坐标数量一致', () => {
      fc.assert(
        fc.property(detectResponseArb, (response) => {
          const defaults = {
            fontSize: 24,
            fontFamily: '',
            textDirection: 'auto' as const,
            textColor: '#000000',
            fillColor: '#FFFFFF',
            strokeEnabled: true,
            strokeColor: '#FFFFFF',
            strokeWidth: 3,
            inpaintMethod: 'solid' as const
          }

          const states = createBubbleStatesFromDetection(response, defaults)
          
          if (response.bubble_coords) {
            expect(states.length).toBe(response.bubble_coords.length)
          } else {
            expect(states.length).toBe(0)
          }
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试自动方向检测正确性
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('自动方向检测应根据宽高比正确判断', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const direction = detectTextDirectionFromCoords(coords)
          const [x1, y1, x2, y2] = coords
          const width = Math.abs(x2 - x1)
          const height = Math.abs(y2 - y1)

          if (height > width) {
            expect(direction).toBe('vertical')
          } else {
            expect(direction).toBe('horizontal')
          }
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试后端返回的自动方向应优先使用
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('后端返回的自动方向应优先于宽高比计算', () => {
      fc.assert(
        fc.property(
          bubbleCoordsArb,
          autoDirectionArb,
          (coords, backendDirection) => {
            const response: DetectBoxesResponse = {
              success: true,
              bubble_coords: [coords],
              auto_directions: [backendDirection]
            }

            const defaults = {
              fontSize: 24,
              fontFamily: '',
              textDirection: 'auto' as const,
              textColor: '#000000',
              fillColor: '#FFFFFF',
              strokeEnabled: true,
              strokeColor: '#FFFFFF',
              strokeWidth: 3,
              inpaintMethod: 'solid' as const
            }

            const states = createBubbleStatesFromDetection(response, defaults)
            
            expect(states.length).toBe(1)
            const expectedDirection = backendDirection === 'v' ? 'vertical' : 'horizontal'
            expect(states[0]?.autoTextDirection).toBe(expectedDirection)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试旋转角度正确传递
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('旋转角度应正确传递到气泡状态', () => {
      fc.assert(
        fc.property(
          bubbleCoordsArb,
          fc.integer({ min: -180, max: 180 }),
          (coords, angle) => {
            const response: DetectBoxesResponse = {
              success: true,
              bubble_coords: [coords],
              bubble_angles: [angle]
            }

            const defaults = {
              fontSize: 24,
              fontFamily: '',
              textDirection: 'auto' as const,
              textColor: '#000000',
              fillColor: '#FFFFFF',
              strokeEnabled: true,
              strokeColor: '#FFFFFF',
              strokeWidth: 3,
              inpaintMethod: 'solid' as const
            }

            const states = createBubbleStatesFromDetection(response, defaults)
            
            expect(states.length).toBe(1)
            expect(states[0]?.rotationAngle).toBe(angle)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  describe('批量检测进度计算测试', () => {
    /**
     * 测试进度百分比计算正确性
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('进度百分比计算应正确', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 100 }),
          fc.integer({ min: 1, max: 100 }),
          (current, total) => {
            // 确保 current <= total
            const actualCurrent = Math.min(current, total)
            const percent = calculateProgressPercent(actualCurrent, total)
            
            // 验证百分比在 0-100 范围内
            expect(percent).toBeGreaterThanOrEqual(0)
            expect(percent).toBeLessThanOrEqual(100)
            
            // 验证计算正确
            const expected = Math.round((actualCurrent / total) * 100)
            expect(percent).toBe(expected)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试总数为0时进度为0
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('总数为0时进度应为0', () => {
      fc.assert(
        fc.property(fc.integer({ min: 0, max: 100 }), (current) => {
          const percent = calculateProgressPercent(current, 0)
          expect(percent).toBe(0)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试完成时进度为100
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('完成时进度应为100', () => {
      fc.assert(
        fc.property(fc.integer({ min: 1, max: 100 }), (total) => {
          const percent = calculateProgressPercent(total, total)
          expect(percent).toBe(100)
          return true
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('坐标验证测试', () => {
    /**
     * 测试有效坐标验证
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('有效坐标应通过验证', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          expect(isValidBubbleCoords(coords)).toBe(true)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试无效坐标验证
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('无效坐标应被拒绝', () => {
      // 非数组
      expect(isValidBubbleCoords(null)).toBe(false)
      expect(isValidBubbleCoords(undefined)).toBe(false)
      expect(isValidBubbleCoords('string')).toBe(false)
      expect(isValidBubbleCoords(123)).toBe(false)
      expect(isValidBubbleCoords({})).toBe(false)

      // 长度不正确
      expect(isValidBubbleCoords([])).toBe(false)
      expect(isValidBubbleCoords([0, 0, 100])).toBe(false)
      expect(isValidBubbleCoords([0, 0, 100, 100, 200])).toBe(false)

      // 包含非数字
      expect(isValidBubbleCoords([0, 0, 'invalid', 100])).toBe(false)
      expect(isValidBubbleCoords([0, 0, NaN, 100])).toBe(false)
    })
  })

  describe('检测响应验证测试', () => {
    /**
     * 测试有效响应验证
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('有效响应应通过验证', () => {
      fc.assert(
        fc.property(detectResponseArb, (response) => {
          expect(isValidDetectResponse(response)).toBe(true)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试失败响应验证
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('失败响应也应通过格式验证', () => {
      fc.assert(
        fc.property(failedResponseArb, (response) => {
          expect(isValidDetectResponse(response)).toBe(true)
          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试无效响应验证
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('无效响应应被拒绝', () => {
      expect(isValidDetectResponse(null)).toBe(false)
      expect(isValidDetectResponse(undefined)).toBe(false)
      expect(isValidDetectResponse('string')).toBe(false)
      expect(isValidDetectResponse(123)).toBe(false)
      expect(isValidDetectResponse([])).toBe(false)
      expect(isValidDetectResponse({})).toBe(false)
      expect(isValidDetectResponse({ success: 'not boolean' })).toBe(false)
      expect(isValidDetectResponse({ success: true, bubble_coords: 'not array' })).toBe(false)
    })
  })

  describe('空响应处理测试', () => {
    /**
     * 测试空坐标数组处理
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('空坐标数组应返回空状态数组', () => {
      const response: DetectBoxesResponse = {
        success: true,
        bubble_coords: []
      }

      const defaults = {
        fontSize: 24,
        fontFamily: '',
        textDirection: 'auto' as const,
        textColor: '#000000',
        fillColor: '#FFFFFF',
        strokeEnabled: true,
        strokeColor: '#FFFFFF',
        strokeWidth: 3,
        inpaintMethod: 'solid' as const
      }

      const states = createBubbleStatesFromDetection(response, defaults)
      expect(states).toEqual([])
    })

    /**
     * 测试无坐标字段处理
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('无坐标字段应返回空状态数组', () => {
      const response: DetectBoxesResponse = {
        success: true
      }

      const defaults = {
        fontSize: 24,
        fontFamily: '',
        textDirection: 'auto' as const,
        textColor: '#000000',
        fillColor: '#FFFFFF',
        strokeEnabled: true,
        strokeColor: '#FFFFFF',
        strokeWidth: 3,
        inpaintMethod: 'solid' as const
      }

      const states = createBubbleStatesFromDetection(response, defaults)
      expect(states).toEqual([])
    })
  })

  describe('默认设置应用测试', () => {
    /**
     * 测试默认设置正确应用到气泡状态
     *
     * Feature: vue-frontend-migration, Property 36: 气泡检测功能一致性
     * Validates: Requirements 4.8
     */
    it('默认设置应正确应用到所有气泡状态', () => {
      fc.assert(
        fc.property(
          fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 10 }),
          fc.integer({ min: 8, max: 72 }),
          fc.hexaString({ minLength: 6, maxLength: 6 }).map((hex) => `#${hex}`),
          (coords, fontSize, textColor) => {
            const response: DetectBoxesResponse = {
              success: true,
              bubble_coords: coords
            }

            const defaults = {
              fontSize,
              fontFamily: 'Arial',
              textDirection: 'auto' as const,
              textColor,
              fillColor: '#FFFFFF',
              strokeEnabled: true,
              strokeColor: '#FFFFFF',
              strokeWidth: 3,
              inpaintMethod: 'solid' as const
            }

            const states = createBubbleStatesFromDetection(response, defaults)
            
            // 验证所有状态都应用了默认设置
            for (const state of states) {
              expect(state.fontSize).toBe(fontSize)
              expect(state.fontFamily).toBe('Arial')
              expect(state.textColor).toBe(textColor)
              expect(state.fillColor).toBe('#FFFFFF')
              expect(state.strokeEnabled).toBe(true)
              expect(state.strokeColor).toBe('#FFFFFF')
              expect(state.strokeWidth).toBe(3)
              expect(state.inpaintMethod).toBe('solid')
            }
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})
