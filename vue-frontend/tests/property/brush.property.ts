/**
 * 笔刷工具属性测试
 * 
 * Feature: vue-frontend-migration
 * Property 22: 笔刷大小边界一致性
 * Property 23: 气泡旋转角度计算一致性
 * 
 * Validates: Requirements 17.3, 33.1, 33.2
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { BRUSH_MIN_SIZE, BRUSH_MAX_SIZE, BRUSH_DEFAULT_SIZE } from '@/constants'

// ============================================================
// 辅助函数 - 笔刷大小控制
// ============================================================

/**
 * 设置笔刷大小（带边界约束）
 * @param size 目标大小
 * @returns 约束后的大小
 */
function setBrushSize(size: number): number {
  return Math.max(BRUSH_MIN_SIZE, Math.min(BRUSH_MAX_SIZE, size))
}

/**
 * 调整笔刷大小
 * @param currentSize 当前大小
 * @param delta 调整量（正数增大，负数减小）
 * @returns 调整后的大小
 */
function adjustBrushSize(currentSize: number, delta: number): number {
  return setBrushSize(currentSize + delta)
}

/**
 * 处理滚轮调整笔刷大小
 * @param currentSize 当前大小
 * @param deltaY 滚轮方向（正数向下，负数向上）
 * @returns 调整后的大小
 */
function handleBrushWheel(currentSize: number, deltaY: number): number {
  const delta = deltaY > 0 ? -5 : 5
  return adjustBrushSize(currentSize, delta)
}

// ============================================================
// 辅助函数 - 气泡旋转计算
// ============================================================

/**
 * 计算旋转中心
 * @param x1 左上角X
 * @param y1 左上角Y
 * @param x2 右下角X
 * @param y2 右下角Y
 * @returns 旋转中心坐标
 */
function calculateRotationCenter(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { centerX: number; centerY: number } {
  return {
    centerX: (x1 + x2) / 2,
    centerY: (y1 + y2) / 2
  }
}

/**
 * 归一化角度到 0-360 度范围（不包含360）
 * @param angle 原始角度
 * @returns 归一化后的角度 [0, 360)，如果输入为 NaN 或 Infinity 则返回 0
 */
function normalizeAngle(angle: number): number {
  // 处理无效输入（NaN、Infinity）
  if (!Number.isFinite(angle)) {
    return 0
  }
  // 处理负角度和超过360度的角度
  let normalized = angle % 360
  if (normalized < 0) {
    normalized += 360
  }
  // 处理浮点数精度问题，确保结果在 [0, 360) 范围内
  if (normalized >= 360) {
    normalized = 0
  }
  return normalized
}

/**
 * 计算从中心点到鼠标位置的角度
 * @param centerX 中心点X
 * @param centerY 中心点Y
 * @param mouseX 鼠标X
 * @param mouseY 鼠标Y
 * @returns 角度（0-360度）
 */
function calculateAngleFromCenter(
  centerX: number,
  centerY: number,
  mouseX: number,
  mouseY: number
): number {
  const dx = mouseX - centerX
  const dy = mouseY - centerY
  // atan2 返回 -PI 到 PI 的弧度
  const radians = Math.atan2(dy, dx)
  // 转换为角度
  const degrees = radians * (180 / Math.PI)
  // 归一化到 0-360
  return normalizeAngle(degrees)
}

/**
 * 计算旋转后的角度
 * @param startAngle 起始角度
 * @param currentAngle 当前角度
 * @param initialRotation 初始旋转角度
 * @returns 新的旋转角度
 */
function calculateRotatedAngle(
  startAngle: number,
  currentAngle: number,
  initialRotation: number
): number {
  const deltaAngle = currentAngle - startAngle
  return normalizeAngle(initialRotation + deltaAngle)
}

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的笔刷大小
 */
const validBrushSizeArb = fc.integer({ min: BRUSH_MIN_SIZE, max: BRUSH_MAX_SIZE })

/**
 * 生成任意笔刷大小（可能超出边界）
 */
const anyBrushSizeArb = fc.integer({ min: -100, max: 500 })

/**
 * 生成滚轮调整量
 */
const wheelDeltaArb = fc.integer({ min: -1000, max: 1000 })

/**
 * 生成调整步长
 */
const adjustDeltaArb = fc.integer({ min: -300, max: 300 })

/**
 * 生成有效的气泡坐标
 */
const bubbleCoordsArb = fc.tuple(
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 50, max: 1000 }),
  fc.integer({ min: 50, max: 1000 })
).map(([x1, y1, w, h]) => ({
  x1,
  y1,
  x2: x1 + w,
  y2: y1 + h
}))

/**
 * 生成任意角度（排除 NaN 和 Infinity）
 */
const anyAngleArb = fc.double({ min: -720, max: 720, noNaN: true })

/**
 * 生成鼠标位置（排除 NaN 和 Infinity）
 */
const mousePositionArb = fc.record({
  x: fc.double({ min: -1000, max: 2000, noNaN: true }),
  y: fc.double({ min: -1000, max: 2000, noNaN: true })
})

// ============================================================
// 属性测试 - 笔刷大小控制
// ============================================================

describe('笔刷大小控制属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 22: 笔刷大小边界一致性
   * 
   * 对于任意笔刷大小设置，结果应当始终在 MIN-MAX 范围内。
   * 
   * Feature: vue-frontend-migration, Property 22: 笔刷大小边界一致性
   * Validates: Requirements 17.3
   */
  describe('Property 22: 笔刷大小边界一致性', () => {
    it('设置任意大小后应当在 MIN-MAX 范围内', () => {
      fc.assert(
        fc.property(anyBrushSizeArb, (size) => {
          const result = setBrushSize(size)
          
          // 验证结果在有效范围内
          expect(result).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
          expect(result).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('从有效大小调整后应当仍在 MIN-MAX 范围内', () => {
      fc.assert(
        fc.property(validBrushSizeArb, adjustDeltaArb, (currentSize, delta) => {
          const result = adjustBrushSize(currentSize, delta)
          
          // 验证结果在有效范围内
          expect(result).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
          expect(result).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('滚轮调整后应当在 MIN-MAX 范围内', () => {
      fc.assert(
        fc.property(validBrushSizeArb, wheelDeltaArb, (currentSize, deltaY) => {
          const result = handleBrushWheel(currentSize, deltaY)
          
          // 验证结果在有效范围内
          expect(result).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
          expect(result).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('滚轮向上应当增大笔刷，向下应当减小笔刷', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: BRUSH_MIN_SIZE + 10, max: BRUSH_MAX_SIZE - 10 }),
          (currentSize) => {
            // 滚轮向上（deltaY < 0）应当增大
            const resultUp = handleBrushWheel(currentSize, -100)
            expect(resultUp).toBeGreaterThan(currentSize)
            
            // 滚轮向下（deltaY > 0）应当减小
            const resultDown = handleBrushWheel(currentSize, 100)
            expect(resultDown).toBeLessThan(currentSize)
            
            return true
          }
        ),
        { numRuns: 50 }
      )
    })

    it('连续调整应当正确累积', () => {
      fc.assert(
        fc.property(
          validBrushSizeArb,
          fc.array(adjustDeltaArb, { minLength: 1, maxLength: 10 }),
          (initialSize, deltas) => {
            let currentSize = initialSize
            
            for (const delta of deltas) {
              currentSize = adjustBrushSize(currentSize, delta)
              
              // 每次调整后都应在有效范围内
              expect(currentSize).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
              expect(currentSize).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
            }
            
            return true
          }
        ),
        { numRuns: 50 }
      )
    })

    it('边界值测试 - 最小值', () => {
      // 设置为最小值
      const result1 = setBrushSize(BRUSH_MIN_SIZE)
      expect(result1).toBe(BRUSH_MIN_SIZE)
      
      // 设置为小于最小值
      const result2 = setBrushSize(BRUSH_MIN_SIZE - 10)
      expect(result2).toBe(BRUSH_MIN_SIZE)
      
      // 从最小值减小
      const result3 = adjustBrushSize(BRUSH_MIN_SIZE, -10)
      expect(result3).toBe(BRUSH_MIN_SIZE)
    })

    it('边界值测试 - 最大值', () => {
      // 设置为最大值
      const result1 = setBrushSize(BRUSH_MAX_SIZE)
      expect(result1).toBe(BRUSH_MAX_SIZE)
      
      // 设置为大于最大值
      const result2 = setBrushSize(BRUSH_MAX_SIZE + 100)
      expect(result2).toBe(BRUSH_MAX_SIZE)
      
      // 从最大值增大
      const result3 = adjustBrushSize(BRUSH_MAX_SIZE, 50)
      expect(result3).toBe(BRUSH_MAX_SIZE)
    })

    it('默认值应当在有效范围内', () => {
      expect(BRUSH_DEFAULT_SIZE).toBeGreaterThanOrEqual(BRUSH_MIN_SIZE)
      expect(BRUSH_DEFAULT_SIZE).toBeLessThanOrEqual(BRUSH_MAX_SIZE)
    })
  })
})

// ============================================================
// 属性测试 - 气泡旋转计算
// ============================================================

describe('气泡旋转计算属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 23: 气泡旋转角度计算一致性
   * 
   * 对于任意气泡坐标，旋转中心应当正确计算；
   * 对于任意角度，归一化后应当在 0-360 度范围内。
   * 
   * Feature: vue-frontend-migration, Property 23: 气泡旋转角度计算一致性
   * Validates: Requirements 33.1, 33.2
   */
  describe('Property 23: 气泡旋转角度计算一致性', () => {
    it('旋转中心应当是气泡的几何中心', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const { centerX, centerY } = calculateRotationCenter(
            coords.x1,
            coords.y1,
            coords.x2,
            coords.y2
          )
          
          // 验证中心点计算正确
          const expectedCenterX = (coords.x1 + coords.x2) / 2
          const expectedCenterY = (coords.y1 + coords.y2) / 2
          
          expect(centerX).toBeCloseTo(expectedCenterX, 10)
          expect(centerY).toBeCloseTo(expectedCenterY, 10)
          
          // 验证中心点在气泡范围内
          expect(centerX).toBeGreaterThanOrEqual(coords.x1)
          expect(centerX).toBeLessThanOrEqual(coords.x2)
          expect(centerY).toBeGreaterThanOrEqual(coords.y1)
          expect(centerY).toBeLessThanOrEqual(coords.y2)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('角度归一化后应当在 0-360 度范围内', () => {
      fc.assert(
        fc.property(anyAngleArb, (angle) => {
          const normalized = normalizeAngle(angle)
          
          // 验证归一化后在有效范围内
          expect(normalized).toBeGreaterThanOrEqual(0)
          expect(normalized).toBeLessThan(360)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('归一化应当保持角度的等价性', () => {
      fc.assert(
        fc.property(anyAngleArb, (angle) => {
          const normalized = normalizeAngle(angle)
          
          // 归一化后的角度与原角度应当表示相同的方向
          // 通过比较正弦和余弦值来验证
          const originalRad = angle * (Math.PI / 180)
          const normalizedRad = normalized * (Math.PI / 180)
          
          expect(Math.sin(normalizedRad)).toBeCloseTo(Math.sin(originalRad), 10)
          expect(Math.cos(normalizedRad)).toBeCloseTo(Math.cos(originalRad), 10)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('从中心点计算的角度应当在 0-360 度范围内', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, mousePositionArb, (coords, mouse) => {
          const { centerX, centerY } = calculateRotationCenter(
            coords.x1,
            coords.y1,
            coords.x2,
            coords.y2
          )
          
          const angle = calculateAngleFromCenter(centerX, centerY, mouse.x, mouse.y)
          
          // 验证角度在有效范围内
          expect(angle).toBeGreaterThanOrEqual(0)
          expect(angle).toBeLessThan(360)
          
          return true
        }),
        { numRuns: 100 }
      )
    })

    it('旋转计算应当正确累积角度变化', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 360, noNaN: true }),
          fc.double({ min: 0, max: 360, noNaN: true }),
          fc.double({ min: 0, max: 360, noNaN: true }),
          (startAngle, currentAngle, initialRotation) => {
            const result = calculateRotatedAngle(startAngle, currentAngle, initialRotation)
            
            // 验证结果在有效范围内
            expect(result).toBeGreaterThanOrEqual(0)
            expect(result).toBeLessThan(360)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('特殊角度值测试', () => {
      // 0度
      expect(normalizeAngle(0)).toBe(0)
      
      // 90度
      expect(normalizeAngle(90)).toBe(90)
      
      // 180度
      expect(normalizeAngle(180)).toBe(180)
      
      // 270度
      expect(normalizeAngle(270)).toBe(270)
      
      // 360度应当归一化为0
      expect(normalizeAngle(360)).toBe(0)
      
      // 负角度
      expect(normalizeAngle(-90)).toBe(270)
      expect(normalizeAngle(-180)).toBe(180)
      expect(normalizeAngle(-270)).toBe(90)
      
      // 超过360度
      expect(normalizeAngle(450)).toBe(90)
      expect(normalizeAngle(720)).toBe(0)
    })

    it('旋转中心对称性测试', () => {
      fc.assert(
        fc.property(bubbleCoordsArb, (coords) => {
          const { centerX, centerY } = calculateRotationCenter(
            coords.x1,
            coords.y1,
            coords.x2,
            coords.y2
          )
          
          // 中心点到四个角的距离应当满足对称性
          const distToTopLeft = Math.sqrt(
            Math.pow(centerX - coords.x1, 2) + Math.pow(centerY - coords.y1, 2)
          )
          const distToBottomRight = Math.sqrt(
            Math.pow(centerX - coords.x2, 2) + Math.pow(centerY - coords.y2, 2)
          )
          const distToTopRight = Math.sqrt(
            Math.pow(centerX - coords.x2, 2) + Math.pow(centerY - coords.y1, 2)
          )
          const distToBottomLeft = Math.sqrt(
            Math.pow(centerX - coords.x1, 2) + Math.pow(centerY - coords.y2, 2)
          )
          
          // 对角线上的点到中心距离应当相等
          expect(distToTopLeft).toBeCloseTo(distToBottomRight, 10)
          expect(distToTopRight).toBeCloseTo(distToBottomLeft, 10)
          
          return true
        }),
        { numRuns: 50 }
      )
    })

    it('角度计算方向正确性测试', () => {
      const centerX = 100
      const centerY = 100
      
      // 正右方应当是0度
      const angleRight = calculateAngleFromCenter(centerX, centerY, 200, 100)
      expect(angleRight).toBeCloseTo(0, 1)
      
      // 正下方应当是90度
      const angleDown = calculateAngleFromCenter(centerX, centerY, 100, 200)
      expect(angleDown).toBeCloseTo(90, 1)
      
      // 正左方应当是180度
      const angleLeft = calculateAngleFromCenter(centerX, centerY, 0, 100)
      expect(angleLeft).toBeCloseTo(180, 1)
      
      // 正上方应当是270度
      const angleUp = calculateAngleFromCenter(centerX, centerY, 100, 0)
      expect(angleUp).toBeCloseTo(270, 1)
    })
  })
})
