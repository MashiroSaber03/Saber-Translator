/**
 * 触摸手势处理属性测试
 * 
 * **Feature: vue-frontend-migration, Property 44: 触摸手势处理一致性**
 * **Validates: Requirements 25.3**
 * 
 * 测试内容：
 * - 滑动方向检测正确（左滑/右滑/上滑/下滑）
 * - 滑动距离阈值判断正确
 * - 双指缩放比例计算正确
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import { detectSwipe, calculatePinchScale, type SwipeDirection } from '@/composables/useTouch'

// ============================================================
// 辅助函数
// ============================================================

/**
 * 计算两点之间的距离
 */
function calculateDistance(x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1
  const dy = y2 - y1
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * 手动实现滑动方向检测（用于验证）
 */
function manualDetectSwipe(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  threshold: number
): SwipeDirection {
  const dx = endX - startX
  const dy = endY - startY
  const absDx = Math.abs(dx)
  const absDy = Math.abs(dy)
  
  // 未达到阈值
  if (absDx < threshold && absDy < threshold) {
    return null
  }
  
  // 判断主要方向
  if (absDx > absDy) {
    return dx > 0 ? 'right' : 'left'
  } else {
    return dy > 0 ? 'down' : 'up'
  }
}

// ============================================================
// 属性测试
// ============================================================

describe('触摸手势处理属性测试', () => {
  /**
   * **Property 44: 触摸手势处理一致性**
   * **Validates: Requirements 25.3**
   */
  describe('Property 44: 触摸手势处理一致性', () => {
    // 生成坐标值
    const coordArb = fc.integer({ min: 0, max: 2000 })
    
    // 生成阈值
    const thresholdArb = fc.integer({ min: 10, max: 200 })
    
    describe('滑动方向检测', () => {
      it('水平滑动应该正确检测左右方向', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            fc.integer({ min: 60, max: 500 }), // 水平位移（大于默认阈值50）
            fc.integer({ min: -20, max: 20 }), // 垂直位移（小于水平位移）
            (startX, startY, horizontalDelta, verticalDelta) => {
              // 确保水平位移大于垂直位移
              if (Math.abs(horizontalDelta) <= Math.abs(verticalDelta)) return true
              
              const endX = startX + horizontalDelta
              const endY = startY + verticalDelta
              
              const direction = detectSwipe(startX, startY, endX, endY)
              
              if (horizontalDelta > 0) {
                expect(direction).toBe('right')
              } else {
                expect(direction).toBe('left')
              }
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('垂直滑动应该正确检测上下方向', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            fc.integer({ min: -20, max: 20 }), // 水平位移（小于垂直位移）
            fc.integer({ min: 60, max: 500 }), // 垂直位移（大于默认阈值50）
            (startX, startY, horizontalDelta, verticalDelta) => {
              // 确保垂直位移大于水平位移
              if (Math.abs(verticalDelta) <= Math.abs(horizontalDelta)) return true
              
              const endX = startX + horizontalDelta
              const endY = startY + verticalDelta
              
              const direction = detectSwipe(startX, startY, endX, endY)
              
              if (verticalDelta > 0) {
                expect(direction).toBe('down')
              } else {
                expect(direction).toBe('up')
              }
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('小于阈值的移动应该返回 null', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            fc.integer({ min: -40, max: 40 }), // 小于默认阈值50
            fc.integer({ min: -40, max: 40 }), // 小于默认阈值50
            (startX, startY, dx, dy) => {
              const endX = startX + dx
              const endY = startY + dy
              
              const direction = detectSwipe(startX, startY, endX, endY)
              
              // 如果两个方向的位移都小于阈值，应该返回 null
              if (Math.abs(dx) < 50 && Math.abs(dy) < 50) {
                expect(direction).toBeNull()
              }
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('自定义阈值应该正确生效', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            thresholdArb,
            fc.integer({ min: -300, max: 300 }),
            fc.integer({ min: -300, max: 300 }),
            (startX, startY, threshold, dx, dy) => {
              const endX = startX + dx
              const endY = startY + dy
              
              const direction = detectSwipe(startX, startY, endX, endY, threshold)
              
              // 如果两个方向的位移都小于阈值，应该返回 null
              if (Math.abs(dx) < threshold && Math.abs(dy) < threshold) {
                expect(direction).toBeNull()
              }
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('滑动方向检测应该与手动实现一致', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            coordArb,
            coordArb,
            thresholdArb,
            (startX, startY, endX, endY, threshold) => {
              const result = detectSwipe(startX, startY, endX, endY, threshold)
              const expected = manualDetectSwipe(startX, startY, endX, endY, threshold)
              
              expect(result).toBe(expected)
            }
          ),
          { numRuns: 100 }
        )
      })
    })
    
    describe('双指缩放比例计算', () => {
      it('缩放比例应该正确计算', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: 10, max: 1000 }), // 初始距离
            fc.integer({ min: 10, max: 1000 }), // 当前距离
            (initialDistance, currentDistance) => {
              const scale = calculatePinchScale(initialDistance, currentDistance)
              const expectedScale = currentDistance / initialDistance
              
              expect(scale).toBeCloseTo(expectedScale, 10)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('初始距离为0时应该返回1', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: 0, max: 1000 }),
            (currentDistance) => {
              const scale = calculatePinchScale(0, currentDistance)
              expect(scale).toBe(1)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('负初始距离应该返回1', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: -1000, max: -1 }),
            fc.integer({ min: 0, max: 1000 }),
            (initialDistance, currentDistance) => {
              const scale = calculatePinchScale(initialDistance, currentDistance)
              expect(scale).toBe(1)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('相同距离应该返回1', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: 1, max: 1000 }),
            (distance) => {
              const scale = calculatePinchScale(distance, distance)
              expect(scale).toBe(1)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('放大操作应该返回大于1的比例', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: 10, max: 500 }),
            fc.integer({ min: 1, max: 500 }),
            (initialDistance, additionalDistance) => {
              const currentDistance = initialDistance + additionalDistance
              const scale = calculatePinchScale(initialDistance, currentDistance)
              
              expect(scale).toBeGreaterThan(1)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('缩小操作应该返回小于1的比例', () => {
        fc.assert(
          fc.property(
            fc.integer({ min: 100, max: 1000 }),
            fc.integer({ min: 1, max: 99 }),
            (initialDistance, reductionPercent) => {
              // 确保当前距离小于初始距离
              const currentDistance = Math.floor(initialDistance * (1 - reductionPercent / 100))
              if (currentDistance <= 0) return true
              
              const scale = calculatePinchScale(initialDistance, currentDistance)
              
              expect(scale).toBeLessThan(1)
              expect(scale).toBeGreaterThan(0)
            }
          ),
          { numRuns: 100 }
        )
      })
    })
    
    describe('距离计算', () => {
      it('两点距离应该正确计算', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            coordArb,
            coordArb,
            (x1, y1, x2, y2) => {
              const distance = calculateDistance(x1, y1, x2, y2)
              
              // 距离应该非负
              expect(distance).toBeGreaterThanOrEqual(0)
              
              // 验证勾股定理
              const dx = x2 - x1
              const dy = y2 - y1
              const expectedDistance = Math.sqrt(dx * dx + dy * dy)
              
              expect(distance).toBeCloseTo(expectedDistance, 10)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('同一点的距离应该为0', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            (x, y) => {
              const distance = calculateDistance(x, y, x, y)
              expect(distance).toBe(0)
            }
          ),
          { numRuns: 100 }
        )
      })
      
      it('距离应该满足对称性', () => {
        fc.assert(
          fc.property(
            coordArb,
            coordArb,
            coordArb,
            coordArb,
            (x1, y1, x2, y2) => {
              const distance1 = calculateDistance(x1, y1, x2, y2)
              const distance2 = calculateDistance(x2, y2, x1, y1)
              
              expect(distance1).toBeCloseTo(distance2, 10)
            }
          ),
          { numRuns: 100 }
        )
      })
    })
  })
  
  describe('边界情况测试', () => {
    // 生成坐标值
    const coordArb = fc.integer({ min: 0, max: 2000 })
    
    it('起点和终点相同时应该返回 null', () => {
      fc.assert(
        fc.property(
          coordArb,
          coordArb,
          (x, y) => {
            const direction = detectSwipe(x, y, x, y)
            expect(direction).toBeNull()
          }
        ),
        { numRuns: 100 }
      )
    })
    
    it('对角线滑动应该返回主要方向', () => {
      // 45度对角线滑动，水平和垂直位移相等时，应该返回垂直方向
      const direction = detectSwipe(0, 0, 100, 100)
      // 当 absDx === absDy 时，由于 absDx > absDy 为 false，返回垂直方向
      expect(direction).toBe('down')
    })
    
    it('刚好达到阈值时应该检测到滑动', () => {
      // 刚好达到默认阈值50
      const direction = detectSwipe(0, 0, 50, 0)
      expect(direction).toBe('right')
    })
    
    it('刚好未达到阈值时应该返回 null', () => {
      // 刚好未达到默认阈值50
      const direction = detectSwipe(0, 0, 49, 0)
      expect(direction).toBeNull()
    })
  })
})
