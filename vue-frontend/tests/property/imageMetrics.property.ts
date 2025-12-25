/**
 * 图片显示指标计算属性测试
 * 
 * **Feature: vue-frontend-migration, Property 37: 图片显示指标计算一致性**
 * **Validates: Requirements 17.2**
 * 
 * 测试内容：
 * - 缩放比例计算正确性
 * - 坐标转换往返一致性（屏幕坐标↔图片坐标）
 * - 气泡坐标转换正确性
 * - 多边形坐标转换正确性
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import {
  imageToScreenCoords,
  screenToImageCoords,
  bubbleCoordsToScreen,
  screenCoordsToBubble,
  polygonToScreen,
  screenPolygonToImage,
  scaleSize,
  isPointInVisualContent,
  type ImageDisplayMetrics
} from '@/utils/imageMetrics'

/**
 * 生成有效的图片显示指标
 * 确保所有值都是正数且合理
 */
const validMetricsArb = fc.record({
  // 可视内容尺寸（正数）
  visualContentWidth: fc.integer({ min: 100, max: 2000 }),
  visualContentHeight: fc.integer({ min: 100, max: 2000 }),
  // 偏移量（可以是0或正数）
  visualContentOffsetX: fc.integer({ min: 0, max: 500 }),
  visualContentOffsetY: fc.integer({ min: 0, max: 500 }),
  // 原始尺寸（正数）
  naturalWidth: fc.integer({ min: 100, max: 4000 }),
  naturalHeight: fc.integer({ min: 100, max: 4000 }),
  // 元素尺寸（正数）
  elementWidth: fc.integer({ min: 100, max: 2000 }),
  elementHeight: fc.integer({ min: 100, max: 2000 })
}).map(m => ({
  ...m,
  // 根据可视内容和原始尺寸计算缩放比例
  scaleX: m.visualContentWidth / m.naturalWidth,
  scaleY: m.visualContentHeight / m.naturalHeight
})) as fc.Arbitrary<ImageDisplayMetrics>

/**
 * 生成有效的气泡坐标 [x1, y1, x2, y2]
 */
const validBubbleCoordsArb = (metrics: ImageDisplayMetrics) => 
  fc.tuple(
    fc.integer({ min: 0, max: Math.floor(metrics.naturalWidth * 0.8) }),
    fc.integer({ min: 0, max: Math.floor(metrics.naturalHeight * 0.8) }),
    fc.integer({ min: 0, max: metrics.naturalWidth }),
    fc.integer({ min: 0, max: metrics.naturalHeight })
  ).filter(([x1, y1, x2, y2]) => x1 < x2 && y1 < y2) as fc.Arbitrary<[number, number, number, number]>

/**
 * 生成有效的多边形坐标
 */
const validPolygonArb = (metrics: ImageDisplayMetrics) =>
  fc.array(
    fc.tuple(
      fc.integer({ min: 0, max: metrics.naturalWidth }),
      fc.integer({ min: 0, max: metrics.naturalHeight })
    ).map(([x, y]) => [x, y]),
    { minLength: 3, maxLength: 8 }
  )

describe('图片显示指标计算属性测试', () => {
  /**
   * Property 37.1: 坐标转换往返一致性
   * 图片坐标 → 屏幕坐标 → 图片坐标 应该得到原始值（允许浮点误差）
   */
  describe('坐标转换往返一致性', () => {
    it('图片坐标转屏幕坐标再转回应该得到原始值', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          fc.integer({ min: 0, max: 1000 }),
          fc.integer({ min: 0, max: 1000 }),
          (metrics, imageX, imageY) => {
            // 图片坐标 → 屏幕坐标
            const screenCoords = imageToScreenCoords(imageX, imageY, metrics)
            
            // 屏幕坐标 → 图片坐标
            const backToImage = screenToImageCoords(screenCoords.x, screenCoords.y, metrics)
            
            // 验证往返一致性（允许浮点误差）
            expect(backToImage.x).toBeCloseTo(imageX, 5)
            expect(backToImage.y).toBeCloseTo(imageY, 5)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('屏幕坐标转图片坐标再转回应该得到原始值', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          fc.integer({ min: 0, max: 1000 }),
          fc.integer({ min: 0, max: 1000 }),
          (metrics, screenX, screenY) => {
            // 屏幕坐标 → 图片坐标
            const imageCoords = screenToImageCoords(screenX, screenY, metrics)
            
            // 图片坐标 → 屏幕坐标
            const backToScreen = imageToScreenCoords(imageCoords.x, imageCoords.y, metrics)
            
            // 验证往返一致性（允许浮点误差）
            expect(backToScreen.x).toBeCloseTo(screenX, 5)
            expect(backToScreen.y).toBeCloseTo(screenY, 5)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 37.2: 缩放比例计算正确性
   * 转换后的坐标应该正确反映缩放比例
   */
  describe('缩放比例计算正确性', () => {
    it('屏幕坐标应该等于图片坐标乘以缩放比例加上偏移', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          fc.integer({ min: 0, max: 1000 }),
          fc.integer({ min: 0, max: 1000 }),
          (metrics, imageX, imageY) => {
            const screenCoords = imageToScreenCoords(imageX, imageY, metrics)
            
            // 验证公式：screenX = imageX * scaleX + offsetX
            const expectedX = imageX * metrics.scaleX + metrics.visualContentOffsetX
            const expectedY = imageY * metrics.scaleY + metrics.visualContentOffsetY
            
            expect(screenCoords.x).toBeCloseTo(expectedX, 10)
            expect(screenCoords.y).toBeCloseTo(expectedY, 10)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('scaleSize 应该正确缩放尺寸', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          fc.integer({ min: 1, max: 500 }),
          fc.integer({ min: 1, max: 500 }),
          (metrics, width, height) => {
            const scaled = scaleSize(width, height, metrics)
            
            expect(scaled.width).toBeCloseTo(width * metrics.scaleX, 10)
            expect(scaled.height).toBeCloseTo(height * metrics.scaleY, 10)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 37.3: 气泡坐标转换一致性
   * 气泡坐标转换应该保持矩形的相对位置关系
   */
  describe('气泡坐标转换一致性', () => {
    it('气泡坐标转换后应该保持矩形形状', () => {
      fc.assert(
        fc.property(
          validMetricsArb.chain(metrics => 
            validBubbleCoordsArb(metrics).map(coords => ({ metrics, coords }))
          ),
          ({ metrics, coords }) => {
            const screenCoords = bubbleCoordsToScreen(coords, metrics)
            
            // 验证转换后仍然是有效矩形（x1 < x2, y1 < y2）
            expect(screenCoords[0]).toBeLessThan(screenCoords[2])
            expect(screenCoords[1]).toBeLessThan(screenCoords[3])
            
            // 验证宽高比例正确
            const originalWidth = coords[2] - coords[0]
            const originalHeight = coords[3] - coords[1]
            const screenWidth = screenCoords[2] - screenCoords[0]
            const screenHeight = screenCoords[3] - screenCoords[1]
            
            expect(screenWidth).toBeCloseTo(originalWidth * metrics.scaleX, 5)
            expect(screenHeight).toBeCloseTo(originalHeight * metrics.scaleY, 5)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('气泡坐标往返转换应该得到原始值（允许取整误差）', () => {
      fc.assert(
        fc.property(
          validMetricsArb.chain(metrics => 
            validBubbleCoordsArb(metrics).map(coords => ({ metrics, coords }))
          ),
          ({ metrics, coords }) => {
            // 图片坐标 → 屏幕坐标 → 图片坐标
            const screenCoords = bubbleCoordsToScreen(coords, metrics)
            const backToImage = screenCoordsToBubble(screenCoords, metrics)
            
            // 由于 screenCoordsToBubble 会取整，允许 ±1 的误差
            expect(Math.abs(backToImage[0] - coords[0])).toBeLessThanOrEqual(1)
            expect(Math.abs(backToImage[1] - coords[1])).toBeLessThanOrEqual(1)
            expect(Math.abs(backToImage[2] - coords[2])).toBeLessThanOrEqual(1)
            expect(Math.abs(backToImage[3] - coords[3])).toBeLessThanOrEqual(1)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 37.4: 多边形坐标转换一致性
   */
  describe('多边形坐标转换一致性', () => {
    it('多边形顶点数量应该保持不变', () => {
      fc.assert(
        fc.property(
          validMetricsArb.chain(metrics => 
            validPolygonArb(metrics).map(polygon => ({ metrics, polygon }))
          ),
          ({ metrics, polygon }) => {
            const screenPolygon = polygonToScreen(polygon, metrics)
            
            expect(screenPolygon.length).toBe(polygon.length)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('多边形往返转换应该得到原始值（允许取整误差）', () => {
      fc.assert(
        fc.property(
          validMetricsArb.chain(metrics => 
            validPolygonArb(metrics).map(polygon => ({ metrics, polygon }))
          ),
          ({ metrics, polygon }) => {
            // 图片坐标 → 屏幕坐标 → 图片坐标
            const screenPolygon = polygonToScreen(polygon, metrics)
            const backToImage = screenPolygonToImage(screenPolygon, metrics)
            
            // 验证每个顶点（允许取整误差）
            for (let i = 0; i < polygon.length; i++) {
              const backPoint = backToImage[i]
              const origPoint = polygon[i]
              if (backPoint && origPoint && 
                  backPoint[0] !== undefined && backPoint[1] !== undefined &&
                  origPoint[0] !== undefined && origPoint[1] !== undefined) {
                expect(Math.abs(backPoint[0] - origPoint[0])).toBeLessThanOrEqual(1)
                expect(Math.abs(backPoint[1] - origPoint[1])).toBeLessThanOrEqual(1)
              }
            }
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 37.5: 点在可视区域内检测正确性
   */
  describe('点在可视区域内检测', () => {
    it('可视区域内的点应该返回 true', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          // 使用 integer 生成 0-100 的值，然后除以 100 得到 0-1 的比例
          // 避免 fc.float 可能生成 NaN 的问题
          fc.integer({ min: 0, max: 100 }),
          fc.integer({ min: 0, max: 100 }),
          (metrics, ratioXInt, ratioYInt) => {
            // 将整数转换为 0-1 的比例
            const ratioX = ratioXInt / 100
            const ratioY = ratioYInt / 100
            
            // 生成可视区域内的点
            const x = metrics.visualContentOffsetX + ratioX * metrics.visualContentWidth
            const y = metrics.visualContentOffsetY + ratioY * metrics.visualContentHeight
            
            expect(isPointInVisualContent(x, y, metrics)).toBe(true)
          }
        ),
        { numRuns: 100 }
      )
    })

    it('可视区域外的点应该返回 false', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          fc.integer({ min: 1, max: 100 }),
          (metrics, offset) => {
            // 生成可视区域外的点（左边）
            const leftX = metrics.visualContentOffsetX - offset
            expect(isPointInVisualContent(leftX, metrics.visualContentOffsetY, metrics)).toBe(false)
            
            // 生成可视区域外的点（上边）
            const topY = metrics.visualContentOffsetY - offset
            expect(isPointInVisualContent(metrics.visualContentOffsetX, topY, metrics)).toBe(false)
            
            // 生成可视区域外的点（右边）
            const rightX = metrics.visualContentOffsetX + metrics.visualContentWidth + offset
            expect(isPointInVisualContent(rightX, metrics.visualContentOffsetY, metrics)).toBe(false)
            
            // 生成可视区域外的点（下边）
            const bottomY = metrics.visualContentOffsetY + metrics.visualContentHeight + offset
            expect(isPointInVisualContent(metrics.visualContentOffsetX, bottomY, metrics)).toBe(false)
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 37.6: 边界条件处理
   */
  describe('边界条件处理', () => {
    it('零缩放比例时 screenToImageCoords 应该返回 (0, 0)', () => {
      const zeroScaleMetrics: ImageDisplayMetrics = {
        visualContentWidth: 0,
        visualContentHeight: 0,
        visualContentOffsetX: 100,
        visualContentOffsetY: 100,
        scaleX: 0,
        scaleY: 0,
        naturalWidth: 1000,
        naturalHeight: 1000,
        elementWidth: 500,
        elementHeight: 500
      }
      
      const result = screenToImageCoords(200, 200, zeroScaleMetrics)
      expect(result.x).toBe(0)
      expect(result.y).toBe(0)
    })

    it('原点坐标转换应该正确', () => {
      fc.assert(
        fc.property(
          validMetricsArb,
          (metrics) => {
            // 图片原点 (0, 0) 转换到屏幕
            const screenOrigin = imageToScreenCoords(0, 0, metrics)
            
            // 应该等于偏移量
            expect(screenOrigin.x).toBeCloseTo(metrics.visualContentOffsetX, 10)
            expect(screenOrigin.y).toBeCloseTo(metrics.visualContentOffsetY, 10)
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})
