/**
 * ImageViewer 组件属性测试
 *
 * Feature: vue-frontend-migration
 * Property 17: 图片缩放平移状态一致性
 *
 * Validates: Requirements 8.3
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 图片查看器核心逻辑（从组件中提取用于测试）
// ============================================================

/**
 * 图片查看器状态接口
 */
interface ViewerState {
  scale: number
  translateX: number
  translateY: number
}

/**
 * 图片查看器配置接口
 */
interface ViewerConfig {
  minScale: number
  maxScale: number
  zoomSpeed: number
}

/**
 * 在指定点缩放
 * @param state - 当前状态
 * @param config - 配置
 * @param x - 缩放中心 X 坐标
 * @param y - 缩放中心 Y 坐标
 * @param factor - 缩放因子
 * @returns 新状态
 */
function zoomAt(
  state: ViewerState,
  config: ViewerConfig,
  x: number,
  y: number,
  factor: number
): ViewerState {
  const newScale = Math.min(Math.max(state.scale * factor, config.minScale), config.maxScale)
  const scaleChange = newScale / state.scale

  return {
    scale: newScale,
    translateX: x - (x - state.translateX) * scaleChange,
    translateY: y - (y - state.translateY) * scaleChange
  }
}

/**
 * 以视口中心缩放
 * @param state - 当前状态
 * @param config - 配置
 * @param viewportWidth - 视口宽度
 * @param viewportHeight - 视口高度
 * @param factor - 缩放因子
 * @returns 新状态
 */
function zoom(
  state: ViewerState,
  config: ViewerConfig,
  viewportWidth: number,
  viewportHeight: number,
  factor: number
): ViewerState {
  return zoomAt(state, config, viewportWidth / 2, viewportHeight / 2, factor)
}

/**
 * 重置变换
 * @param initialScale - 初始缩放比例
 * @returns 重置后的状态
 */
function reset(initialScale: number = 1): ViewerState {
  return {
    scale: initialScale,
    translateX: 0,
    translateY: 0
  }
}

/**
 * 适应视口大小
 * @param viewportWidth - 视口宽度
 * @param viewportHeight - 视口高度
 * @param imageWidth - 图片宽度
 * @param imageHeight - 图片高度
 * @returns 适应后的状态
 */
function fitToViewport(
  viewportWidth: number,
  viewportHeight: number,
  imageWidth: number,
  imageHeight: number
): ViewerState {
  if (imageWidth <= 0 || imageHeight <= 0) {
    return reset()
  }

  const scaleX = viewportWidth / imageWidth
  const scaleY = viewportHeight / imageHeight
  const scale = Math.min(scaleX, scaleY) * 0.95 // 留5%边距

  return {
    scale,
    translateX: (viewportWidth - imageWidth * scale) / 2,
    translateY: (viewportHeight - imageHeight * scale) / 2
  }
}

/**
 * 平移
 * @param state - 当前状态
 * @param deltaX - X 方向偏移
 * @param deltaY - Y 方向偏移
 * @returns 新状态
 */
function pan(state: ViewerState, deltaX: number, deltaY: number): ViewerState {
  return {
    ...state,
    translateX: state.translateX + deltaX,
    translateY: state.translateY + deltaY
  }
}

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的缩放比例
 */
const scaleArb = fc.double({ min: 0.1, max: 5, noNaN: true })

/**
 * 生成有效的平移值
 */
const translateArb = fc.double({ min: -1000, max: 1000, noNaN: true })

/**
 * 生成有效的视口尺寸
 */
const dimensionArb = fc.integer({ min: 100, max: 2000 })

/**
 * 生成有效的缩放因子
 */
const zoomFactorArb = fc.double({ min: 0.5, max: 2, noNaN: true })

/**
 * 生成有效的查看器状态
 */
const viewerStateArb: fc.Arbitrary<ViewerState> = fc.record({
  scale: scaleArb,
  translateX: translateArb,
  translateY: translateArb
})

/**
 * 生成有效的查看器配置
 */
const viewerConfigArb: fc.Arbitrary<ViewerConfig> = fc.record({
  minScale: fc.double({ min: 0.01, max: 0.5, noNaN: true }),
  maxScale: fc.double({ min: 2, max: 10, noNaN: true }),
  zoomSpeed: fc.double({ min: 0.05, max: 0.3, noNaN: true })
})

// ============================================================
// Property 17: 图片缩放平移状态一致性
// ============================================================

describe('ImageViewer 属性测试', () => {
  describe('Property 17: 图片缩放平移状态一致性', () => {
    /**
     * 测试缩放后图片尺寸计算正确性
     *
     * Feature: vue-frontend-migration, Property 17: 图片缩放平移状态一致性
     * Validates: Requirements 8.3
     */
    it('缩放后 scale 应在 minScale 和 maxScale 范围内', () => {
      fc.assert(
        fc.property(viewerStateArb, viewerConfigArb, zoomFactorArb, (state, config, factor) => {
          const newState = zoom(state, config, 800, 600, factor)

          // 验证缩放比例在有效范围内
          expect(newState.scale).toBeGreaterThanOrEqual(config.minScale)
          expect(newState.scale).toBeLessThanOrEqual(config.maxScale)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试连续缩放的边界限制
     */
    it('连续放大不应超过 maxScale', () => {
      fc.assert(
        fc.property(viewerConfigArb, fc.integer({ min: 1, max: 50 }), (config, iterations) => {
          let state: ViewerState = { scale: 1, translateX: 0, translateY: 0 }

          // 连续放大
          for (let i = 0; i < iterations; i++) {
            state = zoom(state, config, 800, 600, 1.5)
          }

          // 验证不超过最大值
          expect(state.scale).toBeLessThanOrEqual(config.maxScale)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试连续缩小的边界限制
     */
    it('连续缩小不应低于 minScale', () => {
      fc.assert(
        fc.property(viewerConfigArb, fc.integer({ min: 1, max: 50 }), (config, iterations) => {
          let state: ViewerState = { scale: 1, translateX: 0, translateY: 0 }

          // 连续缩小
          for (let i = 0; i < iterations; i++) {
            state = zoom(state, config, 800, 600, 0.5)
          }

          // 验证不低于最小值
          expect(state.scale).toBeGreaterThanOrEqual(config.minScale)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试平移操作正确性
     */
    it('平移应正确更新 translateX 和 translateY', () => {
      fc.assert(
        fc.property(viewerStateArb, translateArb, translateArb, (state, deltaX, deltaY) => {
          const newState = pan(state, deltaX, deltaY)

          // 验证平移值正确更新
          expect(newState.translateX).toBeCloseTo(state.translateX + deltaX, 5)
          expect(newState.translateY).toBeCloseTo(state.translateY + deltaY, 5)

          // 验证缩放比例不变
          expect(newState.scale).toBe(state.scale)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试双击重置功能
     */
    it('reset 应将状态重置为初始值', () => {
      fc.assert(
        fc.property(scaleArb, (initialScale) => {
          const state = reset(initialScale)

          // 验证重置后的状态
          expect(state.scale).toBe(initialScale)
          expect(state.translateX).toBe(0)
          expect(state.translateY).toBe(0)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试适应视口功能
     */
    it('fitToViewport 应正确计算缩放和居中', () => {
      fc.assert(
        fc.property(dimensionArb, dimensionArb, dimensionArb, dimensionArb, (vw, vh, iw, ih) => {
          const state = fitToViewport(vw, vh, iw, ih)

          // 验证缩放比例为正数
          expect(state.scale).toBeGreaterThan(0)

          // 验证图片在视口内（考虑5%边距）
          const scaledWidth = iw * state.scale
          const scaledHeight = ih * state.scale
          expect(scaledWidth).toBeLessThanOrEqual(vw)
          expect(scaledHeight).toBeLessThanOrEqual(vh)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试在指定点缩放的中心点保持
     */
    it('zoomAt 应保持缩放中心点位置不变', () => {
      fc.assert(
        fc.property(
          viewerConfigArb,
          fc.double({ min: 100, max: 700, noNaN: true }),
          fc.double({ min: 100, max: 500, noNaN: true }),
          fc.double({ min: 1.1, max: 1.5, noNaN: true }),
          (config, x, y, factor) => {
            const initialState: ViewerState = { scale: 1, translateX: 0, translateY: 0 }
            const newState = zoomAt(initialState, config, x, y, factor)

            // 计算缩放前后中心点在图片上的位置
            // 缩放前：(x - translateX) / scale
            // 缩放后：(x - newTranslateX) / newScale
            // 这两个值应该相等

            const beforeX = (x - initialState.translateX) / initialState.scale
            const afterX = (x - newState.translateX) / newState.scale

            const beforeY = (y - initialState.translateY) / initialState.scale
            const afterY = (y - newState.translateY) / newState.scale

            // 允许小的浮点误差
            expect(afterX).toBeCloseTo(beforeX, 5)
            expect(afterY).toBeCloseTo(beforeY, 5)

            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * 测试缩放因子为1时状态不变
     * 注意：当 scale 在边界附近时，由于浮点精度问题可能有微小变化
     */
    it('缩放因子为1时状态应保持不变', () => {
      fc.assert(
        fc.property(viewerConfigArb, (config) => {
          // 使用一个明确在有效范围内的状态，避免边界情况
          const state: ViewerState = {
            scale: (config.minScale + config.maxScale) / 2, // 使用中间值
            translateX: 100,
            translateY: 100
          }
          const newState = zoom(state, config, 800, 600, 1)

          // 验证状态不变（使用较宽松的精度）
          expect(newState.scale).toBeCloseTo(state.scale, 8)
          expect(newState.translateX).toBeCloseTo(state.translateX, 8)
          expect(newState.translateY).toBeCloseTo(state.translateY, 8)

          return true
        }),
        { numRuns: 100 }
      )
    })

    /**
     * 测试平移的可逆性
     */
    it('平移操作应可逆', () => {
      fc.assert(
        fc.property(viewerStateArb, translateArb, translateArb, (state, deltaX, deltaY) => {
          // 平移
          const movedState = pan(state, deltaX, deltaY)
          // 反向平移
          const restoredState = pan(movedState, -deltaX, -deltaY)

          // 验证恢复到原始状态
          expect(restoredState.translateX).toBeCloseTo(state.translateX, 5)
          expect(restoredState.translateY).toBeCloseTo(state.translateY, 5)
          expect(restoredState.scale).toBe(state.scale)

          return true
        }),
        { numRuns: 100 }
      )
    })
  })

  describe('边界条件测试', () => {
    /**
     * 测试零尺寸图片
     */
    it('零尺寸图片应返回默认状态', () => {
      const state = fitToViewport(800, 600, 0, 0)
      expect(state.scale).toBe(1)
      expect(state.translateX).toBe(0)
      expect(state.translateY).toBe(0)
    })

    /**
     * 测试负尺寸图片
     */
    it('负尺寸图片应返回默认状态', () => {
      const state = fitToViewport(800, 600, -100, -100)
      expect(state.scale).toBe(1)
      expect(state.translateX).toBe(0)
      expect(state.translateY).toBe(0)
    })

    /**
     * 测试极端缩放因子
     */
    it('极端缩放因子应被限制在有效范围内', () => {
      const config: ViewerConfig = { minScale: 0.1, maxScale: 5, zoomSpeed: 0.1 }
      const state: ViewerState = { scale: 1, translateX: 0, translateY: 0 }

      // 极大的缩放因子
      const zoomedIn = zoom(state, config, 800, 600, 1000)
      expect(zoomedIn.scale).toBeLessThanOrEqual(config.maxScale)

      // 极小的缩放因子
      const zoomedOut = zoom(state, config, 800, 600, 0.001)
      expect(zoomedOut.scale).toBeGreaterThanOrEqual(config.minScale)
    })
  })
})
