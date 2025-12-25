/**
 * 编辑模式属性测试
 * 
 * Feature: vue-frontend-migration
 * Property 9: 图片切换状态保存一致性
 * Property 28: 气泡拖拽移动一致性
 * Property 29: 气泡大小调整一致性
 * 
 * Validates: Requirements 30.3, 17.5
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import * as fc from 'fast-check'
import { useBubbleStore, createBubbleState } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleCoords } from '@/types/bubble'

// ============================================================
// 测试数据生成器
// ============================================================

/**
 * 生成有效的气泡坐标
 */
const bubbleCoordsArb = fc.tuple(
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 0, max: 800 }),
  fc.integer({ min: 50, max: 1000 }),
  fc.integer({ min: 50, max: 1000 })
).map(([x1, y1, w, h]): BubbleCoords => {
  // 确保有效的矩形坐标
  return [x1, y1, x1 + w, y1 + h]
})

/**
 * 生成拖拽偏移量
 */
const dragDeltaArb = fc.record({
  deltaX: fc.integer({ min: -500, max: 500 }),
  deltaY: fc.integer({ min: -500, max: 500 })
})

/**
 * 生成调整大小的手柄类型
 */
const resizeHandleArb = fc.constantFrom(
  'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'
)

/**
 * 生成图片尺寸
 */
const imageSizeArb = fc.record({
  width: fc.integer({ min: 500, max: 2000 }),
  height: fc.integer({ min: 500, max: 2000 })
})

// ============================================================
// 辅助函数
// ============================================================

/**
 * 计算拖拽后的新坐标
 * @param coords 原始坐标
 * @param deltaX X轴偏移
 * @param deltaY Y轴偏移
 * @param imageWidth 图片宽度
 * @param imageHeight 图片高度
 * @returns 新坐标
 */
function calculateDraggedCoords(
  coords: BubbleCoords,
  deltaX: number,
  deltaY: number,
  imageWidth: number,
  imageHeight: number
): BubbleCoords {
  const [x1, y1, x2, y2] = coords
  const width = x2 - x1
  const height = y2 - y1

  // 计算新位置
  let newX1 = Math.round(x1 + deltaX)
  let newY1 = Math.round(y1 + deltaY)

  // 边界约束
  const safeWidth = Math.min(width, imageWidth)
  const safeHeight = Math.min(height, imageHeight)

  newX1 = Math.max(0, Math.min(newX1, imageWidth - safeWidth))
  newY1 = Math.max(0, Math.min(newY1, imageHeight - safeHeight))

  return [newX1, newY1, newX1 + safeWidth, newY1 + safeHeight]
}

/**
 * 计算调整大小后的新坐标
 * @param coords 原始坐标
 * @param handle 调整手柄类型
 * @param deltaX X轴偏移
 * @param deltaY Y轴偏移
 * @param imageWidth 图片宽度
 * @param imageHeight 图片高度
 * @returns 新坐标或 null（如果尺寸无效）
 */
function calculateResizedCoords(
  coords: BubbleCoords,
  handle: string,
  deltaX: number,
  deltaY: number,
  imageWidth: number,
  imageHeight: number
): BubbleCoords | null {
  let [x1, y1, x2, y2] = coords

  // 根据手柄类型调整坐标
  if (handle.includes('w')) x1 += deltaX
  if (handle.includes('e')) x2 += deltaX
  if (handle.includes('n')) y1 += deltaY
  if (handle.includes('s')) y2 += deltaY

  // 确保有效性（交换坐标如果反转）
  if (x1 > x2) [x1, x2] = [x2, x1]
  if (y1 > y2) [y1, y2] = [y2, y1]

  // 边界约束
  x1 = Math.max(0, Math.round(x1))
  y1 = Math.max(0, Math.round(y1))
  x2 = Math.min(imageWidth, Math.round(x2))
  y2 = Math.min(imageHeight, Math.round(y2))

  // 最小尺寸检查
  const minSize = 10
  if (x2 - x1 < minSize || y2 - y1 < minSize) {
    return null
  }

  return [x1, y1, x2, y2]
}

// ============================================================
// 属性测试
// ============================================================

describe('编辑模式属性测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  /**
   * Property 9: 图片切换状态保存一致性
   * 
   * 对于任意气泡状态，切换图片前保存的状态应当在切换回来后能够正确恢复。
   * 
   * Feature: vue-frontend-migration, Property 9: 图片切换状态保存一致性
   * Validates: Requirements 30.3
   */
  it('图片切换时气泡状态应当正确保存和恢复', () => {
    fc.assert(
      fc.property(
        fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 5 }),
        fc.array(bubbleCoordsArb, { minLength: 1, maxLength: 5 }),
        (coords1, coords2) => {
          const bubbleStore = useBubbleStore()
          const imageStore = useImageStore()

          // 创建两张图片
          const image1 = {
            id: 'img1',
            fileName: 'test1.png',
            originalDataURL: 'data:image/png;base64,test1',
            translatedDataURL: null,
            cleanImageData: null,
            bubbleStates: coords1.map(c => createBubbleState(c)),
            bubbleCoords: coords1,
            bubbleTexts: coords1.map(() => ''),
            originalTexts: coords1.map(() => ''),
            bubbleAngles: coords1.map(() => 0),
            translationStatus: 'pending' as const,
            translationFailed: false,
            isManualAnnotation: false
          }

          const image2 = {
            id: 'img2',
            fileName: 'test2.png',
            originalDataURL: 'data:image/png;base64,test2',
            translatedDataURL: null,
            cleanImageData: null,
            bubbleStates: coords2.map(c => createBubbleState(c)),
            bubbleCoords: coords2,
            bubbleTexts: coords2.map(() => ''),
            originalTexts: coords2.map(() => ''),
            bubbleAngles: coords2.map(() => 0),
            translationStatus: 'pending' as const,
            translationFailed: false,
            isManualAnnotation: false
          }

          // 添加图片
          imageStore.addImage('test1.png', 'data:image/png;base64,test1', {
            bubbleStates: image1.bubbleStates,
            bubbleCoords: image1.bubbleCoords
          })
          imageStore.addImage('test2.png', 'data:image/png;base64,test2', {
            bubbleStates: image2.bubbleStates,
            bubbleCoords: image2.bubbleCoords
          })

          // 加载第一张图片的气泡状态
          bubbleStore.setBubbles([...image1.bubbleStates!])

          // 修改第一张图片的气泡状态
          if (bubbleStore.bubbles.length > 0) {
            bubbleStore.updateBubble(0, { translatedText: '修改后的文本' })
          }

          // 保存到图片
          imageStore.updateCurrentBubbleStates([...bubbleStore.bubbles])

          // 切换到第二张图片
          imageStore.setCurrentImageIndex(1)
          bubbleStore.setBubbles([...image2.bubbleStates!])

          // 验证第二张图片的气泡数量
          expect(bubbleStore.bubbles.length).toBe(coords2.length)

          // 切换回第一张图片
          imageStore.setCurrentImageIndex(0)
          const savedStates = imageStore.currentImage?.bubbleStates
          if (savedStates) {
            bubbleStore.setBubbles([...savedStates])
          }

          // 验证第一张图片的气泡数量
          expect(bubbleStore.bubbles.length).toBe(coords1.length)

          // 验证修改被保存
          if (bubbleStore.bubbles.length > 0) {
            expect(bubbleStore.bubbles[0]?.translatedText).toBe('修改后的文本')
          }

          return true
        }
      ),
      { numRuns: 50 }
    )
  })


  /**
   * Property 28: 气泡拖拽移动一致性
   * 
   * 对于任意气泡坐标和拖拽偏移量，拖拽后的坐标应当：
   * 1. 保持气泡尺寸不变
   * 2. 在图片边界内
   * 3. 正确反映偏移量（在边界允许范围内）
   * 
   * Feature: vue-frontend-migration, Property 28: 气泡拖拽移动一致性
   * Validates: Requirements 17.5
   */
  it('气泡拖拽后坐标应当正确更新且在边界内', () => {
    fc.assert(
      fc.property(
        bubbleCoordsArb,
        dragDeltaArb,
        imageSizeArb,
        (coords, delta, imageSize) => {
          const bubbleStore = useBubbleStore()

          // 创建气泡
          const bubble = createBubbleState(coords)
          bubbleStore.setBubbles([bubble])

          // 计算原始尺寸
          const originalWidth = coords[2] - coords[0]
          const originalHeight = coords[3] - coords[1]

          // 计算拖拽后的坐标
          const newCoords = calculateDraggedCoords(
            coords,
            delta.deltaX,
            delta.deltaY,
            imageSize.width,
            imageSize.height
          )

          // 更新坐标
          bubbleStore.updateBubbleCoords(0, newCoords)

          // 验证坐标已更新
          const updatedBubble = bubbleStore.bubbles[0]
          expect(updatedBubble).toBeDefined()

          if (updatedBubble) {
            const [x1, y1, x2, y2] = updatedBubble.coords

            // 验证尺寸保持不变（或在边界约束下尽可能保持）
            const newWidth = x2 - x1
            const newHeight = y2 - y1
            expect(newWidth).toBeLessThanOrEqual(originalWidth)
            expect(newHeight).toBeLessThanOrEqual(originalHeight)

            // 验证在图片边界内
            expect(x1).toBeGreaterThanOrEqual(0)
            expect(y1).toBeGreaterThanOrEqual(0)
            expect(x2).toBeLessThanOrEqual(imageSize.width)
            expect(y2).toBeLessThanOrEqual(imageSize.height)

            // 验证坐标有效性
            expect(x2).toBeGreaterThan(x1)
            expect(y2).toBeGreaterThan(y1)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 29: 气泡大小调整一致性
   * 
   * 对于任意气泡坐标、调整手柄和偏移量，调整后的坐标应当：
   * 1. 满足最小尺寸限制
   * 2. 在图片边界内
   * 3. 8个调整手柄方向正确
   * 
   * Feature: vue-frontend-migration, Property 29: 气泡大小调整一致性
   * Validates: Requirements 17.5
   */
  it('气泡大小调整后坐标应当正确更新且满足约束', () => {
    fc.assert(
      fc.property(
        bubbleCoordsArb,
        resizeHandleArb,
        dragDeltaArb,
        imageSizeArb,
        (coords, handle, delta, imageSize) => {
          const bubbleStore = useBubbleStore()

          // 创建气泡
          const bubble = createBubbleState(coords)
          bubbleStore.setBubbles([bubble])

          // 计算调整后的坐标
          const newCoords = calculateResizedCoords(
            coords,
            handle,
            delta.deltaX,
            delta.deltaY,
            imageSize.width,
            imageSize.height
          )

          // 如果新坐标有效，更新并验证
          if (newCoords) {
            bubbleStore.updateBubbleCoords(0, newCoords)

            const updatedBubble = bubbleStore.bubbles[0]
            expect(updatedBubble).toBeDefined()

            if (updatedBubble) {
              const [x1, y1, x2, y2] = updatedBubble.coords

              // 验证最小尺寸
              const minSize = 10
              expect(x2 - x1).toBeGreaterThanOrEqual(minSize)
              expect(y2 - y1).toBeGreaterThanOrEqual(minSize)

              // 验证在图片边界内
              expect(x1).toBeGreaterThanOrEqual(0)
              expect(y1).toBeGreaterThanOrEqual(0)
              expect(x2).toBeLessThanOrEqual(imageSize.width)
              expect(y2).toBeLessThanOrEqual(imageSize.height)

              // 验证坐标有效性
              expect(x2).toBeGreaterThan(x1)
              expect(y2).toBeGreaterThan(y1)
            }
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })


  /**
   * 8个调整手柄方向正确性测试
   */
  it('8个调整手柄应当按正确方向调整坐标', () => {
    const testCases = [
      { handle: 'nw', expectX1Change: true, expectY1Change: true, expectX2Change: false, expectY2Change: false },
      { handle: 'n', expectX1Change: false, expectY1Change: true, expectX2Change: false, expectY2Change: false },
      { handle: 'ne', expectX1Change: false, expectY1Change: true, expectX2Change: true, expectY2Change: false },
      { handle: 'e', expectX1Change: false, expectY1Change: false, expectX2Change: true, expectY2Change: false },
      { handle: 'se', expectX1Change: false, expectY1Change: false, expectX2Change: true, expectY2Change: true },
      { handle: 's', expectX1Change: false, expectY1Change: false, expectX2Change: false, expectY2Change: true },
      { handle: 'sw', expectX1Change: true, expectY1Change: false, expectX2Change: false, expectY2Change: true },
      { handle: 'w', expectX1Change: true, expectY1Change: false, expectX2Change: false, expectY2Change: false }
    ]

    for (const testCase of testCases) {
      const coords: BubbleCoords = [100, 100, 300, 300]
      const delta = 50
      const imageSize = { width: 1000, height: 1000 }

      // 计算调整后的坐标
      const newCoords = calculateResizedCoords(
        coords,
        testCase.handle,
        testCase.handle.includes('w') ? -delta : (testCase.handle.includes('e') ? delta : 0),
        testCase.handle.includes('n') ? -delta : (testCase.handle.includes('s') ? delta : 0),
        imageSize.width,
        imageSize.height
      )

      expect(newCoords).not.toBeNull()

      if (newCoords) {
        const [x1, y1, x2, y2] = newCoords

        // 验证各边是否按预期变化
        if (testCase.expectX1Change) {
          expect(x1).not.toBe(coords[0])
        }
        if (testCase.expectY1Change) {
          expect(y1).not.toBe(coords[1])
        }
        if (testCase.expectX2Change) {
          expect(x2).not.toBe(coords[2])
        }
        if (testCase.expectY2Change) {
          expect(y2).not.toBe(coords[3])
        }
      }
    }
  })

  /**
   * 拖拽边界限制测试
   */
  it('拖拽应当正确限制在图片边界内', () => {
    const coords: BubbleCoords = [100, 100, 200, 200]
    const imageSize = { width: 500, height: 500 }

    // 测试向左上角拖拽超出边界
    const newCoords1 = calculateDraggedCoords(coords, -200, -200, imageSize.width, imageSize.height)
    expect(newCoords1[0]).toBe(0)
    expect(newCoords1[1]).toBe(0)

    // 测试向右下角拖拽超出边界
    const newCoords2 = calculateDraggedCoords(coords, 500, 500, imageSize.width, imageSize.height)
    expect(newCoords2[2]).toBeLessThanOrEqual(imageSize.width)
    expect(newCoords2[3]).toBeLessThanOrEqual(imageSize.height)
  })

  /**
   * 调整大小最小尺寸限制测试
   */
  it('调整大小应当遵守最小尺寸限制', () => {
    const coords: BubbleCoords = [100, 100, 150, 150]
    const imageSize = { width: 1000, height: 1000 }

    // 尝试将宽度缩小到小于最小尺寸
    const newCoords = calculateResizedCoords(coords, 'e', -45, 0, imageSize.width, imageSize.height)
    
    // 应当返回 null 因为尺寸太小
    expect(newCoords).toBeNull()
  })

  /**
   * 气泡坐标更新后自动排版方向重新计算测试
   */
  it('更新坐标后应当重新计算自动排版方向', () => {
    const bubbleStore = useBubbleStore()

    // 创建一个宽大于高的气泡（应该是水平方向）
    const wideCoords: BubbleCoords = [0, 0, 200, 100]
    const bubble = createBubbleState(wideCoords)
    bubbleStore.setBubbles([bubble])

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('horizontal')

    // 更新为高大于宽的坐标（应该变成垂直方向）
    const tallCoords: BubbleCoords = [0, 0, 100, 200]
    bubbleStore.updateBubbleCoords(0, tallCoords)

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('vertical')
  })
})
