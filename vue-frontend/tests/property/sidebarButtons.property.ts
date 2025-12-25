/**
 * 侧边栏按钮禁用状态属性测试
 * 
 * **Feature: vue-frontend-migration, Property 48: 侧边栏按钮禁用状态一致性**
 * **Validates: Requirements 4.4, 4.5**
 * 
 * 测试内容：
 * - 无图片时相关按钮正确禁用
 * - 批量翻译进行中时按钮状态正确
 * - 翻译失败时重试按钮正确显示
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 类型定义
// ============================================================

/**
 * 图片状态
 */
interface ImageState {
  /** 图片ID */
  id: string
  /** 翻译状态 */
  translationStatus: 'pending' | 'processing' | 'completed' | 'failed'
  /** 是否翻译失败 */
  translationFailed: boolean
}

/**
 * 侧边栏状态
 */
interface SidebarState {
  /** 图片列表 */
  images: ImageState[]
  /** 当前图片索引 */
  currentImageIndex: number
  /** 批量翻译是否进行中 */
  isBatchTranslationInProgress: boolean
  /** 批量翻译是否暂停 */
  isBatchTranslationPaused: boolean
}

/**
 * 按钮禁用状态
 */
interface ButtonDisabledState {
  /** 翻译当前图片按钮 */
  translateCurrent: boolean
  /** 翻译所有图片按钮 */
  translateAll: boolean
  /** 高质量翻译按钮 */
  hqTranslate: boolean
  /** AI校对按钮 */
  proofread: boolean
  /** 仅消除文字按钮 */
  removeText: boolean
  /** 消除所有图片文字按钮 */
  removeAllText: boolean
  /** 删除当前图片按钮 */
  deleteCurrent: boolean
  /** 清除所有图片按钮 */
  clearAll: boolean
  /** 上一张按钮 */
  previous: boolean
  /** 下一张按钮 */
  next: boolean
}

// ============================================================
// 按钮状态计算函数（模拟组件逻辑）
// ============================================================

/**
 * 计算按钮禁用状态
 * @param state 侧边栏状态
 * @returns 按钮禁用状态
 */
function calculateButtonDisabledState(state: SidebarState): ButtonDisabledState {
  const hasImages = state.images.length > 0
  const currentImage = state.currentImageIndex >= 0 && state.currentImageIndex < state.images.length
    ? state.images[state.currentImageIndex]
    : null
  const canTranslate = hasImages && !state.isBatchTranslationInProgress
  const canGoPrevious = state.currentImageIndex > 0
  const canGoNext = state.currentImageIndex < state.images.length - 1

  return {
    translateCurrent: !canTranslate,
    translateAll: !canTranslate,
    hqTranslate: !canTranslate,
    proofread: !canTranslate,
    removeText: !currentImage,
    removeAllText: !hasImages,
    deleteCurrent: !currentImage,
    clearAll: !hasImages,
    previous: !canGoPrevious,
    next: !canGoNext
  }
}

/**
 * 检查是否有翻译失败的图片
 * @param state 侧边栏状态
 * @returns 是否有失败图片
 */
function hasFailedImages(state: SidebarState): boolean {
  return state.images.some(img => img.translationFailed)
}

/**
 * 获取失败图片数量
 * @param state 侧边栏状态
 * @returns 失败图片数量
 */
function getFailedImageCount(state: SidebarState): number {
  return state.images.filter(img => img.translationFailed).length
}

/**
 * 检查重试按钮是否应该显示
 * @param state 侧边栏状态
 * @returns 是否显示重试按钮
 */
function shouldShowRetryButton(state: SidebarState): boolean {
  return hasFailedImages(state) && !state.isBatchTranslationInProgress
}

// ============================================================
// 生成器
// ============================================================

/**
 * 生成翻译状态
 */
const translationStatusArb = fc.constantFrom(
  'pending' as const,
  'processing' as const,
  'completed' as const,
  'failed' as const
)

/**
 * 生成图片状态
 */
const imageStateArb = fc.record({
  id: fc.uuid(),
  translationStatus: translationStatusArb,
  translationFailed: fc.boolean()
}).map(img => ({
  ...img,
  // 确保 translationFailed 与 translationStatus 一致
  translationFailed: img.translationStatus === 'failed'
}))

// ============================================================
// 属性测试
// ============================================================

describe('侧边栏按钮禁用状态属性测试', () => {
  /**
   * Property 48.1: 无图片时翻译相关按钮正确禁用
   */
  it('Property 48.1: 无图片时翻译相关按钮正确禁用', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        (isBatchInProgress, isPaused) => {
          const state: SidebarState = {
            images: [],
            currentImageIndex: -1,
            isBatchTranslationInProgress: isBatchInProgress,
            isBatchTranslationPaused: isBatchInProgress ? isPaused : false
          }

          const buttonState = calculateButtonDisabledState(state)

          // 无图片时，所有翻译相关按钮都应禁用
          expect(buttonState.translateCurrent).toBe(true)
          expect(buttonState.translateAll).toBe(true)
          expect(buttonState.hqTranslate).toBe(true)
          expect(buttonState.proofread).toBe(true)
          expect(buttonState.removeText).toBe(true)
          expect(buttonState.removeAllText).toBe(true)
          expect(buttonState.deleteCurrent).toBe(true)
          expect(buttonState.clearAll).toBe(true)
          expect(buttonState.previous).toBe(true)
          expect(buttonState.next).toBe(true)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.2: 有图片时基本按钮正确启用
   */
  it('Property 48.2: 有图片时基本按钮正确启用', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        (images) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: false,
            isBatchTranslationPaused: false
          }

          const buttonState = calculateButtonDisabledState(state)

          // 有图片且不在批量翻译中时，翻译按钮应启用
          expect(buttonState.translateCurrent).toBe(false)
          expect(buttonState.translateAll).toBe(false)
          expect(buttonState.hqTranslate).toBe(false)
          expect(buttonState.proofread).toBe(false)
          expect(buttonState.removeText).toBe(false)
          expect(buttonState.removeAllText).toBe(false)
          expect(buttonState.deleteCurrent).toBe(false)
          expect(buttonState.clearAll).toBe(false)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.3: 批量翻译进行中时翻译按钮正确禁用
   */
  it('Property 48.3: 批量翻译进行中时翻译按钮正确禁用', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.boolean(),
        (images, isPaused) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: true,
            isBatchTranslationPaused: isPaused
          }

          const buttonState = calculateButtonDisabledState(state)

          // 批量翻译进行中时，翻译按钮应禁用
          expect(buttonState.translateCurrent).toBe(true)
          expect(buttonState.translateAll).toBe(true)
          expect(buttonState.hqTranslate).toBe(true)
          expect(buttonState.proofread).toBe(true)

          // 但删除和清除按钮仍可用
          expect(buttonState.removeText).toBe(false)
          expect(buttonState.removeAllText).toBe(false)
          expect(buttonState.deleteCurrent).toBe(false)
          expect(buttonState.clearAll).toBe(false)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.4: 导航按钮状态与图片索引一致
   */
  it('Property 48.4: 导航按钮状态与图片索引一致', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.nat({ max: 9 }),
        (images, indexOffset) => {
          const currentIndex = Math.min(indexOffset, images.length - 1)
          const state: SidebarState = {
            images,
            currentImageIndex: currentIndex,
            isBatchTranslationInProgress: false,
            isBatchTranslationPaused: false
          }

          const buttonState = calculateButtonDisabledState(state)

          // 第一张图片时，上一张按钮应禁用
          if (currentIndex === 0) {
            expect(buttonState.previous).toBe(true)
          } else {
            expect(buttonState.previous).toBe(false)
          }

          // 最后一张图片时，下一张按钮应禁用
          if (currentIndex === images.length - 1) {
            expect(buttonState.next).toBe(true)
          } else {
            expect(buttonState.next).toBe(false)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.5: 翻译失败时重试按钮正确显示
   */
  it('Property 48.5: 翻译失败时重试按钮正确显示', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.boolean(),
        (images, isBatchInProgress) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: isBatchInProgress,
            isBatchTranslationPaused: false
          }

          const hasFailed = hasFailedImages(state)
          const showRetry = shouldShowRetryButton(state)

          // 有失败图片且不在批量翻译中时，应显示重试按钮
          if (hasFailed && !isBatchInProgress) {
            expect(showRetry).toBe(true)
          } else if (!hasFailed) {
            expect(showRetry).toBe(false)
          } else {
            // 在批量翻译中时，不显示重试按钮
            expect(showRetry).toBe(false)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.6: 失败图片计数正确
   */
  it('Property 48.6: 失败图片计数正确', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 0, maxLength: 10 }),
        (images) => {
          const state: SidebarState = {
            images,
            currentImageIndex: images.length > 0 ? 0 : -1,
            isBatchTranslationInProgress: false,
            isBatchTranslationPaused: false
          }

          const failedCount = getFailedImageCount(state)
          const expectedCount = images.filter(img => img.translationFailed).length

          expect(failedCount).toBe(expectedCount)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.7: 单张图片时导航按钮都禁用
   */
  it('Property 48.7: 单张图片时导航按钮都禁用', () => {
    fc.assert(
      fc.property(
        imageStateArb,
        (image) => {
          const state: SidebarState = {
            images: [image],
            currentImageIndex: 0,
            isBatchTranslationInProgress: false,
            isBatchTranslationPaused: false
          }

          const buttonState = calculateButtonDisabledState(state)

          // 只有一张图片时，上一张和下一张都应禁用
          expect(buttonState.previous).toBe(true)
          expect(buttonState.next).toBe(true)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 48.8: 批量翻译暂停状态不影响按钮禁用逻辑
   */
  it('Property 48.8: 批量翻译暂停状态不影响按钮禁用逻辑', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        (images) => {
          // 创建两个状态：一个暂停，一个未暂停
          const statePaused: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: true,
            isBatchTranslationPaused: true
          }

          const stateNotPaused: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: true,
            isBatchTranslationPaused: false
          }

          const buttonStatePaused = calculateButtonDisabledState(statePaused)
          const buttonStateNotPaused = calculateButtonDisabledState(stateNotPaused)

          // 暂停状态不应影响按钮禁用逻辑
          expect(buttonStatePaused.translateCurrent).toBe(buttonStateNotPaused.translateCurrent)
          expect(buttonStatePaused.translateAll).toBe(buttonStateNotPaused.translateAll)
          expect(buttonStatePaused.hqTranslate).toBe(buttonStateNotPaused.hqTranslate)
          expect(buttonStatePaused.proofread).toBe(buttonStateNotPaused.proofread)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
