/**
 * 侧边栏工作流按钮禁用状态属性测试
 *
 * **Feature: vue-frontend-migration, Property 48: 侧边栏工作流禁用状态一致性**
 * **Validates: Requirements 4.4, 4.5**
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'

// ============================================================
// 类型定义
// ============================================================

type WorkflowMode =
  | 'translate-current'
  | 'translate-batch'
  | 'hq-batch'
  | 'proofread-batch'
  | 'remove-current'
  | 'remove-batch'
  | 'retry-failed'
  | 'delete-current'
  | 'clear-all'

interface ImageState {
  id: string
  translationStatus: 'pending' | 'processing' | 'completed' | 'failed'
  translationFailed: boolean
}

interface SidebarState {
  images: ImageState[]
  currentImageIndex: number
  isBatchTranslationInProgress: boolean
}

interface NavigationDisabledState {
  previous: boolean
  next: boolean
}

// ============================================================
// 逻辑函数（模拟组件）
// ============================================================

function hasCurrentImage(state: SidebarState): boolean {
  return state.currentImageIndex >= 0 && state.currentImageIndex < state.images.length
}

function hasImages(state: SidebarState): boolean {
  return state.images.length > 0
}

function hasFailedImages(state: SidebarState): boolean {
  return state.images.some(img => img.translationFailed)
}

function canTranslate(state: SidebarState): boolean {
  return hasImages(state) && !state.isBatchTranslationInProgress
}

function supportsPageSelection(mode: WorkflowMode): boolean {
  return mode === 'translate-batch'
    || mode === 'hq-batch'
    || mode === 'proofread-batch'
    || mode === 'remove-batch'
}

/**
 * 计算「启动按钮」是否禁用
 */
function isRunWorkflowDisabled(
  state: SidebarState,
  mode: WorkflowMode,
  isPageSelectionEnabled: boolean,
  hasValidPageSelection: boolean
): boolean {
  const pageSelectionInvalid = supportsPageSelection(mode) && isPageSelectionEnabled && !hasValidPageSelection

  switch (mode) {
    case 'translate-current':
      return !(hasCurrentImage(state) && canTranslate(state))
    case 'translate-batch':
    case 'hq-batch':
    case 'proofread-batch':
      return !(canTranslate(state) && !pageSelectionInvalid)
    case 'remove-current':
    case 'delete-current':
      return !hasCurrentImage(state)
    case 'remove-batch':
      return !(hasImages(state) && !pageSelectionInvalid)
    case 'clear-all':
      return !hasImages(state)
    case 'retry-failed':
      return !(hasFailedImages(state) && !state.isBatchTranslationInProgress)
    default:
      return true
  }
}

function calculateNavigationDisabledState(state: SidebarState): NavigationDisabledState {
  return {
    previous: state.currentImageIndex <= 0,
    next: state.currentImageIndex >= state.images.length - 1
  }
}

// ============================================================
// 生成器
// ============================================================

const workflowModeArb = fc.constantFrom<WorkflowMode>(
  'translate-current',
  'translate-batch',
  'hq-batch',
  'proofread-batch',
  'remove-current',
  'remove-batch',
  'retry-failed',
  'delete-current',
  'clear-all'
)

const imageStateArb = fc.record({
  id: fc.uuid(),
  translationStatus: fc.constantFrom('pending' as const, 'processing' as const, 'completed' as const, 'failed' as const),
  translationFailed: fc.boolean()
}).map(img => ({
  ...img,
  translationFailed: img.translationStatus === 'failed'
}))

// ============================================================
// 属性测试
// ============================================================

describe('侧边栏工作流按钮禁用状态属性测试', () => {
  it('Property 48.1: 无图片时启动按钮在所有模式都禁用', () => {
    fc.assert(
      fc.property(
        workflowModeArb,
        fc.boolean(),
        fc.boolean(),
        (mode, isBatchInProgress, isPageSelectionEnabled) => {
          const state: SidebarState = {
            images: [],
            currentImageIndex: -1,
            isBatchTranslationInProgress: isBatchInProgress
          }

          const disabled = isRunWorkflowDisabled(state, mode, isPageSelectionEnabled, true)
          expect(disabled).toBe(true)
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('Property 48.2: 有图片且非批量中时，基础流程模式可用', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.boolean(),
        (images, isPageSelectionEnabled) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: false
          }

          expect(isRunWorkflowDisabled(state, 'translate-current', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'translate-batch', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'hq-batch', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'proofread-batch', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'remove-current', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'remove-batch', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'delete-current', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'clear-all', isPageSelectionEnabled, true)).toBe(false)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('Property 48.3: 批量翻译进行中时翻译类模式禁用', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.boolean(),
        (images, isPageSelectionEnabled) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: true
          }

          expect(isRunWorkflowDisabled(state, 'translate-current', isPageSelectionEnabled, true)).toBe(true)
          expect(isRunWorkflowDisabled(state, 'translate-batch', isPageSelectionEnabled, true)).toBe(true)
          expect(isRunWorkflowDisabled(state, 'hq-batch', isPageSelectionEnabled, true)).toBe(true)
          expect(isRunWorkflowDisabled(state, 'proofread-batch', isPageSelectionEnabled, true)).toBe(true)
          expect(isRunWorkflowDisabled(state, 'retry-failed', isPageSelectionEnabled, true)).toBe(true)

          // 危险操作和消字当前页不受批量锁影响（与组件逻辑一致）
          expect(isRunWorkflowDisabled(state, 'remove-current', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'delete-current', isPageSelectionEnabled, true)).toBe(false)
          expect(isRunWorkflowDisabled(state, 'clear-all', isPageSelectionEnabled, true)).toBe(false)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('Property 48.4: 页码选择为空时仅影响支持页码选择的批量模式', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        workflowModeArb,
        (images, mode) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: false
          }

          const disabledWithEmptySelection = isRunWorkflowDisabled(state, mode, true, false)
          const disabledWithValidSelection = isRunWorkflowDisabled(state, mode, true, true)

          if (supportsPageSelection(mode)) {
            expect(disabledWithEmptySelection).toBe(true)
            expect(disabledWithValidSelection).toBe(false)
          } else {
            expect(disabledWithEmptySelection).toBe(disabledWithValidSelection)
          }

          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('Property 48.5: 失败重试模式仅在有失败图片且非批量中可用', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.boolean(),
        (images, isBatchInProgress) => {
          const state: SidebarState = {
            images,
            currentImageIndex: 0,
            isBatchTranslationInProgress: isBatchInProgress
          }

          const disabled = isRunWorkflowDisabled(state, 'retry-failed', false, true)
          const shouldBeDisabled = !hasFailedImages(state) || isBatchInProgress

          expect(disabled).toBe(shouldBeDisabled)
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  it('Property 48.6: 导航按钮状态与图片索引一致', () => {
    fc.assert(
      fc.property(
        fc.array(imageStateArb, { minLength: 1, maxLength: 10 }),
        fc.nat({ max: 9 }),
        (images, indexOffset) => {
          const currentIndex = Math.min(indexOffset, images.length - 1)
          const state: SidebarState = {
            images,
            currentImageIndex: currentIndex,
            isBatchTranslationInProgress: false
          }

          const navState = calculateNavigationDisabledState(state)

          expect(navState.previous).toBe(currentIndex === 0)
          expect(navState.next).toBe(currentIndex === images.length - 1)

          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
