import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import EditToolbar from '@/components/edit/EditToolbar.vue'

function createToolbarProps() {
  return {
    currentImageIndex: 0,
    imageCount: 3,
    canGoPrevious: false,
    canGoNext: true,
    showThumbnails: false,
    hasBubbles: true,
    selectedBubbleIndex: 0,
    bubbleCount: 2,
    layoutMode: 'horizontal' as const,
    syncEnabled: false,
    scale: 1,
    isDrawingMode: false,
    hasSelection: true,
    brushMode: null,
    brushSize: 24,
    mouseX: 10,
    mouseY: 20,
    isProcessing: true,
    progressText: '处理中',
    progressCurrent: 1,
    progressTotal: 4,
    isRepairLoading: false,
  }
}

describe('EditToolbar accessibility', () => {
  it('uses an explicit button for the image indicator thumbnail toggle', async () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const indicator = wrapper.get('.image-indicator')
    expect(indicator.element.tagName).toBe('BUTTON')
    expect(indicator.attributes('aria-label')).toBe('显示或隐藏缩略图')

    await indicator.trigger('click')
    expect(wrapper.emitted('toggle-thumbnails')).toHaveLength(1)
  })

  it('exposes edit processing progress through progressbar semantics', () => {
    const wrapper = mount(EditToolbar, {
      props: createToolbarProps(),
    })

    const progressbar = wrapper.get('[role="progressbar"]')
    expect(progressbar.attributes('aria-valuemin')).toBe('0')
    expect(progressbar.attributes('aria-valuemax')).toBe('4')
    expect(progressbar.attributes('aria-valuenow')).toBe('1')
    expect(progressbar.attributes('aria-label')).toBe('编辑处理进度')
  })
})
