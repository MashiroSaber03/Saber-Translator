import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import EditImageComparison from '@/components/edit/EditImageComparison.vue'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { ImageData } from '@/types/image'

function createImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    ...TEXT_STYLE_DEFAULTS,
    id: 'page', fileName: 'manga.jpg', sourceAssetUrl: '/original.png',
    translatedAssetUrl: '/translated.png', cleanAssetUrl: '/clean.png',
    thumbnailSourceUrl: '/original.png', bubbleStates: [],
    translationStatus: 'completed', hasUnsavedChanges: false,
    ...overrides,
  }
}

function comparison(currentImage = createImage()) {
  return mount(EditImageComparison, {
    props: {
      currentImage, viewMode: 'dual', layoutMode: 'horizontal', bubbles: [],
      selectedBubble: null, selectedBubbleIndex: -1, selectedIndices: [],
      scale: 0.5, originalScale: 0.5, isDrawingMode: false, brushMode: null,
      currentImageWidth: 800, currentImageHeight: 1200, currentDrawingRect: null,
      drawingRectStyle: {}, originalTransformStyle: { transform: 'translate(10px, 20px) scale(0.5)' },
      translatedTransformStyle: { transform: 'translate(10px, 20px) scale(0.5)' },
      isOcrLoading: false, isTranslateLoading: false, isBusy: false,
    },
    global: { stubs: { BubbleEditor: true, BubbleOverlay: true } },
  })
}

const resultImage = '.edit-image-comparison__image-panel--translated .edit-image-comparison__image'

describe('clean image comparison', () => {
  it('switches the existing result image without changing the source or viewport', async () => {
    const wrapper = comparison()
    const transform = wrapper.get('.edit-image-comparison__image-panel--translated .edit-image-comparison__canvas-wrapper').attributes('style')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    await wrapper.setProps({ showCleanPreview: true })
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    expect(wrapper.get(resultImage).attributes('alt')).toBe('修复图 (无译文)')
    expect(wrapper.get('img[alt="原图"]').attributes('src')).toBe('/original.png')
    expect(wrapper.get('.edit-image-comparison__image-panel--translated .edit-image-comparison__canvas-wrapper').attributes('style')).toBe(transform)
    await wrapper.setProps({ showCleanPreview: false })
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    expect(wrapper.get(resultImage).attributes('alt')).toBe('翻译图 (中文)')
  })

  it.each([
    { translatedAssetUrl: '/translated.png', expectedUrl: '/translated.png', expectedLabel: '翻译图 (中文)' },
    { translatedAssetUrl: null, expectedUrl: '/original.png', expectedLabel: '结果图' },
  ])('falls back safely on a page without a clean asset: $expectedLabel', async ({ translatedAssetUrl, expectedUrl, expectedLabel }) => {
    const wrapper = comparison(createImage({ cleanAssetUrl: null, translatedAssetUrl }))
    await wrapper.setProps({ showCleanPreview: true })
    expect(wrapper.get(resultImage).attributes('src')).toBe(expectedUrl)
    expect(wrapper.get(resultImage).attributes('alt')).toBe(expectedLabel)
  })

  it('uses the latest clean asset while retaining the repair range overlay', async () => {
    const wrapper = comparison()
    await wrapper.setProps({ showCleanPreview: true, maskMode: true, maskOverlay: '/overlay.png' })
    await wrapper.setProps({ currentImage: createImage({ cleanAssetUrl: '/clean-new.png' }) })
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean-new.png')
    expect(wrapper.get('img[alt="结果图修复范围覆盖层"]').attributes('src')).toBe('/overlay.png')
    expect(wrapper.get('img[alt="原图修复范围覆盖层"]').attributes('src')).toBe('/overlay.png')
  })
})
