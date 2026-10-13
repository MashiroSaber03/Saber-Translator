import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import EditImageComparison from '@/components/edit/EditImageComparison.vue'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { ImageData } from '@/types/image'

const toast = vi.hoisted(() => vi.fn())
vi.mock('@/utils/toast', () => ({ showToast: toast }))

function createImage(overrides: Partial<ImageData> = {}): ImageData {
  return {
    ...TEXT_STYLE_DEFAULTS,
    id: 'page', documentRevision: 1, renderedRevision: 1, fileName: 'manga.jpg', sourceAssetUrl: '/original.png',
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
const translatedLayer = '.edit-image-comparison__translated-overlay'

async function loadTextLayers(wrapper: ReturnType<typeof comparison>): Promise<void> {
  await wrapper.get('.edit-image-comparison__source-overlay').trigger('load')
  await wrapper.get(translatedLayer).trigger('load')
}

describe('clean image comparison', () => {
  it('keeps the source overlay aligned through opacity changes, clean preview and page changes', async () => {
    const wrapper = comparison()
    expect(wrapper.find('.edit-image-comparison__source-overlay').exists()).toBe(false)
    expect(wrapper.find(translatedLayer).exists()).toBe(false)
    await wrapper.setProps({ originalTextOpacity: 50 })
    let overlay = wrapper.get('.edit-image-comparison__source-overlay')
    expect(overlay.attributes('src')).toBe('/api/v2/pages/page/original-text-overlay?revision=1&protectTranslation=false')
    await loadTextLayers(wrapper)
    const resultWrapper = wrapper.get('.edit-image-comparison__image-panel--translated .edit-image-comparison__canvas-wrapper')
    const transform = resultWrapper.attributes('style')
    for (const opacity of [50, 100, 0]) {
      await wrapper.setProps({ originalTextOpacity: opacity })
      expect(overlay.attributes('style')).toContain(`opacity: ${opacity / 100}`)
      expect(overlay.attributes('style')?.includes('display: none')).toBe(opacity === 0)
      expect(resultWrapper.attributes('style')).toBe(transform)
      expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    }
    await wrapper.setProps({ originalTextOpacity: 50, showCleanPreview: true, maskMode: true, maskOverlay: '/mask.png' })
    overlay = wrapper.get('.edit-image-comparison__source-overlay')
    expect(overlay.attributes('src')).toContain('protectTranslation=false')
    await overlay.trigger('load')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    expect(wrapper.get(translatedLayer).element.nextElementSibling?.getAttribute('src')).toBe('/mask.png')
    expect(wrapper.get(translatedLayer).attributes('style')).toContain('display: none')
    expect(overlay.element.parentElement).toBe(resultWrapper.element)
    await wrapper.setProps({ currentImage: createImage({ id: 'page-2', sourceAssetUrl: '/page-2.png' }) })
    overlay = wrapper.get('.edit-image-comparison__source-overlay')
    expect(overlay.attributes('src')).toContain('/pages/page-2/original-text-overlay')
    expect(overlay.attributes('style')).toContain('opacity: 0.5')
    expect(wrapper.emitted('imageLoad')).toBeUndefined()
  })

  it('hides source comparison during color picking and restores it afterwards', async () => {
    const wrapper = comparison()
    await wrapper.setProps({ originalTextOpacity: 100 })
    const overlay = wrapper.get('.edit-image-comparison__source-overlay')
    await loadTextLayers(wrapper)
    expect(overlay.attributes('style')).not.toContain('display: none')
    await wrapper.setProps({ isPickingColor: true })
    expect(overlay.attributes('style')).toContain('display: none')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    await wrapper.setProps({ isPickingColor: false })
    expect(overlay.attributes('style')).not.toContain('display: none')
    expect(overlay.attributes('style')).toContain('opacity: 1')
    await wrapper.setProps({ viewMode: 'original' })
    expect(wrapper.find('.edit-image-comparison__source-overlay').exists()).toBe(false)
    await wrapper.setProps({ viewMode: 'translated' })
    const returning = wrapper.get('.edit-image-comparison__source-overlay')
    expect(returning.attributes('style')).toContain('display: none')
    await returning.trigger('load')
    await wrapper.get(translatedLayer).trigger('load')
    expect(returning.attributes('style')).not.toContain('display: none')
  })

  it('hides stale or failed text previews instead of falling back to a full source image', async () => {
    const wrapper = comparison()
    await wrapper.setProps({ originalTextOpacity: 100 })
    await wrapper.get('.edit-image-comparison__source-overlay').trigger('error')
    expect(toast).toHaveBeenCalledWith('原文预览加载失败，可将透明度归零后重试', 'error')
    expect(wrapper.get('.edit-image-comparison__source-overlay').attributes('style')).toContain('display: none')
    await wrapper.setProps({ currentImage: createImage({ documentRevision: 2, renderedRevision: 1 }) })
    expect(wrapper.find('.edit-image-comparison__source-overlay').exists()).toBe(false)
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    await wrapper.setProps({ currentImage: createImage({ documentRevision: 2, renderedRevision: 2, translatedAssetUrl: '/new.png' }) })
    const overlay = wrapper.get('.edit-image-comparison__source-overlay')
    expect(overlay.attributes('src')).toContain('revision=2')
    await loadTextLayers(wrapper)
    expect(overlay.attributes('style')).not.toContain('display: none')
  })

  it.each([false, true])('controls both text layers independently without fading the background (range=%s)', async maskMode => {
    const wrapper = comparison()
    await wrapper.setProps({ originalTextOpacity: 50, translatedTextOpacity: 25, maskMode, maskOverlay: '/mask.png' })
    await loadTextLayers(wrapper)
    const original = wrapper.get('.edit-image-comparison__source-overlay')
    const translated = wrapper.get(translatedLayer)
    const url = translated.attributes('src')
    expect(original.element.nextElementSibling).toBe(translated.element)
    expect(original.attributes('style')).toContain('opacity: 0.5')
    expect(translated.attributes('style')).toContain('opacity: 0.25')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    expect(wrapper.get(resultImage).attributes('style')).toBeUndefined()
    for (const opacity of [0, 50, 100]) {
      await wrapper.setProps({ translatedTextOpacity: opacity })
      expect(original.attributes('style')).toContain('opacity: 0.5')
      expect(translated.attributes('style')).toContain(`opacity: ${opacity / 100}`)
      expect(translated.attributes('src')).toBe(url)
      expect(wrapper.get('img[alt="结果图修复范围覆盖层"]').attributes('style')).toBeUndefined()
    }
    await wrapper.setProps({ showCleanPreview: true })
    expect(translated.attributes('style')).toContain('display: none')
    await wrapper.setProps({ showCleanPreview: false })
    expect(translated.attributes('style')).not.toContain('display: none')
  })

  it('hides translated text completely while retaining the clean background', async () => {
    const wrapper = comparison()
    await wrapper.setProps({ translatedTextOpacity: 0 })
    expect(wrapper.find('.edit-image-comparison__source-overlay').exists()).toBe(false)
    await wrapper.get(translatedLayer).trigger('load')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    expect(wrapper.get(translatedLayer).attributes('style')).toContain('opacity: 0')
    await wrapper.setProps({ currentImage: createImage({ documentRevision: 2, renderedRevision: 1 }) })
    expect(wrapper.find(translatedLayer).exists()).toBe(false)
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    await wrapper.setProps({ currentImage: createImage({ documentRevision: 2, renderedRevision: 2 }) })
    expect(wrapper.get(translatedLayer).attributes('src')).toContain('revision=2')
  })

  it('keeps the published result on preview failure and offers a direct retry', async () => {
    const wrapper = comparison()
    await wrapper.setProps({ originalTextOpacity: 50, translatedTextOpacity: 25 })
    await wrapper.get('.edit-image-comparison__source-overlay').trigger('load')
    const failed = wrapper.get(translatedLayer)
    await failed.trigger('error')
    expect(wrapper.get(resultImage).attributes('src')).toBe('/translated.png')
    expect(wrapper.get('.edit-image-comparison__source-overlay').attributes('style')).toContain('display: none')
    await wrapper.get('button[aria-label="重新加载文字预览"]').trigger('click')
    expect(wrapper.get(translatedLayer).element).not.toBe(failed.element)
    await loadTextLayers(wrapper)
    expect(wrapper.get(resultImage).attributes('src')).toBe('/clean.png')
    expect(wrapper.get(translatedLayer).attributes('style')).toContain('opacity: 0.25')
    expect(wrapper.find('button[aria-label="重新加载文字预览"]').exists()).toBe(false)
  })

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
