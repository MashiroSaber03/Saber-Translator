import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

const pageImage = {
  page_num: 1,
  original: 'data:image/png;base64,original',
  translated: 'data:image/png;base64,translated',
}

function readScopedStyle(filePath: string): string {
  const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
  return source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
}

describe('ReaderCanvas', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('clears delayed page recalculation when unmounted', async () => {
    vi.useFakeTimers()
    const pageChangeSpy = vi.fn()
    const querySelectorAllSpy = vi.spyOn(document, 'querySelectorAll')
    const wrapper = mount(ReaderCanvas, {
      props: {
        images: [],
        viewMode: 'translated',
        isLoading: false,
        onPageChange: pageChangeSpy,
      },
    })

    try {
      await wrapper.setProps({ images: [pageImage] })
      expect(vi.getTimerCount()).toBe(1)
      wrapper.unmount()

      vi.advanceTimersByTime(100)

      expect(querySelectorAllSpy).not.toHaveBeenCalled()
      expect(pageChangeSpy).not.toHaveBeenCalled()
    } finally {
      querySelectorAllSpy.mockRestore()
    }
  })

  it('renders loading feedback through the shared spinner primitive', () => {
    const wrapper = mount(ReaderCanvas, {
      props: {
        images: [],
        viewMode: 'translated',
        isLoading: true,
      },
    })

    const spinner = wrapper.getComponent(UiSpinner)
    expect(spinner.props('label')).toBe('正在加载阅读内容')
  })

  it('renders the empty reader state through the inverse product empty-state pattern', () => {
    const wrapper = mount(ReaderCanvas, {
      props: {
        images: [],
        viewMode: 'translated',
        isLoading: false,
      },
    })

    const emptyState = wrapper.getComponent(ProductEmptyState)
    expect(emptyState.props()).toMatchObject({
      iconName: 'book-open',
      title: '暂无图片',
      description: '该章节还没有图片，点击下方按钮开始翻译',
      variant: 'inverse',
    })
    const translateButton = wrapper.getComponent(UiButton)
    expect(translateButton.props('variant')).toBe('primary')
    expect(translateButton.text()).toContain('进入翻译')
    expect(wrapper.find('.reader-empty-state').exists()).toBe(false)
    expect(wrapper.find('.empty-icon').exists()).toBe(false)
  })

  it('does not keep legacy DOM id hooks for reader canvas states', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/reader/ReaderCanvas.vue'), 'utf8')

    for (const legacyId of [
      'id="loadingState"',
      'id="emptyState"',
      'id="goTranslateBtn"',
      'id="imagesContainer"',
    ]) {
      expect(source).not.toContain(legacyId)
    }
  })

  it('maps canvas style owner colors through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/reader/ReaderCanvas.vue'), 'utf8')
    const style = readScopedStyle('src/components/reader/ReaderCanvas.vue')

    expect(source).not.toContain('document.querySelectorAll')
    expect(style).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(style).toContain('--reader-canvas-page-background: var(--color-surface-inverse)')
  })

  it('keeps the reader canvas shell styles in a single owner block', () => {
    const style = readScopedStyle('src/components/reader/ReaderCanvas.vue')
    const readerMainBlocks = style.match(/^\.reader-canvas\s*\{/gm) ?? []

    expect(readerMainBlocks).toHaveLength(1)
  })

  it('does not keep stale image-loading CSS after the spinner migration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/reader/ReaderCanvas.vue'), 'utf8')
    const style = readScopedStyle('src/components/reader/ReaderCanvas.vue')

    expect(source).not.toContain('class="reader-image loading"')
    expect(style).not.toContain('.reader-image.loading')
    expect(style).not.toContain('--reader-canvas-image-loading-background')
  })

  it('keeps reader canvas state hooks under the reader-canvas owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/reader/ReaderCanvas.vue'), 'utf8')

    for (const currentHook of [
      'reader-canvas',
      'reader-canvas__loading-state',
      'reader-canvas__loading-text',
      'reader-canvas__empty-state',
      'reader-canvas__images',
      'reader-canvas__image-wrapper',
      'reader-canvas__image',
      'reader-canvas__image-index',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'reader-main',
      'loading-state',
      'loading-text',
      'reader-canvas-empty-state',
      'images-container',
      'reader-image-wrapper',
      'reader-image',
      'image-index',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }
  })
})
