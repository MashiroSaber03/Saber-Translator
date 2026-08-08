import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

const pageImage = {
  id: 'page-1',
  chapterId: 'chapter-1',
  ordinal: 1,
  logicalSourcePath: 'page.png',
  sourceRevision: 1,
  documentRevision: 1,
  sourceUrl: '/api/v2/assets/source',
  thumbnailSourceUrl: '/api/v2/assets/thumb',
  translatedUrl: '/api/v2/assets/translated',
  width: 800,
  height: 1200,
}

const VirtualPageStreamStub = defineComponent({
  name: 'VirtualPageStream',
  props: {
    items: { type: Array, default: () => [] },
    overscanScreens: { type: Number, default: 0 },
  },
  emits: ['visibleChange'],
  template: '<div class="virtual-page-stream-stub" />',
})

function readScopedStyle(filePath: string): string {
  const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
  return source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
}

describe('ReaderCanvas', () => {
  it('passes only immutable current-page URLs into the virtual stream', async () => {
    const wrapper = mount(ReaderCanvas, {
      props: {
        images: [pageImage],
        viewMode: 'translated',
        isLoading: false,
      },
      global: {
        stubs: { VirtualPageStream: VirtualPageStreamStub },
      },
    })

    const stream = wrapper.getComponent(VirtualPageStreamStub)
    expect(stream.props('overscanScreens')).toBe(2)
    expect(stream.props('items')).toEqual([expect.objectContaining({
      id: 'page-1',
      url: '/api/v2/assets/translated',
      width: 800,
      height: 1200,
    })])

    await wrapper.setProps({ viewMode: 'original' })
    expect(wrapper.getComponent(VirtualPageStreamStub).props('items')).toEqual([
      expect.objectContaining({ url: '/api/v2/assets/source' }),
    ])
  })

  it('marks source fallbacks as untranslated only in translated mode', async () => {
    const untranslatedPage = {
      ...pageImage,
      id: 'page-2',
      translatedUrl: null,
    }
    const wrapper = mount(ReaderCanvas, {
      props: {
        images: [untranslatedPage],
        viewMode: 'translated',
        isLoading: false,
      },
      global: {
        stubs: { VirtualPageStream: VirtualPageStreamStub },
      },
    })

    expect(wrapper.getComponent(VirtualPageStreamStub).props('items')).toEqual([
      expect.objectContaining({
        badge: '未翻译',
        url: '/api/v2/assets/source',
      }),
    ])

    await wrapper.setProps({ viewMode: 'original' })
    expect(wrapper.getComponent(VirtualPageStreamStub).props('items')).toEqual([
      expect.objectContaining({ badge: undefined }),
    ])
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
      title: '暂无图片',
      description: '该章节还没有图片，点击下方按钮开始翻译',
      variant: 'inverse',
    })
    expect(emptyState.get('.product-empty-state__icon-text').text()).toBe('📖')
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
      'reader-canvas__stream',
    ]) {
      expect(source).toContain(currentHook)
    }

    expect(source).toContain('VirtualPageStream')
    expect(source).not.toContain('<img')

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
