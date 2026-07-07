import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import EditThumbnailPanel from '@/components/edit/EditThumbnailPanel.vue'
import ProductHorizontalScrollStrip from '@/components/product/ProductHorizontalScrollStrip.vue'
import type { ImageData } from '@/types/image'

const images: ImageData[] = [
  {
    id: 'page-1',
    name: 'page-1.png',
    originalDataURL: 'data:image/png;base64,page1',
    translatedDataURL: '',
  },
  {
    id: 'page-2',
    name: 'page-2.png',
    originalDataURL: 'data:image/png;base64,page2',
    translatedDataURL: 'data:image/png;base64,page2-translated',
  },
]

describe('EditThumbnailPanel', () => {
  it('renders edit thumbnails through the shared product thumbnail grid', async () => {
    const wrapper = mount(EditThumbnailPanel, {
      props: {
        visible: true,
        images,
        currentImageIndex: 1,
      },
    })

    const firstThumbnail = wrapper.get('[data-product-thumbnail-id="0"]')
    expect(firstThumbnail.element.tagName).toBe('BUTTON')
    expect(firstThumbnail.attributes('aria-label')).toBe('切换到图片 1')
    expect(firstThumbnail.attributes('aria-pressed')).toBe('false')

    const activeThumbnail = wrapper.get('[data-product-thumbnail-id="1"]')
    expect(activeThumbnail.attributes('aria-pressed')).toBe('true')
    expect(activeThumbnail.classes()).toContain('product-thumbnail-grid__item--selected')

    await firstThumbnail.trigger('click')
    expect(wrapper.emitted('switch-to-image')?.[0]).toEqual([0])
  })

  it('keeps only the edit strip shell local while delegating thumbnail items to the product primitive', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditThumbnailPanel.vue'),
      'utf8',
    )

    expect(source).toContain("import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'")
    expect(source).toContain('edit-thumbnails-panel__grid')
    expect(source).not.toContain("import UiButton from '@/components/ui/UiButton.vue'")
    expect(source).not.toContain('edit-thumbnail-item')
    expect(source).not.toContain('thumb-index')
    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).toContain('--edit-thumbnail-panel-background: color-mix')
  })

  it('delegates horizontal scrolling to the product scroll-strip primitive', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditThumbnailPanel.vue'),
      'utf8',
    )

    expect(source).toContain("import ProductHorizontalScrollStrip from '@/components/product/ProductHorizontalScrollStrip.vue'")
    expect(source).not.toContain('class="thumbnails-scroll"')
    expect(source).not.toContain('::-webkit-scrollbar')

    const wrapper = mount(EditThumbnailPanel, {
      props: {
        visible: true,
        images,
        currentImageIndex: 0,
      },
    })

    const strip = wrapper.getComponent(ProductHorizontalScrollStrip)
    expect(strip.props('ariaLabel')).toBe('编辑模式缩略图滚动条')
  })
})
