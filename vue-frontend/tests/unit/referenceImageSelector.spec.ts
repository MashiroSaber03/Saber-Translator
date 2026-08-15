import { afterEach, describe, expect, it } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'

import type { MangaImageInfo } from '@/api/continuation'
import ReferenceImageSelector from '@/components/insight/continuation/ReferenceImageSelector.vue'
import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'

const wrappers: VueWrapper[] = []

function originalImages(count: number): MangaImageInfo[] {
  return Array.from({ length: count }, (_, index) => ({
    token: `original:${index + 1}`,
    page_number: index + 1,
    path: `/source/${index + 1}`,
    has_image: true,
  }))
}

afterEach(() => {
  while (wrappers.length > 0) wrappers.pop()?.unmount()
  document.body.innerHTML = ''
})

describe('ReferenceImageSelector', () => {
  it('renders a large manga collection through a bounded virtual thumbnail window', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      attachTo: document.body,
      props: {
        visible: true,
        mode: 'script',
        maxCount: 3,
        originalImages: originalImages(1000),
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
      },
    })
    wrappers.push(wrapper)
    await wrapper.vm.$nextTick()

    const grid = wrapper.getComponent(VirtualThumbnailGrid)
    expect(grid.props('items')).toHaveLength(1000)
    expect(grid.findAll('[data-product-thumbnail-id]').length).toBeLessThanOrEqual(8)
    expect(grid.findAll('img').every(image => image.attributes('loading') === 'lazy')).toBe(true)
  })
})
