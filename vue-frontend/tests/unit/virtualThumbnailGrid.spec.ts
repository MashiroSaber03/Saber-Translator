import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'

function items(count: number): ProductThumbnailGridItem[] {
  return Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    src: `/thumbs/${index + 1}.webp`,
    alt: `第 ${index + 1} 页`,
    label: String(index + 1),
  }))
}

describe('VirtualThumbnailGrid', () => {
  it('keeps a 1000-page collection to a bounded thumbnail DOM window', async () => {
    const wrapper = mount(VirtualThumbnailGrid, {
      props: {
        columns: 4,
        items: items(1000),
        maxHeight: 320,
      },
    })
    const container = wrapper.get('.virtual-thumbnail-grid').element as HTMLElement
    Object.defineProperties(container, {
      clientHeight: { configurable: true, value: 320 },
      clientWidth: { configurable: true, value: 320 },
    })
    container.dispatchEvent(new Event('scroll'))
    await wrapper.vm.$nextTick()

    expect(wrapper.findAll('[data-product-thumbnail-id]').length).toBeLessThanOrEqual(32)
    expect(wrapper.find('[data-product-thumbnail-id="1"]').exists()).toBe(true)

    container.scrollTop = 26000
    container.dispatchEvent(new Event('scroll'))
    await wrapper.vm.$nextTick()

    expect(wrapper.findAll('[data-product-thumbnail-id]').length).toBeLessThanOrEqual(32)
    expect(wrapper.find('[data-product-thumbnail-id="1"]').exists()).toBe(false)
    expect(wrapper.findAll('img').every(image => image.attributes('loading') === 'lazy')).toBe(true)
  })

  it('forwards selection from a rendered thumbnail', async () => {
    const wrapper = mount(VirtualThumbnailGrid, {
      props: { items: items(20) },
    })

    await wrapper.get('[data-product-thumbnail-id="1"]').trigger('click')

    expect(wrapper.emitted('select')).toEqual([[1]])
  })
})
