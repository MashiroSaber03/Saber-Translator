import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'
import VirtualThumbnailList from '@/components/virtual/VirtualThumbnailList.vue'
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

  it('derives one-column row height from the measured thumbnail width', async () => {
    const wrapper = mount(VirtualThumbnailList, {
      props: {
        items: items(1000),
      },
    })
    const container = wrapper.get('.virtual-thumbnail-list').element as HTMLElement
    Object.defineProperties(container, {
      clientHeight: { configurable: true, value: 320 },
      clientWidth: { configurable: true, value: 180 },
    })
    container.dispatchEvent(new Event('scroll'))
    await wrapper.vm.$nextTick()

    expect(wrapper.get('.virtual-thumbnail-list__inner').attributes('style')).toContain('height: 245994px;')
    expect(wrapper.findAll('[data-product-thumbnail-id]').length).toBeLessThanOrEqual(12)

    container.scrollTop = 2460
    container.dispatchEvent(new Event('scroll'))
    await wrapper.vm.$nextTick()

    expect(wrapper.find('[data-product-thumbnail-id="1"]').exists()).toBe(false)
    expect(wrapper.find('[data-product-thumbnail-id="11"]').exists()).toBe(true)
  })

  it('scrolls the active list item with the same measured row size', async () => {
    const wrapper = mount(VirtualThumbnailList, {
      props: {
        activeId: 1,
        items: items(30),
      },
    })
    const container = wrapper.get('.virtual-thumbnail-list').element as HTMLElement
    Object.defineProperties(container, {
      clientHeight: { configurable: true, value: 320 },
      clientWidth: { configurable: true, value: 180 },
    })
    container.dispatchEvent(new Event('scroll'))
    await wrapper.setProps({ activeId: 20 })
    await wrapper.vm.$nextTick()

    expect(container.scrollTop).toBe(4600)
  })
})
