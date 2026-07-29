import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import EditThumbnailPanel from '@/components/edit/EditThumbnailPanel.vue'
import type { ImageData } from '@/types/image'

const images: ImageData[] = [
  {
    id: 'page-1',
    name: 'page-1.png',
    sourceAssetUrl: '/api/v2/assets/page1',
    translatedAssetUrl: null,
    thumbnailSourceUrl: '/api/v2/assets/page-1-thumb',
  },
  {
    id: 'page-2',
    name: 'page-2.png',
    sourceAssetUrl: '/api/v2/assets/page2',
    translatedAssetUrl: '/api/v2/assets/page2-translated',
    thumbnailSourceUrl: '/api/v2/assets/page-2-source-thumb',
    thumbnailTranslatedUrl: '/api/v2/assets/page-2-translated-thumb',
  },
]

describe('EditThumbnailPanel', () => {
  it('renders only thumbnail assets with lazy asynchronous decoding', async () => {
    const wrapper = mount(EditThumbnailPanel, {
      props: {
        visible: true,
        images,
        currentImageIndex: 1,
      },
    })

    const thumbnails = wrapper.findAll('.edit-thumbnails-panel__item')
    const firstThumbnail = thumbnails[0]
    expect(firstThumbnail.element.tagName).toBe('BUTTON')
    expect(firstThumbnail.attributes('aria-label')).toBe('切换到图片 1')

    const activeThumbnail = thumbnails[1]
    expect(activeThumbnail.attributes('aria-current')).toBe('page')
    expect(activeThumbnail.classes()).toContain('edit-thumbnails-panel__item--selected')
    const renderedImages = wrapper.findAll('img')
    expect(renderedImages.map(image => image.attributes('src'))).toEqual([
      '/api/v2/assets/page-1-thumb',
      '/api/v2/assets/page-2-translated-thumb',
    ])
    for (const image of renderedImages) {
      expect(image.attributes('loading')).toBe('lazy')
      expect(image.attributes('decoding')).toBe('async')
      expect(image.attributes('src')).not.toContain('data:image')
    }

    await firstThumbnail.trigger('click')
    expect(wrapper.emitted('switch-to-image')?.[0]).toEqual([0])
  })

  it('keeps the DOM window bounded for a thousand-page chapter', () => {
    const manyImages = Array.from({ length: 1_000 }, (_, index): ImageData => ({
      id: `page-${index}`,
      name: `page-${index}.png`,
      sourceAssetUrl: `/api/v2/assets/source-${index}`,
      translatedAssetUrl: null,
      thumbnailSourceUrl: `/api/v2/assets/thumb-${index}`,
    }))
    const wrapper = mount(EditThumbnailPanel, {
      props: {
        visible: true,
        images: manyImages,
        currentImageIndex: 0,
      },
    })

    expect(wrapper.findAll('.edit-thumbnails-panel__item').length).toBeLessThanOrEqual(32)
    expect(wrapper.get('.edit-thumbnails-panel__track').attributes('style')).toContain('70000px')
  })
})
