import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import EditThumbnailPanel from '@/components/edit/EditThumbnailPanel.vue'
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
  it('uses named button controls for image thumbnails', async () => {
    const wrapper = mount(EditThumbnailPanel, {
      props: {
        visible: true,
        images,
        currentImageIndex: 1,
      },
    })

    const firstThumbnail = wrapper.get('.edit-thumbnail-item')
    expect(firstThumbnail.element.tagName).toBe('BUTTON')
    expect(firstThumbnail.attributes('aria-label')).toBe('切换到图片 1')
    expect(firstThumbnail.attributes('aria-pressed')).toBe('false')

    const activeThumbnail = wrapper.findAll('.edit-thumbnail-item')[1]!
    expect(activeThumbnail.attributes('aria-pressed')).toBe('true')

    await firstThumbnail.trigger('click')
    expect(wrapper.emitted('switch-to-image')?.[0]).toEqual([0])
  })
})
