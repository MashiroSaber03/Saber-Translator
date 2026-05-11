import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import ReferenceImageSelector from './ReferenceImageSelector.vue'

vi.mock('@/api/insight', () => ({
  getThumbnailUrl: vi.fn().mockReturnValue('/thumb/page-1.png'),
}))

describe('ReferenceImageSelector', () => {
  it('auto-selects and emits reference tokens instead of raw paths', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      props: {
        visible: true,
        mode: 'script',
        maxCount: 1,
        originalImages: [
          {
            page_number: 1,
            path: '/tmp/page-1.png',
            has_image: true,
            token: 'original:1',
          },
        ],
        continuationImages: [],
        characterForms: [],
        initialSelection: [],
        bookId: 'book-1',
      },
    })

    await wrapper.find('button.btn.primary').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([['original:1']])
  })

  it('prefers the latest real images instead of placeholder continuation pages', async () => {
    const wrapper = mount(ReferenceImageSelector, {
      props: {
        visible: true,
        mode: 'image',
        maxCount: 2,
        originalImages: [
          {
            page_number: 185,
            path: '/tmp/page-185.png',
            has_image: true,
            token: 'original:185',
          },
          {
            page_number: 186,
            path: '/tmp/page-186.png',
            has_image: true,
            token: 'original:186',
          },
        ],
        continuationImages: [
          {
            page_number: 187,
            path: '',
            has_image: false,
            token: 'continuation:1',
            is_placeholder: true,
          },
          {
            page_number: 188,
            path: '',
            has_image: false,
            token: 'continuation:2',
            is_placeholder: true,
          },
        ],
        characterForms: [],
        initialSelection: [],
        bookId: 'book-1',
      },
    })

    await wrapper.find('button.btn.primary').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([['original:185', 'original:186']])
  })
})
