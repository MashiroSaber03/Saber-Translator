import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReaderCanvas from '@/components/reader/ReaderCanvas.vue'

const pageImage = {
  page_num: 1,
  original: 'data:image/png;base64,original',
  translated: 'data:image/png;base64,translated',
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
})
