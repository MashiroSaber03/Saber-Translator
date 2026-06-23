import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReaderControls from '@/components/reader/ReaderControls.vue'

function mountControls() {
  return mount(ReaderControls, {
    props: {
      currentPage: 1,
      totalPages: 3,
      hasPrevChapter: true,
      hasNextChapter: true,
      showChapterNav: true,
    },
  })
}

describe('ReaderControls', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
    document.documentElement.style.removeProperty('--reader-page-background')
    document.documentElement.style.removeProperty('--reader-image-width')
    document.documentElement.style.removeProperty('--reader-gap')
  })

  it('names icon-only controls and color swatches', async () => {
    const wrapper = mountControls()

    expect(wrapper.get('#scrollTopBtn').attributes('aria-label')).toBe('回到顶部')

    wrapper.vm.openSettings()
    await nextTick()

    expect(wrapper.get('.reader-controls__close-button').attributes('aria-label')).toBe('关闭阅读设置')
    const swatchLabels = wrapper.findAll('.reader-controls__bg-option')
      .map(button => button.attributes('aria-label'))
    expect(swatchLabels).toEqual([
      '设置背景颜色为深蓝',
      '设置背景颜色为白色',
      '设置背景颜色为米色',
      '设置背景颜色为深灰',
    ])
  })

  it('ignores incomplete or invalid stored settings payloads', () => {
    localStorage.setItem('readerSettings', JSON.stringify({
      imageWidth: 10,
      imageGap: 200,
      bgColor: 'not-a-reader-preset',
    }))

    mountControls()

    expect(document.documentElement.style.getPropertyValue('--reader-image-width')).toBe('100%')
    expect(document.documentElement.style.getPropertyValue('--reader-gap')).toBe('8px')
    expect(document.documentElement.style.getPropertyValue('--reader-page-background')).toBe('#1a1a2e')
  })

  it('loads complete current-schema stored settings', () => {
    localStorage.setItem('readerSettings', JSON.stringify({
      readerSettingsSchemaVersion: 1,
      imageWidth: 80,
      imageGap: 12,
      bgColor: '#ffffff',
    }))

    mountControls()

    expect(document.documentElement.style.getPropertyValue('--reader-image-width')).toBe('80%')
    expect(document.documentElement.style.getPropertyValue('--reader-gap')).toBe('12px')
    expect(document.documentElement.style.getPropertyValue('--reader-page-background')).toBe('#ffffff')
  })
})
