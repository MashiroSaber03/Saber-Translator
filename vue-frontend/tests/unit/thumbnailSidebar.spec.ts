import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ThumbnailSidebar from '@/components/translate/ThumbnailSidebar.vue'
import { useImageStore } from '@/stores/imageStore'
import type { ImageData } from '@/types/image'

function createImage(id: string, fileName: string, folderPath = ''): ImageData {
  return {
    id,
    fileName,
    folderPath,
    relativePath: folderPath ? `${folderPath}/${fileName}` : fileName,
    width: 100,
    height: 100,
    originalDataURL: `data:image/png;base64,${id}`,
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    hasUnsavedChanges: false,
    fontSize: 18,
    fontFamily: 'Arial',
    textColor: '#000000',
    strokeColor: '#ffffff',
    strokeWidth: 0,
    strokeEnabled: false,
    fillColor: '#ffffff',
    layoutDirection: 'horizontal',
    lineSpacing: 1,
    textAlign: 'center',
    autoFontSize: false,
    useAutoTextColor: false,
  } as ImageData
}

function mountSidebar(images: ImageData[]) {
  setActivePinia(createPinia())
  const imageStore = useImageStore()
  imageStore.images = images
  imageStore.currentImageIndex = images.length > 0 ? 0 : -1

  return mount(ThumbnailSidebar, {
    props: {
      isVisible: true,
    },
  })
}

describe('ThumbnailSidebar', () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollTo = vi.fn()
  })

  it('uses button semantics for flat thumbnail selection', async () => {
    const wrapper = mountSidebar([
      createImage('page-1', '001.png'),
      createImage('page-2', '002.png'),
    ])

    const thumbnails = wrapper.findAll('.thumbnail-sidebar__item')
    expect(thumbnails).toHaveLength(2)
    expect(thumbnails[0]?.element.tagName).toBe('BUTTON')

    await thumbnails[1]?.trigger('click')

    expect(wrapper.emitted('select')?.[0]).toEqual([1])
  })

  it('uses button semantics for folder navigation and tree thumbnail selection', async () => {
    const wrapper = mountSidebar([
      createImage('page-1', '001.png', 'chapter-a'),
      createImage('page-2', '002.png', 'chapter-a'),
    ])

    const folderButton = wrapper.get('.folder-item')
    expect(folderButton.element.tagName).toBe('BUTTON')

    await folderButton.trigger('click')

    const backButton = wrapper.get('.folder-back-btn')
    expect(backButton.element.tagName).toBe('BUTTON')
    expect(wrapper.get('.breadcrumb-item').element.tagName).toBe('BUTTON')

    const thumbnails = wrapper.findAll('.thumbnail-sidebar__item')
    expect(thumbnails[0]?.element.tagName).toBe('BUTTON')

    await thumbnails[1]?.trigger('click')
    expect(wrapper.emitted('select')?.[0]).toEqual([1])
  })
})
