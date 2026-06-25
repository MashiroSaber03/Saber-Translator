import { enableAutoUnmount, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick } from 'vue'

import { useWebImportModal } from '@/components/translate/useWebImportModal'
import { useImageStore } from '@/stores/imageStore'

const { checkGalleryDLSupportMock, getGalleryDLImagesMock, showToastMock } = vi.hoisted(() => ({
  checkGalleryDLSupportMock: vi.fn(),
  getGalleryDLImagesMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/webImport', () => ({
  checkGalleryDLSupport: checkGalleryDLSupportMock,
  downloadImages: vi.fn(),
  extractImages: vi.fn(),
  getGalleryDLImages: getGalleryDLImagesMock,
  testAgentConnection: vi.fn(),
  testFirecrawlConnection: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    fetchModels: vi.fn(),
  },
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

describe('useWebImportModal', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
    localStorage.clear()
    checkGalleryDLSupportMock.mockReset()
    checkGalleryDLSupportMock.mockResolvedValue({ available: true, supported: true })
    getGalleryDLImagesMock.mockReset()
    showToastMock.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('clears pending URL support checks when the owner unmounts', async () => {
    const exposed = mountComposableHost()

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await nextTick()
    exposed.wrapper.unmount()
    await vi.advanceTimersByTimeAsync(600)

    expect(checkGalleryDLSupportMock).not.toHaveBeenCalled()
  })

  it('clears pending input focus work when the owner unmounts', async () => {
    const querySelectorSpy = vi.spyOn(document, 'querySelector')
    const exposed = mountComposableHost()

    exposed.api.webImportStore.modalVisible = true
    await nextTick()
    exposed.wrapper.unmount()
    await vi.advanceTimersByTimeAsync(150)

    expect(querySelectorSpy).not.toHaveBeenCalled()
  })

  it('imports the selected gallery-dl page numbers instead of the first downloaded images', async () => {
    const exposed = mountComposableHost()
    const imageStore = useImageStore()
    exposed.api.webImportStore.setExtractResult({
      success: true,
      comicTitle: 'Chapter',
      chapterTitle: 'Episode',
      sourceUrl: 'https://example.com/chapter',
      totalPages: 4,
      engine: 'gallery-dl',
      pages: [
        { pageNumber: 1, imageUrl: '/api/web-import/static/temp/gallery_dl/001.webp' },
        { pageNumber: 2, imageUrl: '/api/web-import/static/temp/gallery_dl/002.webp' },
        { pageNumber: 3, imageUrl: '/api/web-import/static/temp/gallery_dl/003.webp' },
        { pageNumber: 4, imageUrl: '/api/web-import/static/temp/gallery_dl/004.webp' },
      ],
    })
    exposed.api.webImportStore.togglePageSelection(1)
    exposed.api.webImportStore.togglePageSelection(3)
    getGalleryDLImagesMock.mockResolvedValue({
      success: true,
      total: 4,
      images: [
        { filename: '001.webp', data: 'data:image/webp;base64,page1' },
        { filename: '002.webp', data: 'data:image/webp;base64,page2' },
        { filename: '003.webp', data: 'data:image/webp;base64,page3' },
        { filename: '004.webp', data: 'data:image/webp;base64,page4' },
      ],
    })

    await exposed.api.handleImport()

    expect(imageStore.images.map((image) => image.fileName)).toEqual(['002.webp', '004.webp'])
    expect(imageStore.images.map((image) => image.originalDataURL)).toEqual([
      'data:image/webp;base64,page2',
      'data:image/webp;base64,page4',
    ])
    expect(showToastMock).toHaveBeenCalledWith('成功导入 2 张图片', 'success')
  })
})

function mountComposableHost() {
  let api: ReturnType<typeof useWebImportModal> | null = null
  const Host = defineComponent({
    setup() {
      api = useWebImportModal()
      return () => null
    },
  })
  const wrapper = mount(Host)
  if (!api) {
    throw new Error('useWebImportModal host did not initialize')
  }
  return { wrapper, api }
}
