import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick } from 'vue'

import { useWebImportModal } from '@/components/translate/useWebImportModal'
import { useImageStore } from '@/stores/imageStore'
import { useWebImportStore } from '@/stores/webImportStore'

const { checkGalleryDLSupportMock, fetchModelsMock, getGalleryDLImagesMock, showToastMock } = vi.hoisted(() => ({
  checkGalleryDLSupportMock: vi.fn(),
  fetchModelsMock: vi.fn(),
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
    fetchModels: fetchModelsMock,
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
    fetchModelsMock.mockReset()
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
    const webImportStore = useWebImportStore()

    webImportStore.modalVisible = true
    await nextTick()
    exposed.wrapper.unmount()
    await vi.advanceTimersByTimeAsync(150)

    expect(querySelectorSpy).not.toHaveBeenCalled()
  })

  it('requests source URL focus when the modal opens without exposing a child instance ref', async () => {
    const querySelectorSpy = vi.spyOn(document, 'querySelector')
    const exposed = mountComposableHost()
    const webImportStore = useWebImportStore()

    expect(exposed.api).not.toHaveProperty('extractBarRef')
    expect(exposed.api.focusSourceUrlRequestId.value).toBe(0)

    webImportStore.modalVisible = true
    await nextTick()
    await vi.advanceTimersByTimeAsync(150)

    expect(querySelectorSpy).not.toHaveBeenCalled()
    expect(exposed.api.focusSourceUrlRequestId.value).toBe(1)
  })

  it('ignores stale URL support responses after the source URL changes', async () => {
    const firstSupport = deferred<{ available: boolean; supported: boolean }>()
    const secondSupport = deferred<{ available: boolean; supported: boolean }>()
    checkGalleryDLSupportMock
      .mockReturnValueOnce(firstSupport.promise)
      .mockReturnValueOnce(secondSupport.promise)
    const exposed = mountComposableHost()

    exposed.api.urlInput.value = 'https://example.com/old'
    await nextTick()
    await vi.advanceTimersByTimeAsync(500)

    exposed.api.urlInput.value = 'https://example.com/current'
    await nextTick()
    await vi.advanceTimersByTimeAsync(500)

    secondSupport.resolve({ available: true, supported: true })
    await flushPromises()
    expect(exposed.api.galleryDLAvailable.value).toBe(true)
    expect(exposed.api.galleryDLSupported.value).toBe(true)

    firstSupport.resolve({ available: false, supported: false })
    await flushPromises()

    expect(exposed.api.galleryDLAvailable.value).toBe(true)
    expect(exposed.api.galleryDLSupported.value).toBe(true)
  })

  it('ignores model fetch responses after the modal owner unmounts', async () => {
    const modelFetch = deferred<{ success: boolean; models: Array<{ id: string }> }>()
    fetchModelsMock.mockReturnValueOnce(modelFetch.promise)
    const exposed = mountComposableHost()
    const webImportStore = useWebImportStore()

    webImportStore.beginSettingsEdit()
    webImportStore.setAgentApiKey('sk-test')

    const fetchPromise = exposed.api.handleFetchModels()
    expect(exposed.api.isFetchingModels.value).toBe(true)

    exposed.wrapper.unmount()
    modelFetch.resolve({ success: true, models: [{ id: 'gpt-4o-mini' }] })
    await fetchPromise
    await flushPromises()

    expect(exposed.api.modelList.value).toEqual([])
    expect(exposed.api.isFetchingModels.value).toBe(false)
    expect(showToastMock).not.toHaveBeenCalledWith('获取到 1 个模型', 'success')
  })

  it('imports the selected gallery-dl page numbers instead of the first downloaded images', async () => {
    const exposed = mountComposableHost()
    const imageStore = useImageStore()
    const webImportStore = useWebImportStore()
    webImportStore.setExtractResult({
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
    webImportStore.togglePageSelection(1)
    webImportStore.togglePageSelection(3)
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

  it('keeps store and internal workflow helpers private to the modal owner', () => {
    const exposed = mountComposableHost()

    expect(exposed.api).not.toHaveProperty('webImportStore')
    expect(exposed.api).not.toHaveProperty('ensureSettingsReady')
    expect(exposed.api).not.toHaveProperty('currentEngine')
    expect(exposed.api).not.toHaveProperty('providerRequiresBaseUrl')
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

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}
