import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick } from 'vue'

import { useWebImportModal } from '@/components/translate/useWebImportModal'
import { useWebImportStore } from '@/stores/webImportStore'

const {
  checkWebImportSupportMock,
  commitWebImportDraftMock,
  createWebImportDraftMock,
  fetchModelsMock,
  getTranslationBootstrapMock,
  getWebImportDraftMock,
  listAllWebImportDraftPagesMock,
  showToastMock,
  updateWebImportSelectionMock,
} = vi.hoisted(() => ({
  checkWebImportSupportMock: vi.fn(),
  commitWebImportDraftMock: vi.fn(),
  createWebImportDraftMock: vi.fn(),
  fetchModelsMock: vi.fn(),
  getTranslationBootstrapMock: vi.fn(),
  getWebImportDraftMock: vi.fn(),
  listAllWebImportDraftPagesMock: vi.fn(),
  showToastMock: vi.fn(),
  updateWebImportSelectionMock: vi.fn(),
}))

vi.mock('@/api/webImport', () => ({
  testAgentConnection: vi.fn(),
  testFirecrawlConnection: vi.fn(),
}))

vi.mock('@/api/v2/webImport', () => ({
  checkWebImportSupport: checkWebImportSupportMock,
  commitWebImportDraft: commitWebImportDraftMock,
  createWebImportDraft: createWebImportDraftMock,
  getWebImportDraft: getWebImportDraftMock,
  listAllWebImportDraftPages: listAllWebImportDraftPagesMock,
  updateWebImportSelection: updateWebImportSelectionMock,
}))

vi.mock('@/api/v2/content', () => ({
  getTranslationBootstrap: getTranslationBootstrapMock,
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
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
    checkWebImportSupportMock.mockReset()
    checkWebImportSupportMock.mockResolvedValue({
      galleryDlAvailable: true,
      galleryDlSupported: true,
    })
    commitWebImportDraftMock.mockReset()
    createWebImportDraftMock.mockReset()
    getTranslationBootstrapMock.mockReset()
    getWebImportDraftMock.mockReset()
    listAllWebImportDraftPagesMock.mockReset()
    updateWebImportSelectionMock.mockReset()
    fetchModelsMock.mockReset()
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

    expect(checkWebImportSupportMock).not.toHaveBeenCalled()
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
    const firstSupport = deferred<{ galleryDlAvailable: boolean; galleryDlSupported: boolean }>()
    const secondSupport = deferred<{ galleryDlAvailable: boolean; galleryDlSupported: boolean }>()
    checkWebImportSupportMock
      .mockReturnValueOnce(firstSupport.promise)
      .mockReturnValueOnce(secondSupport.promise)
    const exposed = mountComposableHost()

    exposed.api.urlInput.value = 'https://example.com/old'
    await nextTick()
    await vi.advanceTimersByTimeAsync(500)

    exposed.api.urlInput.value = 'https://example.com/current'
    await nextTick()
    await vi.advanceTimersByTimeAsync(500)

    secondSupport.resolve({ galleryDlAvailable: true, galleryDlSupported: true })
    await flushPromises()
    expect(exposed.api.galleryDLAvailable.value).toBe(true)
    expect(exposed.api.galleryDLSupported.value).toBe(true)

    firstSupport.resolve({ galleryDlAvailable: false, galleryDlSupported: false })
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

  it('commits the selected backend draft page ids without loading image payloads', async () => {
    const exposed = mountComposableHost()
    const webImportStore = useWebImportStore()
    getTranslationBootstrapMock.mockResolvedValue({
      activeWebImportDraft: null,
      chapter: { id: 'chapter-1' },
    })
    createWebImportDraftMock.mockResolvedValue({
      draftId: 'draft-1',
      status: 'queued',
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
    getWebImportDraftMock.mockResolvedValue({
      id: 'draft-1',
      sourceUrl: 'https://example.com/chapter',
      status: 'ready',
      revision: 2,
      candidateCount: 4,
      failedCount: 0,
      actualEngine: 'gallery-dl',
    })
    listAllWebImportDraftPagesMock.mockResolvedValue([
      { id: 'page-1', error: null, thumbnailUrl: '/thumb/1', sourceMediaUrl: '/media/1' },
      { id: 'page-2', error: null, thumbnailUrl: '/thumb/2', sourceMediaUrl: '/media/2' },
      { id: 'page-3', error: null, thumbnailUrl: '/thumb/3', sourceMediaUrl: '/media/3' },
      { id: 'page-4', error: null, thumbnailUrl: '/thumb/4', sourceMediaUrl: '/media/4' },
    ])
    updateWebImportSelectionMock.mockResolvedValue({ revision: 3 })
    commitWebImportDraftMock.mockResolvedValue({
      status: 'queued',
      batchId: 'batch-2',
      jobIds: ['job-2'],
    })

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()
    webImportStore.togglePageSelection(1)
    webImportStore.togglePageSelection(3)

    await exposed.api.handleImport()

    expect(updateWebImportSelectionMock).toHaveBeenCalledWith('draft-1', 2, ['page-2', 'page-4'])
    expect(commitWebImportDraftMock).toHaveBeenCalledWith('draft-1', 3)
    expect(showToastMock).toHaveBeenCalledWith('入库任务已进入后端任务中心，可安全关闭页面', 'success')
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
