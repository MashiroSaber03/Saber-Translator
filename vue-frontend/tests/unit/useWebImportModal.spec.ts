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
  confirmProductActionMock,
  fetchModelsMock,
  getTranslationBootstrapMock,
  getWebImportDraftMock,
  listWebImportDraftPagesMock,
  showToastMock,
  updateWebImportSelectionMock,
} = vi.hoisted(() => ({
  checkWebImportSupportMock: vi.fn(),
  commitWebImportDraftMock: vi.fn(),
  createWebImportDraftMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
  fetchModelsMock: vi.fn(),
  getTranslationBootstrapMock: vi.fn(),
  getWebImportDraftMock: vi.fn(),
  listWebImportDraftPagesMock: vi.fn(),
  showToastMock: vi.fn(),
  updateWebImportSelectionMock: vi.fn(),
}))

vi.mock('@/api/v2/webImport', () => ({
  checkWebImportSupport: checkWebImportSupportMock,
  commitWebImportDraft: commitWebImportDraftMock,
  createWebImportDraft: createWebImportDraftMock,
  getWebImportDraft: getWebImportDraftMock,
  listWebImportDraftPages: listWebImportDraftPagesMock,
  testAgentConnection: vi.fn(),
  testFirecrawlConnection: vi.fn(),
  updateWebImportSelection: updateWebImportSelectionMock,
}))

vi.mock('@/api/v2/content', () => ({
  getTranslationBootstrap: getTranslationBootstrapMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
}))

vi.mock('@/api/v2/diagnostics', () => ({
  fetchModels: fetchModelsMock,
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
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
    getTranslationBootstrapMock.mockReset()
    getWebImportDraftMock.mockReset()
    listWebImportDraftPagesMock.mockReset()
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

  it('does not run Gallery-DL support checks for an explicit AI Agent extraction', async () => {
    const exposed = mountComposableHost()

    exposed.api.selectedEngine.value = 'ai-agent'
    exposed.api.urlInput.value = 'https://example.com/chapter'
    await nextTick()
    await vi.advanceTimersByTimeAsync(600)

    expect(checkWebImportSupportMock).not.toHaveBeenCalled()
    expect(exposed.api.checkingSupport.value).toBe(false)
  })

  it('ignores model fetch responses after the modal owner unmounts', async () => {
    const modelFetch = deferred<{ models: Array<{ id: string }> }>()
    fetchModelsMock.mockReturnValueOnce(modelFetch.promise)
    const exposed = mountComposableHost()
    const webImportStore = useWebImportStore()

    webImportStore.setAgentApiKey('sk-test')

    const fetchPromise = exposed.api.handleFetchModels()
    expect(exposed.api.isFetchingModels.value).toBe(true)

    exposed.wrapper.unmount()
    modelFetch.resolve({ models: [{ id: 'gpt-4o-mini' }] })
    await fetchPromise
    await flushPromises()

    expect(exposed.api.modelList.value).toEqual([])
    expect(exposed.api.isFetchingModels.value).toBe(false)
    expect(showToastMock).not.toHaveBeenCalledWith('获取到 1 个模型', 'success')
  })

  it('commits the selected backend draft page ids without loading image payloads', async () => {
    const onCommitAccepted = vi.fn()
    const exposed = mountComposableHost({ onCommitAccepted })
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
      autoImport: false,
      candidateCount: 4,
      selectedCount: 4,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: 'gallery-dl',
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock.mockResolvedValue({
      items: [
        {
          id: 'page-1',
          selected: true,
          error: null,
          thumbnailUrl: '/thumb/1',
          sourceMediaUrl: '/media/1',
        },
        {
          id: 'page-2',
          selected: true,
          error: null,
          thumbnailUrl: '/thumb/2',
          sourceMediaUrl: '/media/2',
        },
        {
          id: 'page-3',
          selected: true,
          error: null,
          thumbnailUrl: '/thumb/3',
          sourceMediaUrl: '/media/3',
        },
        {
          id: 'page-4',
          selected: true,
          error: null,
          thumbnailUrl: '/thumb/4',
          sourceMediaUrl: '/media/4',
        },
      ],
      nextCursor: null,
    })
    updateWebImportSelectionMock.mockResolvedValue({ revision: 3 })
    commitWebImportDraftMock.mockResolvedValue({
      status: 'queued',
      batchId: 'batch-2',
      jobIds: ['job-2'],
    })

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()
    exposed.api.togglePage(1)
    exposed.api.togglePage(3)

    await exposed.api.handleImport()

    expect(updateWebImportSelectionMock).toHaveBeenCalledWith('draft-1', 2, ['page-2', 'page-4'])
    expect(commitWebImportDraftMock).toHaveBeenCalledWith('draft-1', 3)
    expect(onCommitAccepted).toHaveBeenCalledWith({
      status: 'queued',
      batchId: 'batch-2',
      jobIds: ['job-2'],
    })
    expect(showToastMock).toHaveBeenCalledWith(
      '入库任务已进入后端任务中心，可安全关闭页面',
      'success'
    )
  })

  it('restores draft candidates one cursor page at a time', async () => {
    const exposed = mountComposableHost()
    getTranslationBootstrapMock.mockResolvedValue({
      activeWebImportDraft: null,
      chapter: { id: 'chapter-1' },
    })
    createWebImportDraftMock.mockResolvedValue({
      draftId: 'draft-paged',
      status: 'queued',
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
    getWebImportDraftMock.mockResolvedValue({
      id: 'draft-paged',
      sourceUrl: 'https://example.com/chapter',
      status: 'ready',
      revision: 2,
      autoImport: false,
      candidateCount: 3,
      selectedCount: 3,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: 'gallery-dl',
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock
      .mockResolvedValueOnce({
        items: [
          { id: 'page-1', selected: true, error: null, thumbnailUrl: '/thumb/1' },
          { id: 'page-2', selected: true, error: null, thumbnailUrl: '/thumb/2' },
        ],
        nextCursor: 2,
      })
      .mockResolvedValueOnce({
        items: [{ id: 'page-3', selected: true, error: null, thumbnailUrl: '/thumb/3' }],
        nextCursor: null,
      })

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()

    expect(exposed.api.extractResult.value?.pages).toHaveLength(2)
    expect(exposed.api.hasMorePages.value).toBe(true)
    expect(listWebImportDraftPagesMock).toHaveBeenLastCalledWith('draft-paged', {
      cursor: 0,
      limit: 100,
    })

    await exposed.api.loadMoreDraftPages()

    expect(exposed.api.extractResult.value?.pages).toHaveLength(3)
    expect(exposed.api.hasMorePages.value).toBe(false)
    expect(listWebImportDraftPagesMock).toHaveBeenLastCalledWith('draft-paged', {
      cursor: 2,
      limit: 100,
    })
  })

  it('reports a failed next candidate page without creating an unhandled rejection', async () => {
    const exposed = mountComposableHost()
    getTranslationBootstrapMock.mockResolvedValue({
      activeWebImportDraft: null,
      chapter: { id: 'chapter-1' },
    })
    createWebImportDraftMock.mockResolvedValue({
      draftId: 'draft-paged',
      status: 'queued',
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
    getWebImportDraftMock.mockResolvedValue({
      id: 'draft-paged',
      sourceUrl: 'https://example.com/chapter',
      status: 'ready',
      revision: 2,
      autoImport: false,
      candidateCount: 2,
      selectedCount: 2,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: 'gallery-dl',
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock
      .mockResolvedValueOnce({
        items: [{ id: 'page-1', selected: true, error: null, thumbnailUrl: '/thumb/1' }],
        nextCursor: 1,
      })
      .mockRejectedValueOnce(new Error('候选页读取失败'))

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()

    await expect(exposed.api.loadMoreDraftPages()).resolves.toBeUndefined()
    expect(showToastMock).toHaveBeenCalledWith('候选页读取失败', 'error')
    expect(exposed.api.status.value).toBe('extracted')
  })

  it('polls an accepted extraction again when its first status read fails', async () => {
    const exposed = mountComposableHost()
    const webImportStore = useWebImportStore()
    getTranslationBootstrapMock.mockResolvedValue({
      activeWebImportDraft: null,
      chapter: { id: 'chapter-1' },
    })
    createWebImportDraftMock.mockResolvedValue({
      draftId: 'draft-retry',
      status: 'queued',
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
    getWebImportDraftMock.mockRejectedValueOnce(new Error('瞬时网络错误')).mockResolvedValueOnce({
      id: 'draft-retry',
      sourceUrl: 'https://example.com/chapter',
      status: 'ready',
      revision: 2,
      autoImport: false,
      candidateCount: 1,
      selectedCount: 1,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: 'gallery-dl',
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock.mockResolvedValue({
      items: [{ id: 'page-1', selected: true, error: null, thumbnailUrl: '/thumb/1' }],
      nextCursor: null,
    })
    webImportStore.modalVisible = true
    await flushPromises()

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()

    expect(exposed.api.status.value).toBe('extracting')
    expect(exposed.api.error.value).toBeNull()

    await vi.advanceTimersByTimeAsync(1000)
    await flushPromises()

    expect(exposed.api.status.value).toBe('extracted')
    expect(exposed.api.extractResult.value?.pages).toHaveLength(1)
  })

  it('rejects a ready draft without a resolved engine instead of guessing AI Agent', async () => {
    const exposed = mountComposableHost()
    getTranslationBootstrapMock.mockResolvedValue({
      activeWebImportDraft: null,
      chapter: { id: 'chapter-1' },
    })
    createWebImportDraftMock.mockResolvedValue({
      draftId: 'draft-invalid',
      status: 'queued',
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
    getWebImportDraftMock.mockResolvedValue({
      id: 'draft-invalid',
      sourceUrl: 'https://example.com/chapter',
      status: 'ready',
      revision: 2,
      autoImport: false,
      candidateCount: 1,
      selectedCount: 1,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: null,
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock.mockResolvedValue({
      items: [{ id: 'page-1', selected: true, error: null, thumbnailUrl: '/thumb/1' }],
      nextCursor: null,
    })

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()

    expect(exposed.api.status.value).toBe('error')
    expect(exposed.api.error.value).toBe('网页导入草稿缺少有效的实际提取引擎')
    expect(exposed.api.extractResult.value).toBeNull()
  })

  it('does not let a commit from a closed session close a later modal session', async () => {
    const acceptedCommit = deferred<{
      status: 'queued'
      batchId: string
      jobIds: string[]
    }>()
    const onCommitAccepted = vi.fn()
    const exposed = mountComposableHost({ onCommitAccepted })
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
      autoImport: false,
      candidateCount: 1,
      selectedCount: 1,
      failedCount: 0,
      requestedEngine: 'auto',
      actualEngine: 'gallery-dl',
      jobs: [{ id: 'job-1', kind: 'web_extract', status: 'completed' }],
    })
    listWebImportDraftPagesMock.mockResolvedValue({
      items: [{ id: 'page-1', selected: true, error: null, thumbnailUrl: '/thumb/1' }],
      nextCursor: null,
    })
    commitWebImportDraftMock.mockReturnValue(acceptedCommit.promise)

    exposed.api.urlInput.value = 'https://example.com/chapter'
    await exposed.api.handleExtract()
    const importPromise = exposed.api.handleImport()
    await flushPromises()
    expect(exposed.api.status.value).toBe('downloading')

    await exposed.api.handleClose()
    webImportStore.modalVisible = true
    webImportStore.setStatus('idle')
    await nextTick()

    acceptedCommit.resolve({ status: 'queued', batchId: 'batch-2', jobIds: ['job-2'] })
    await importPromise
    await flushPromises()

    expect(onCommitAccepted).toHaveBeenCalledWith({
      status: 'queued',
      batchId: 'batch-2',
      jobIds: ['job-2'],
    })
    expect(webImportStore.modalVisible).toBe(true)
    expect(webImportStore.status).toBe('idle')
  })

  it('keeps store and internal workflow helpers private to the modal owner', () => {
    const exposed = mountComposableHost()

    expect(exposed.api).not.toHaveProperty('webImportStore')
    expect(exposed.api).not.toHaveProperty('ensureSettingsReady')
    expect(exposed.api).not.toHaveProperty('currentEngine')
    expect(exposed.api).not.toHaveProperty('providerRequiresBaseUrl')
  })
})

function mountComposableHost(callbacks: Parameters<typeof useWebImportModal>[0] = {}) {
  let api: ReturnType<typeof useWebImportModal> | null = null
  const Host = defineComponent({
    setup() {
      api = useWebImportModal(callbacks)
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
