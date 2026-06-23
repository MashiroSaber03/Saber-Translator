import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, nextTick } from 'vue'

import { useWebImportModal } from '@/components/translate/useWebImportModal'

const { checkGalleryDLSupportMock } = vi.hoisted(() => ({
  checkGalleryDLSupportMock: vi.fn(),
}))

vi.mock('@/api/webImport', () => ({
  checkGalleryDLSupport: checkGalleryDLSupportMock,
  downloadImages: vi.fn(),
  extractImages: vi.fn(),
  getGalleryDLImages: vi.fn(),
  testAgentConnection: vi.fn(),
  testFirecrawlConnection: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    fetchModels: vi.fn(),
  },
}))

describe('useWebImportModal', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
    localStorage.clear()
    checkGalleryDLSupportMock.mockReset()
    checkGalleryDLSupportMock.mockResolvedValue({ available: true, supported: true })
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
