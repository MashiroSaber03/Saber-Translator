import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

const {
  extractImagesMock,
  downloadImagesMock,
  checkGalleryDLSupportMock,
  getGalleryDLImagesMock,
  testFirecrawlConnectionMock,
  testAgentConnectionMock,
} = vi.hoisted(() => ({
  extractImagesMock: vi.fn(),
  downloadImagesMock: vi.fn(),
  checkGalleryDLSupportMock: vi.fn(),
  getGalleryDLImagesMock: vi.fn(),
  testFirecrawlConnectionMock: vi.fn(),
  testAgentConnectionMock: vi.fn(),
}))

vi.mock('@/api/webImport', async () => {
  const actual = await vi.importActual<typeof import('@/api/webImport')>('@/api/webImport')
  return {
    ...actual,
    extractImages: extractImagesMock,
    downloadImages: downloadImagesMock,
    checkGalleryDLSupport: checkGalleryDLSupportMock,
    getGalleryDLImages: getGalleryDLImagesMock,
    testFirecrawlConnection: testFirecrawlConnectionMock,
    testAgentConnection: testAgentConnectionMock,
  }
})

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['modelValue'],
    emits: ['close'],
    setup(_props, { slots }) {
      return () => h('div', [
        h('div', { class: 'modal-default-slot' }, slots.default ? slots.default() : []),
        h('div', { class: 'modal-footer-slot' }, slots.footer ? slots.footer() : []),
      ])
    },
  }),
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'options'],
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          class: 'custom-select-stub',
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        (props.options || []).map((option: { label: string; value: string }) =>
          h('option', { value: option.value }, option.label)
        )
      )
    },
  }),
}))

import WebImportModal from '@/components/translate/WebImportModal.vue'
import { useImageStore } from '@/stores/imageStore'
import { useWebImportStore } from '@/stores/webImportStore'

describe('WebImportModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()

    extractImagesMock.mockReset()
    downloadImagesMock.mockReset()
    checkGalleryDLSupportMock.mockReset()
    getGalleryDLImagesMock.mockReset()
    testFirecrawlConnectionMock.mockReset()
    testAgentConnectionMock.mockReset()

    checkGalleryDLSupportMock.mockResolvedValue({ available: true, supported: false })
    downloadImagesMock.mockResolvedValue({
      success: true,
      images: [{ index: 0, filename: 'page_0001.jpg', dataUrl: 'data:image/jpeg;base64,abc', size: 3 }],
      failedCount: 0,
    })
    getGalleryDLImagesMock.mockResolvedValue({ success: true, images: [], total: 0 })

    vi.spyOn(window, 'alert').mockImplementation(() => undefined)
    vi.spyOn(window, 'confirm').mockImplementation(() => true)

    extractImagesMock.mockImplementation(
      async (
        _url: string,
        _config: unknown,
        _onLog: unknown,
        onResult: (result: unknown) => void,
      ) => {
        onResult({
          success: true,
          comicTitle: 'Comic',
          chapterTitle: 'Chapter',
          pages: [{ pageNumber: 1, imageUrl: 'https://img.example/1.jpg' }],
          totalPages: 1,
          sourceUrl: 'https://example.com/chapter-1',
          engine: 'ai-agent',
        })
      }
    )
  })

  it('auto-imports extracted pages when autoImport is enabled', async () => {
    const webImportStore = useWebImportStore()
    const imageStore = useImageStore()
    webImportStore.modalVisible = true
    webImportStore.settings.ui.autoImport = true
    webImportStore.draftSettings.ui.autoImport = true

    const wrapper = mount(WebImportModal)

    await wrapper.find('.url-input').setValue('https://example.com/chapter-1')
    await wrapper.find('.extract-btn').trigger('click')
    await flushPromises()

    expect(downloadImagesMock).toHaveBeenCalledTimes(1)
    expect(imageStore.imageCount).toBe(1)
    expect(webImportStore.status).toBe('idle')
  })
})
