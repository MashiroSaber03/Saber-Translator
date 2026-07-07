import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

const {
  extractImagesMock,
  downloadImagesMock,
  checkGalleryDLSupportMock,
  getGalleryDLImagesMock,
  fetchModelsMock,
  testFirecrawlConnectionMock,
  testAgentConnectionMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  extractImagesMock: vi.fn(),
  downloadImagesMock: vi.fn(),
  checkGalleryDLSupportMock: vi.fn(),
  getGalleryDLImagesMock: vi.fn(),
  fetchModelsMock: vi.fn(),
  testFirecrawlConnectionMock: vi.fn(),
  testAgentConnectionMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
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

vi.mock('@/api/config', async () => {
  const actual = await vi.importActual<typeof import('@/api/config')>('@/api/config')
  return {
    ...actual,
    configApi: {
      ...actual.configApi,
      fetchModels: fetchModelsMock,
    },
  }
})

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'frameVariant', 'dividerVariant'],
    emits: ['close'],
    setup(props, { emit, slots }) {
      return () => h('div', {
        'data-frame': props.frameVariant,
        'data-divider': props.dividerVariant,
      }, [
        h('button', { type: 'button', class: 'modal-close', onClick: () => emit('close') }, '关闭'),
        h('div', { class: 'modal-default-slot' }, slots.default ? slots.default() : []),
        h('div', { class: 'modal-footer-slot' }, slots.footer ? slots.footer() : []),
      ])
    },
  }),
}))

import WebImportModal from '@/components/translate/WebImportModal.vue'
import ProductCollapsibleSection from '@/components/product/ProductCollapsibleSection.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import { useImageStore } from '@/stores/imageStore'
import { useWebImportStore } from '@/stores/webImportStore'

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

function mountedComboboxOptionValues(wrapper: ReturnType<typeof mount>) {
  return wrapper.findAllComponents(UiCombobox).flatMap(combobox =>
    (combobox.props('options') || []).map((option: { value: string | number }) => String(option.value))
  )
}

async function openWebImportSettings(wrapper: ReturnType<typeof mount>): Promise<void> {
  await wrapper.getComponent(ProductCollapsibleSection).get('button').trigger('click')
}

describe('WebImportModal', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()

    extractImagesMock.mockReset()
    downloadImagesMock.mockReset()
    checkGalleryDLSupportMock.mockReset()
    getGalleryDLImagesMock.mockReset()
    fetchModelsMock.mockReset()
    testFirecrawlConnectionMock.mockReset()
    testAgentConnectionMock.mockReset()
    confirmProductActionMock.mockReset()

    checkGalleryDLSupportMock.mockResolvedValue({ available: true, supported: false })
    downloadImagesMock.mockResolvedValue({
      success: true,
      images: [{ index: 0, filename: 'page_0001.jpg', dataUrl: 'data:image/jpeg;base64,abc', size: 3 }],
      failedCount: 0,
    })
    getGalleryDLImagesMock.mockResolvedValue({ success: true, images: [], total: 0 })
    fetchModelsMock.mockResolvedValue({
      success: true,
      models: [
        { id: 'gpt-4o-mini', name: 'gpt-4o-mini' },
        { id: 'gpt-4.1-mini', name: 'gpt-4.1-mini' },
      ],
    })
    confirmProductActionMock.mockResolvedValue(true)

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

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('auto-imports extracted pages when autoImport is enabled', async () => {
    const webImportStore = useWebImportStore()
    const imageStore = useImageStore()
    webImportStore.modalVisible = true
    webImportStore.settings.ui.autoImport = true
    webImportStore.draftSettings.ui.autoImport = true

    const wrapper = mount(WebImportModal)

    await wrapper.get('#webImportSourceUrl').setValue('https://example.com/chapter-1')
    await wrapper.get('form[aria-label="网页导入提取"]').trigger('submit')
    await flushPromises()

    expect(downloadImagesMock).toHaveBeenCalledTimes(1)
    expect(imageStore.imageCount).toBe(1)
    expect(webImportStore.status).toBe('idle')
  })

  it('fetches available models for the selected agent provider', async () => {
    const webImportStore = useWebImportStore()
    webImportStore.modalVisible = true
    webImportStore.draftSettings.agent.apiKey = 'test-key'
    webImportStore.draftSettings.agent.provider = 'openai'

    const wrapper = mount(WebImportModal)
    await openWebImportSettings(wrapper)

    const fetchButton = wrapper.findAll('button')
      .find(button => button.text().includes('获取模型'))
    expect(fetchButton).toBeTruthy()

    await fetchButton!.trigger('click')
    await flushPromises()

    expect(fetchModelsMock).toHaveBeenCalledWith('openai', 'test-key', '')

    const modelCombobox = wrapper.getComponent(UiCombobox)
    expect(modelCombobox.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ label: 'gpt-4o-mini', value: 'gpt-4o-mini' }),
    ]))
  })

  it('ignores stale model responses after the agent provider changes', async () => {
    const webImportStore = useWebImportStore()
    webImportStore.modalVisible = true
    webImportStore.draftSettings.agent.apiKey = 'test-key'
    webImportStore.draftSettings.agent.provider = 'openai'

    const pendingModels = createDeferred<{ success: boolean; models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReset()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(WebImportModal)
    await openWebImportSettings(wrapper)

    const fetchButton = wrapper.findAll('button')
      .find(button => button.text().includes('获取模型'))
    expect(fetchButton).toBeTruthy()

    await fetchButton!.trigger('click')
    expect(fetchModelsMock).toHaveBeenCalledWith('openai', 'test-key', '')

    webImportStore.draftSettings.agent.provider = 'deepseek'
    await flushPromises()

    pendingModels.resolve({
      success: true,
      models: [{ id: 'stale-web-import-model', name: 'Stale WebImport Model' }],
    })
    await flushPromises()

    expect(mountedComboboxOptionValues(wrapper)).not.toContain('stale-web-import-model')
  })

  it('uses the product confirmation workflow before closing during processing', async () => {
    const webImportStore = useWebImportStore()
    webImportStore.modalVisible = true
    webImportStore.setStatus('extracting')
    confirmProductActionMock.mockResolvedValueOnce(false)

    const wrapper = mount(WebImportModal)

    await wrapper.get('.modal-close').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '关闭网页导入',
      message: '正在处理中，确定要关闭吗？',
      confirmText: '关闭',
      cancelText: '继续处理',
      tone: 'danger',
    })
    expect(window.confirm).not.toHaveBeenCalled()
    expect(webImportStore.modalVisible).toBe(true)
    expect(webImportStore.status).toBe('extracting')

    confirmProductActionMock.mockResolvedValueOnce(true)
    await wrapper.get('.modal-close').trigger('click')
    await flushPromises()

    expect(webImportStore.modalVisible).toBe(false)
    expect(webImportStore.status).toBe('idle')
  })

  it('keeps modal shell visuals on typed BaseModal variants instead of a global style entry', () => {
    const webImportStore = useWebImportStore()
    webImportStore.modalVisible = true

    const wrapper = mount(WebImportModal)
    const modal = wrapper.get('[data-frame]')
    expect(modal.attributes('data-frame')).toBe('floating')
    expect(modal.attributes('data-divider')).toBe('soft')

    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/WebImportModal.vue'), 'utf8')
    expect(source).toContain('class="web-import-modal__body"')
    expect(source).not.toContain('web-import-modal-body')
    expect(source).not.toContain('WebImportModal.global.styles.css')
    expect(source).not.toContain('box-shadow=')
    expect(source).not.toContain('footer-border=')
    expect(source).not.toContain('footer-gap=')
  })
})
