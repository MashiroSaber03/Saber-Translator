import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ImageUpload from '@/components/translate/ImageUpload.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import type {
  SequentialImportOptions,
  SequentialImportSummary,
} from '@/api/v2/content'
import { getTextStyleDefaults } from '@/defaults/textStyleDefaults'
import { useWebImportStore } from '@/stores/webImportStore'

const textStyle = {
  ...getTextStyleDefaults(),
  fontSize: 37,
}

function mountImageUpload(chapterId = 'chapter-1') {
  return mount(ImageUpload, {
    props: {
      chapterId,
      textStyle,
    },
  })
}

const mocks = vi.hoisted(() => ({
  createContainerImportJob: vi.fn(),
  importImagesSequentially: vi.fn(),
  retryFailedImageImports: vi.fn(),
  toast: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  createContainerImportJob: mocks.createContainerImportJob,
  importImagesSequentially: mocks.importImagesSequentially,
  retryFailedImageImports: mocks.retryFailedImageImports,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.toast,
}))

describe('ImageUpload', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.importImagesSequentially.mockResolvedValue({
      failures: [],
      results: [{ pageId: 'page-1' }],
    })
    mocks.retryFailedImageImports.mockResolvedValue({ failures: [], results: [] })
    mocks.createContainerImportJob.mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
  })

  it('uses product upload primitives for files, folders, and web import', async () => {
    const webImportStore = useWebImportStore()
    const openModal = vi.spyOn(webImportStore, 'openModal')
    const wrapper = mountImageUpload()

    expect(wrapper.getComponent(ProductFileDropzone).props()).toMatchObject({
      inputId: 'imageUpload',
      accept: 'image/*,application/pdf,.zip,.cbz,.mobi,.azw,.azw3',
      multiple: true,
      label: '上传翻译源文件',
    })
    expect(wrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('其他导入方式')

    await wrapper.get('button[aria-label="从网页导入漫画图片"]').trigger('click')
    expect(openModal).toHaveBeenCalled()
  })

  it('uploads ordinary images directly into the backend chapter', async () => {
    const wrapper = mountImageUpload()
    const file = new File(['image'], '001.png', { type: 'image/png' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await flushPromises()

    expect(mocks.importImagesSequentially).toHaveBeenCalledWith(
      'chapter-1',
      [file],
      textStyle,
      expect.objectContaining({ onProgress: expect.any(Function) }),
    )
    expect(wrapper.emitted('uploadComplete')).toEqual([[1]])
    expect(mocks.toast).toHaveBeenCalledWith('已写入后端 1 张图片', 'success')
  })

  it('keeps small non-zero upload progress visible for large batches', async () => {
    let resolveImport!: (summary: SequentialImportSummary) => void
    mocks.importImagesSequentially.mockReturnValueOnce(
      new Promise<SequentialImportSummary>(resolve => {
        resolveImport = resolve
      }),
    )
    const wrapper = mountImageUpload()
    const file = new File(['image'], '001.png', { type: 'image/png' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await vi.waitFor(() => expect(mocks.importImagesSequentially).toHaveBeenCalled())
    const options = mocks.importImagesSequentially.mock.calls[0]?.[3] as SequentialImportOptions
    options.onProgress?.({
      completed: 8,
      currentPath: '008.png',
      failed: 0,
      succeeded: 8,
      total: 2702,
    })
    await wrapper.vm.$nextTick()

    const percent = wrapper.getComponent(UiProgressBar).props('value') as number
    expect(percent).toBeCloseTo(8 / 2702 * 100)
    expect(percent).toBeGreaterThan(0)

    resolveImport({ failures: [], results: [] })
    await flushPromises()
  })

  it('submits PDF and comic archives as durable backend jobs', async () => {
    const wrapper = mountImageUpload()
    const file = new File(['pdf'], 'chapter.pdf', { type: 'application/pdf' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await flushPromises()

    expect(mocks.createContainerImportJob).toHaveBeenCalledWith(
      'chapter-1',
      file,
      textStyle,
    )
    expect(wrapper.emitted('contentImportAccepted')).toEqual([[['job-1']]])
    expect(mocks.toast).toHaveBeenCalledWith(
      '已创建 1 个后端解析任务，可安全关闭页面',
      'success',
    )
    expect(wrapper.emitted('uploadComplete')).toBeUndefined()
  })

  it('does not accept files before the translation context is ready', async () => {
    const wrapper = mount(ImageUpload, {
      props: {
        chapterId: 'chapter-1',
        disabled: true,
        textStyle,
      },
    })

    expect(wrapper.getComponent(ProductFileDropzone).props('disabled')).toBe(true)
    wrapper.getComponent(ProductFileDropzone).vm.$emit(
      'select',
      [new File(['image'], '001.png', { type: 'image/png' })],
    )
    await flushPromises()

    expect(mocks.importImagesSequentially).not.toHaveBeenCalled()
    expect(mocks.createContainerImportJob).not.toHaveBeenCalled()
  })

  it('renders backend upload errors through the product status banner', async () => {
    mocks.importImagesSequentially.mockRejectedValueOnce(new Error('backend rejected'))
    const wrapper = mountImageUpload()

    wrapper.getComponent(ProductFileDropzone).vm.$emit(
      'select',
      [new File(['image'], 'broken.png', { type: 'image/png' })],
    )
    await flushPromises()

    expect(wrapper.getComponent(ProductStatusBanner).props('tone')).toBe('danger')
    expect(wrapper.text()).toContain('backend rejected')
  })

  it('keeps partial success and retries only failed images with their original keys', async () => {
    const good = new File(['good'], '001.png', { type: 'image/png' })
    const bad = new File(['bad'], '002.png', { type: 'image/png' })
    const failed = {
      entry: { file: bad, logicalPath: '002.png' },
      error: new Error('连接中断'),
      idempotencyKey: 'stable-key',
    }
    mocks.importImagesSequentially.mockResolvedValueOnce({
      failures: [failed],
      results: [{ pageId: 'page-1' }],
    })
    mocks.retryFailedImageImports.mockResolvedValueOnce({
      failures: [],
      results: [{ pageId: 'page-2' }],
    })
    const wrapper = mountImageUpload()

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [good, bad])
    await flushPromises()
    await wrapper.setProps({
      textStyle: {
        ...textStyle,
        fontSize: 99,
      },
    })

    expect(wrapper.emitted('uploadComplete')).toEqual([[1]])
    expect(wrapper.text()).toContain('仅重试失败项')
    const retryButton = wrapper.findAll('button').find(button => button.text() === '仅重试失败项')
    if (!retryButton) throw new Error('retry button was not rendered')
    await retryButton.trigger('click')
    await flushPromises()

    expect(mocks.retryFailedImageImports).toHaveBeenCalledWith(
      'chapter-1',
      [failed],
      textStyle,
      expect.objectContaining({
        onProgress: expect.any(Function),
        onRetry: expect.any(Function),
      }),
    )
    expect(wrapper.emitted('uploadComplete')).toEqual([[1], [1]])
  })

  it('does not start a second upload while the first batch is in flight', async () => {
    let resolveImport!: (summary: SequentialImportSummary) => void
    mocks.importImagesSequentially.mockReturnValueOnce(new Promise(resolve => {
      resolveImport = resolve
    }))
    const wrapper = mountImageUpload()
    const first = new File(['first'], '001.png', { type: 'image/png' })
    const second = new File(['second'], '002.png', { type: 'image/png' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [first])
    await vi.waitFor(() => expect(mocks.importImagesSequentially).toHaveBeenCalledTimes(1))
    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [second])
    await flushPromises()

    expect(mocks.importImagesSequentially).toHaveBeenCalledTimes(1)
    resolveImport({ failures: [], results: [] })
    await flushPromises()
  })

  it('keeps an in-flight upload bound to its starting chapter without refreshing a new chapter', async () => {
    let resolveImport!: (summary: SequentialImportSummary) => void
    mocks.importImagesSequentially.mockReturnValueOnce(new Promise(resolve => {
      resolveImport = resolve
    }))
    const wrapper = mountImageUpload()
    const file = new File(['image'], '001.png', { type: 'image/png' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await vi.waitFor(() => expect(mocks.importImagesSequentially).toHaveBeenCalled())
    await wrapper.setProps({ chapterId: 'chapter-2' })
    resolveImport({ failures: [], results: [{ pageId: 'page-1' }] })
    await flushPromises()

    expect(mocks.importImagesSequentially).toHaveBeenCalledWith(
      'chapter-1',
      [file],
      textStyle,
      expect.any(Object),
    )
    expect(wrapper.emitted('uploadComplete')).toBeUndefined()
    expect(mocks.toast).not.toHaveBeenCalledWith('已写入后端 1 张图片', 'success')
  })

  it('contains no FileReader, PDF.js, or browser Base64 import pipeline', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'),
      'utf8',
    )

    expect(source).toContain('importImagesSequentially')
    expect(source).toContain('createContainerImportJob')
    expect(source).not.toContain('FileReader')
    expect(source).not.toContain('pdfjs-dist')
    expect(source).not.toContain('readAsDataURL')
    expect(source).not.toContain('base64')
  })
})
