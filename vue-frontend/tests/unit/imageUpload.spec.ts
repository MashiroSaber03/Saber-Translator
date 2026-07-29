import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ImageUpload from '@/components/translate/ImageUpload.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import { useWebImportStore } from '@/stores/webImportStore'

const mocks = vi.hoisted(() => ({
  createContainerImportJob: vi.fn(),
  importImagesSequentially: vi.fn(),
  toast: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  createContainerImportJob: mocks.createContainerImportJob,
  importImagesSequentially: mocks.importImagesSequentially,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.toast,
}))

describe('ImageUpload', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.importImagesSequentially.mockResolvedValue([{ pageId: 'page-1' }])
    mocks.createContainerImportJob.mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
  })

  it('uses product upload primitives for files, folders, and web import', async () => {
    const webImportStore = useWebImportStore()
    const openModal = vi.spyOn(webImportStore, 'openModal')
    const wrapper = mount(ImageUpload, { props: { chapterId: 'chapter-1' } })

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
    const wrapper = mount(ImageUpload, { props: { chapterId: 'chapter-1' } })
    const file = new File(['image'], '001.png', { type: 'image/png' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await flushPromises()

    expect(mocks.importImagesSequentially).toHaveBeenCalledWith(
      'chapter-1',
      [file],
      expect.objectContaining({ onProgress: expect.any(Function) }),
    )
    expect(wrapper.emitted('uploadComplete')).toEqual([[1]])
    expect(mocks.toast).toHaveBeenCalledWith('已写入后端 1 张图片', 'success')
  })

  it('submits PDF and comic archives as durable backend jobs', async () => {
    const wrapper = mount(ImageUpload, { props: { chapterId: 'chapter-1' } })
    const file = new File(['pdf'], 'chapter.pdf', { type: 'application/pdf' })

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [file])
    await flushPromises()

    expect(mocks.createContainerImportJob).toHaveBeenCalledWith('chapter-1', file)
    expect(mocks.toast).toHaveBeenCalledWith(
      '已创建 1 个后端解析任务，可安全关闭页面',
      'success',
    )
  })

  it('renders backend upload errors through the product status banner', async () => {
    mocks.importImagesSequentially.mockRejectedValueOnce(new Error('backend rejected'))
    const wrapper = mount(ImageUpload, { props: { chapterId: 'chapter-1' } })

    wrapper.getComponent(ProductFileDropzone).vm.$emit(
      'select',
      [new File(['image'], 'broken.png', { type: 'image/png' })],
    )
    await flushPromises()

    expect(wrapper.getComponent(ProductStatusBanner).props('tone')).toBe('danger')
    expect(wrapper.text()).toContain('backend rejected')
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
