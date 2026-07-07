import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ImageUpload from '@/components/translate/ImageUpload.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiProgressBar from '@/components/ui/UiProgressBar.vue'
import { useWebImportStore } from '@/stores/webImportStore'

vi.mock('@/api/system', () => ({
  parsePdfStart: vi.fn(),
  parsePdfBatch: vi.fn(),
  parsePdfCleanup: vi.fn(),
  parseMobiStart: vi.fn(),
  parseMobiBatch: vi.fn(),
  parseMobiCleanup: vi.fn(),
}))

describe('ImageUpload', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses product upload primitives for the entry shell and secondary actions', async () => {
    const webImportStore = useWebImportStore()
    const openModalSpy = vi.spyOn(webImportStore, 'openModal')
    const wrapper = mount(ImageUpload)

    const dropzone = wrapper.getComponent(ProductFileDropzone)
    expect(dropzone.props()).toMatchObject({
      inputId: 'imageUpload',
      accept: 'image/*,application/pdf,.mobi,.azw,.azw3',
      multiple: true,
      label: '上传翻译源文件',
    })
    expect(wrapper.getComponent(ProductActionRow).props()).toMatchObject({
      ariaLabel: '其他导入方式',
      justify: 'center',
    })
    expect(wrapper.find('.drop-area').exists()).toBe(false)
    expect(wrapper.find('.select-link').exists()).toBe(false)

    await wrapper.get('button[aria-label="从网页导入漫画图片"]').trigger('click')

    expect(openModalSpy).toHaveBeenCalled()
  })

  it('uses the typed file-input contract for folder selection', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).toContain('@files-change="handleFolderSelect"')
    expect(source).toContain('folderInputRef.value?.clear()')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('@change="handleFolderSelect"')
    expect(source).not.toContain('input.value =')
  })

  it('renders upload errors through the product status banner', async () => {
    class FailingFileReader {
      onerror: (() => void) | null = null
      onload: (() => void) | null = null
      readAsDataURL(): void {
        this.onerror?.()
      }
    }

    vi.stubGlobal('FileReader', FailingFileReader)

    const wrapper = mount(ImageUpload)

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [new File(['broken'], 'broken.png', { type: 'image/png' })])
    await flushPromises()

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('danger')
    expect(banner.props('ariaLive')).toBe('assertive')
    expect(wrapper.find('.error-message').exists()).toBe(false)
    const dismissAction = wrapper.getComponent(UiIconButton)
    expect(dismissAction.props('label')).toBe('关闭上传错误提示')

    await wrapper.get('button[aria-label="关闭上传错误提示"]').trigger('click')

    expect(wrapper.findComponent(ProductStatusBanner).exists()).toBe(false)
  })

  it('keeps upload error text on owner-prefixed hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).toContain('class="image-upload__error-text"')
    expect(source).not.toContain('class="error-text"')
    expect(source).not.toMatch(/\.error-text\s*{/)
  })

  it('renders upload progress through the shared progress primitive', async () => {
    let finishRead: (() => void) | null = null

    class PendingFileReader {
      onerror: (() => void) | null = null
      onload: ((event: { target: { result: string } }) => void) | null = null

      readAsDataURL(): void {
        finishRead = () => {
          this.onload?.({ target: { result: 'data:image/png;base64,aW1hZ2U=' } })
        }
      }
    }

    vi.stubGlobal('FileReader', PendingFileReader)

    const wrapper = mount(ImageUpload)

    wrapper.getComponent(ProductFileDropzone).vm.$emit('select', [new File(['image'], 'page.png', { type: 'image/png' })])
    await nextTick()

    const progress = wrapper.getComponent(UiProgressBar)
    expect(progress.props('value')).toBe(0)
    expect(progress.text()).toContain('page.png')
    expect(wrapper.find('.progress').exists()).toBe(false)

    finishRead?.()
    await flushPromises()
  })

  it('maps dropzone owner colors through product and semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).toContain('--product-file-dropzone-background')
    expect(styleBlock).toContain('--product-file-dropzone-border')
    expect(styleBlock).toContain('--color-border-muted')
    expect(styleBlock).toContain('--color-surface-interactive-hover')
  })

  it('does not keep the unreachable local loading spinner overlay', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).not.toContain('isLoading && !showProgress')
    expect(source).not.toContain('class="loading-overlay"')
    expect(source).not.toContain('class="spinner"')
    expect(source).not.toMatch(/\.spinner\s*{[\s\S]*animation:\s*spin/)
  })

  it('guards frontend PDF canvas contexts without double type escapes', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).not.toContain('as unknown as CanvasRenderingContext2D')
    expect(source).not.toContain("getContext('2d')!")
    expect(source).toContain("throw new Error('无法创建 PDF 渲染上下文')")
  })

  it('uses shared document parse helpers for backend document imports', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).toContain('buildDocumentParseBatches')
    expect(source).toContain('calculateDocumentParseProgress')
    expect(source).toContain('createDocumentPageFileName')
    expect(source).not.toContain('for (let startIndex = 0; startIndex < totalPages; startIndex += BATCH_SIZE)')
    expect(source).not.toContain('Math.round((startIndex / totalPages) * 100)')
    expect(source).not.toContain('String(imgData.page_index + 1).padStart(4')
    expect(source).not.toContain("file.name.replace(/\\.(mobi|azw|azw3)$/i, '')")
  })

  it('keeps upload workflow private and driven by product input events', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/ImageUpload.vue'), 'utf8')

    expect(source).not.toContain('defineExpose')
    expect(source).toContain('@select="handleFileSelect"')
    expect(source).toContain('@click="triggerFolderSelect"')
    expect(source).toContain('@click="clearError"')
  })
})
