import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

import OrthographicDialog from '@/components/insight/continuation/OrthographicDialog.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

const componentSourcePath = resolve(process.cwd(), 'src/components/insight/continuation/OrthographicDialog.vue')

const dialogShellStub = defineComponent({
  template: '<div><slot /><slot name="footer" /></div>',
})

function setInputFiles(input: HTMLInputElement, files: File[]): void {
  Object.defineProperty(input, 'files', {
    value: files,
    configurable: true,
  })
}

function findButtonByText(wrapper: ReturnType<typeof mount>, text: string) {
  const button = wrapper.findAll('button').find(candidate => candidate.text().includes(text))
  expect(button, `button containing "${text}"`).toBeTruthy()
  return button!
}

describe('OrthographicDialog', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('revokes source image preview URLs when files change and when unmounted', async () => {
    const createObjectURLSpy = vi
      .spyOn(window.URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)
    const revokeObjectURLSpy = vi
      .spyOn(window.URL, 'revokeObjectURL')
      .mockImplementation(() => {})
    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: '主角',
        formId: 'default',
        formName: '默认',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: {
          ContinuationDialogShell: dialogShellStub,
        },
      },
    })

    const input = wrapper.find('input[type="file"]').element as HTMLInputElement
    const firstFile = new File(['first'], 'first.png', { type: 'image/png' })
    const secondFile = new File(['second'], 'second.png', { type: 'image/png' })

    setInputFiles(input, [firstFile])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    expect(createObjectURLSpy).toHaveBeenCalledWith(firstFile)

    setInputFiles(input, [secondFile])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:first.png')

    wrapper.unmount()

    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:second.png')
  })

  it('clears progress message timers when unmounted during generation', async () => {
    vi.useFakeTimers()
    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    vi
      .spyOn(window.URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)
    vi
      .spyOn(window.URL, 'revokeObjectURL')
      .mockImplementation(() => {})

    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: '主角',
        formId: 'default',
        formName: '默认',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: {
          ContinuationDialogShell: dialogShellStub,
        },
      },
    })

    const input = wrapper.find('input[type="file"]').element as HTMLInputElement
    const sourceFile = new File(['first'], 'first.png', { type: 'image/png' })
    setInputFiles(input, [sourceFile])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    await wrapper.findAll('button')[1].trigger('click')
    wrapper.unmount()

    expect(clearTimeoutSpy).toHaveBeenCalledTimes(2)
  })

  it('uses product feedback primitives for generation and result states', async () => {
    vi
      .spyOn(window.URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)
    vi
      .spyOn(window.URL, 'revokeObjectURL')
      .mockImplementation(() => {})

    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: '主角',
        formId: 'default',
        formName: '默认',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: {
          ContinuationDialogShell: dialogShellStub,
        },
      },
    })

    const input = wrapper.find('input[type="file"]').element as HTMLInputElement
    setInputFiles(input, [new File(['first'], 'first.png', { type: 'image/png' })])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    await findButtonByText(wrapper, '生成三视图').trigger('click')
    await wrapper.setProps({ isGenerating: true })
    await nextTick()

    const banner = wrapper.getComponent(ProductStatusBanner)
    expect(banner.props('tone')).toBe('info')
    expect(banner.props('ariaLive')).toBe('polite')
    expect(wrapper.getComponent(UiSpinner).props('label')).toBe('三视图生成中')

    await wrapper.setProps({
      isGenerating: false,
      resultImagePath: 'result.png',
    })
    await nextTick()

    expect(wrapper.getComponent(ProductRecordCard).text()).toContain('生成结果')
    expect(wrapper.get('img[alt="主角三视图生成结果"]').attributes('src')).toContain('result.png')
  })

  it('uses typed props for generation state instead of exposed instance methods', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).toContain('resultImagePath')
    expect(source).toContain('isGenerating')
    expect(source).not.toContain('defineExpose')
    expect(source).not.toContain('setResult')
    expect(source).not.toContain('setGenerating')
  })

  it('renders source-image upload through the product file dropzone', () => {
    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: '主角',
        formId: 'default',
        formName: '默认',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: {
          ContinuationDialogShell: dialogShellStub,
        },
      },
    })

    const dropzone = wrapper.getComponent(ProductFileDropzone)

    expect(dropzone.props()).toMatchObject({
      inputId: 'orthographicSourceImages',
      accept: 'image/*',
      multiple: true,
      label: '上传 主角 默认 三视图源图',
    })
    expect(wrapper.find('.upload-area').exists()).toBe(false)
    expect(readFileSync(componentSourcePath, 'utf8')).not.toContain("import UiFileInput from '@/components/ui/UiFileInput.vue'")
  })

  it('renders selected source-image previews through the product thumbnail grid', async () => {
    vi
      .spyOn(window.URL, 'createObjectURL')
      .mockImplementation(file => `blob:${(file as File).name}`)

    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: '主角',
        formId: 'default',
        formName: '默认',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: {
          ContinuationDialogShell: dialogShellStub,
        },
      },
    })

    const input = wrapper.find('input[type="file"]').element as HTMLInputElement
    setInputFiles(input, [
      new File(['first'], 'first.png', { type: 'image/png' }),
      new File(['second'], 'second.png', { type: 'image/png' }),
    ])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    const thumbnailGrid = wrapper.getComponent(ProductThumbnailGrid)
    expect(thumbnailGrid.props('ariaLabel')).toBe('三视图源图预览')
    expect(thumbnailGrid.props('items')).toEqual([
      expect.objectContaining({
        id: 'blob:first.png',
        alt: '源图1',
        cornerLabel: '1',
        interactive: false,
        label: '源图 1',
        src: 'blob:first.png',
      }),
      expect.objectContaining({
        id: 'blob:second.png',
        alt: '源图2',
        cornerLabel: '2',
        interactive: false,
        label: '源图 2',
        src: 'blob:second.png',
      }),
    ])
    expect(wrapper.find('.source-images').exists()).toBe(false)
    expect(wrapper.find('.source-image').exists()).toBe(false)
  })

  it('maps local upload roles through semantic tokens', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    expect(source).not.toContain('--orthographic-dialog-upload-hover-background: rgba(99, 102, 241, .05)')
    expect(source).toContain('--product-file-dropzone-background-hover: var(--color-focus-brand-soft)')
    expect(source).not.toContain('.source-images')
    expect(source).not.toContain('.source-image')
    expect(source).not.toContain('.image-index')
  })

  it('keeps dialog presentation hooks under the orthographic-dialog owner', () => {
    const source = readFileSync(componentSourcePath, 'utf8')

    for (const oldClass of [
      'orthographic-dialog-body',
      'ortho-upload-section',
      'upload-placeholder',
      'upload-icon',
      'generating-state',
      'progress-message',
      'progress-tip',
      'ortho-result',
      'result-preview',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toContain('class="hint"')
    expect(source).not.toContain('.hint')
    expect(source).not.toMatch(/\.orthographic-dialog__[^{]+ p\b/)
    expect(source).not.toMatch(/\.orthographic-dialog__[^{]+ img\b/)

    for (const ownerClass of [
      'orthographic-dialog__body',
      'orthographic-dialog__upload-section',
      'orthographic-dialog__upload-placeholder',
      'orthographic-dialog__upload-icon',
      'orthographic-dialog__upload-hint',
      'orthographic-dialog__generating-state',
      'orthographic-dialog__generating-content',
      'orthographic-dialog__progress-message',
      'orthographic-dialog__progress-tip',
      'orthographic-dialog__result',
      'orthographic-dialog__result-preview',
      'orthographic-dialog__result-image',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
