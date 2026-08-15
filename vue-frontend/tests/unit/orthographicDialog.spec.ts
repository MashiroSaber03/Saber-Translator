import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

import OrthographicDialog from '@/components/insight/continuation/OrthographicDialog.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

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
      multiple: false,
      label: '上传 主角 默认 三视图源图',
    })
    expect(wrapper.find('.upload-area').exists()).toBe(false)
  })

  it('renders the selected source-image preview through the product thumbnail grid', async () => {
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
    setInputFiles(input, [new File(['first'], 'first.png', { type: 'image/png' })])
    await wrapper.find('input[type="file"]').trigger('change')
    await nextTick()

    const thumbnailGrid = wrapper.getComponent(ProductThumbnailGrid)
    expect(thumbnailGrid.props('ariaLabel')).toBe('三视图源图预览')
    expect(thumbnailGrid.props('items')).toEqual([
      expect.objectContaining({
        id: 'blob:first.png',
        alt: '角色参考图',
        interactive: false,
        label: '角色参考图',
        src: 'blob:first.png',
      }),
    ])
    expect(wrapper.find('.source-images').exists()).toBe(false)
    expect(wrapper.find('.source-image').exists()).toBe(false)
  })

})
