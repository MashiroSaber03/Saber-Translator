import { mount } from '@vue/test-utils'
import { h } from 'vue'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'

describe('ProductFileDropzone', () => {
  it('owns the labelled file-input and dropzone shell', () => {
    const wrapper = mount(ProductFileDropzone, {
      props: {
        inputId: 'coverInput',
        accept: 'image/*',
        label: '上传封面',
      },
      slots: {
        default: '<span data-test="content">选择图片</span>',
      },
    })

    const label = wrapper.get('label.product-file-dropzone')
    const input = wrapper.get('input[type="file"]')

    expect(label.attributes('for')).toBe('coverInput')
    expect(label.attributes('aria-label')).toBe('上传封面')
    expect(input.attributes('id')).toBe('coverInput')
    expect(input.attributes('accept')).toBe('image/*')
    expect(wrapper.get('[data-test="content"]').text()).toBe('选择图片')
  })

  it('uses the typed file-input contract instead of raw input change plumbing', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/product/ProductFileDropzone.vue'), 'utf8')

    expect(source).toContain('@files-change="handleFilesChange"')
    expect(source).toContain('dropzoneInputRef.value?.clear()')
    expect(source).not.toContain('@change="handleInputChange"')
    expect(source).not.toContain('event.target as HTMLInputElement')
    expect(source).not.toContain('input.value =')
  })

  it('emits selected files from input and drop interactions', async () => {
    const wrapper = mount(ProductFileDropzone, {
      props: {
        inputId: 'coverInput',
        label: '上传封面',
      },
    })
    const file = new File(['cover'], 'cover.png', { type: 'image/png' })
    const input = wrapper.get('input[type="file"]').element as HTMLInputElement

    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [file],
    })
    await wrapper.get('input[type="file"]').trigger('change')
    expect(wrapper.emitted('select')?.at(-1)).toEqual([[file]])

    await wrapper.get('label.product-file-dropzone').trigger('drop', {
      dataTransfer: {
        files: [file],
      },
    })
    expect(wrapper.emitted('select')?.at(-1)).toEqual([[file]])
  })

  it('exposes drag-active state to the content slot', async () => {
    const wrapper = mount(ProductFileDropzone, {
      props: {
        inputId: 'sourceImages',
        label: '上传源图',
      },
      slots: {
        default: ({ isDragging }: { isDragging: boolean }) => h(
          'span',
          { 'data-test': 'state' },
          isDragging ? '释放上传' : '拖拽上传'
        ),
      },
    })

    await wrapper.get('label.product-file-dropzone').trigger('dragenter')
    expect(wrapper.classes()).toContain('product-file-dropzone--dragging')
    expect(wrapper.get('[data-test="state"]').text()).toBe('释放上传')

    await wrapper.get('label.product-file-dropzone').trigger('dragleave', {
      relatedTarget: wrapper.get('[data-test="state"]').element,
    })
    expect(wrapper.classes()).toContain('product-file-dropzone--dragging')

    await wrapper.get('label.product-file-dropzone').trigger('dragleave')
    expect(wrapper.classes()).not.toContain('product-file-dropzone--dragging')
    expect(wrapper.get('[data-test="state"]').text()).toBe('拖拽上传')
  })
})
