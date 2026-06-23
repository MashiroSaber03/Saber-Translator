import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

import OrthographicDialog from '@/components/insight/continuation/OrthographicDialog.vue'

const dialogShellStub = defineComponent({
  template: '<div><slot /><slot name="footer" /></div>',
})

function setInputFiles(input: HTMLInputElement, files: File[]): void {
  Object.defineProperty(input, 'files', {
    value: files,
    configurable: true,
  })
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
})
