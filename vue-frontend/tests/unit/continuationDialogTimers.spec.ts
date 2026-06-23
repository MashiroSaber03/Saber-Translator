import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import AddCharacterDialog from '@/components/insight/continuation/AddCharacterDialog.vue'
import AddFormDialog from '@/components/insight/continuation/AddFormDialog.vue'
import EditFormDialog from '@/components/insight/continuation/EditFormDialog.vue'

const shellStub = {
  template: '<section><slot /><slot name="footer" /></section>',
}

describe('Continuation dialog loading timers', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('clears the add-character loading timer on unmount', async () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    const wrapper = mount(AddCharacterDialog, {
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    await wrapper.find('input').setValue('夏')
    await wrapper.findAll('button')[1].trigger('click')
    wrapper.unmount()

    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1)
  })

  it('clears the add-form loading timer on unmount', async () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    const wrapper = mount(AddFormDialog, {
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    await wrapper.find('input').setValue('常服')
    await wrapper.findAll('button')[1].trigger('click')
    wrapper.unmount()

    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1)
  })

  it('clears the edit-form saving timer on unmount', async () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    const wrapper = mount(EditFormDialog, {
      props: {
        form: {
          form_name: '常服',
          description: '日常形态',
          reference_image: null,
        },
      },
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    await wrapper.findAll('button')[1].trigger('click')
    wrapper.unmount()

    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1)
  })
})
