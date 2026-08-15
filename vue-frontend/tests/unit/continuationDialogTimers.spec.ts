import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'

import AddCharacterDialog from '@/components/insight/continuation/AddCharacterDialog.vue'
import AddFormDialog from '@/components/insight/continuation/AddFormDialog.vue'
import EditFormDialog from '@/components/insight/continuation/EditFormDialog.vue'

const shellStub = defineComponent({
  name: 'ContinuationDialogShell',
  props: {
    dismissible: {
      type: Boolean,
      default: true,
    },
  },
  template: '<section><slot /><slot name="footer" /></section>',
})

describe('Continuation dialog pending state', () => {
  it('keeps add-character pending state owned by the real request', async () => {
    const wrapper = mount(AddCharacterDialog, {
      props: { busy: true },
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    expect(wrapper.getComponent(shellStub).props('dismissible')).toBe(false)
    expect(wrapper.findAll('button').every(button => button.attributes('disabled') !== undefined)).toBe(true)
  })

  it('keeps add-form pending state owned by the real request', async () => {
    const wrapper = mount(AddFormDialog, {
      props: { busy: true },
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    expect(wrapper.getComponent(shellStub).props('dismissible')).toBe(false)
    expect(wrapper.findAll('button').every(button => button.attributes('disabled') !== undefined)).toBe(true)
  })

  it('keeps edit-form pending state owned by the real request', async () => {
    const wrapper = mount(EditFormDialog, {
      props: {
        form: {
          form_name: '常服',
          description: '日常形态',
          reference_image: null,
        },
        busy: true,
      },
      global: {
        stubs: {
          ContinuationDialogShell: shellStub,
        },
      },
    })

    expect(wrapper.getComponent(shellStub).props('dismissible')).toBe(false)
    expect(wrapper.findAll('button').every(button => button.attributes('disabled') !== undefined)).toBe(true)
  })
})
