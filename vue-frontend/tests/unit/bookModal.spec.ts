import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it } from 'vitest'
import BookModal from '@/components/bookshelf/BookModal.vue'

const BaseModalStub = defineComponent({
  template: '<section class="base-modal-stub"><slot /><footer><slot name="footer" /></footer></section>',
})

describe('BookModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses a native label and file input association for cover upload', () => {
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const uploadArea = wrapper.get('.cover-upload-area')
    const fileInput = wrapper.get('input[type="file"]')

    expect(uploadArea.element.tagName).toBe('LABEL')
    expect(uploadArea.attributes('for')).toBe('bookCoverInput')
    expect(fileInput.attributes('id')).toBe('bookCoverInput')
  })
})
