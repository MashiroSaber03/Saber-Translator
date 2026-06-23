import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'

import AddCharacterDialog from '@/components/insight/continuation/AddCharacterDialog.vue'
import AddFormDialog from '@/components/insight/continuation/AddFormDialog.vue'
import EditCharacterDialog from '@/components/insight/continuation/EditCharacterDialog.vue'
import EditFormDialog from '@/components/insight/continuation/EditFormDialog.vue'
import OrthographicDialog from '@/components/insight/continuation/OrthographicDialog.vue'

const shellStub = defineComponent({
  template: '<section><slot /><slot name="footer" /></section>',
})

const dialogStubs = {
  ContinuationDialogShell: shellStub,
}

describe('continuation dialog field names', () => {
  it('exposes explicit names for add-character fields', () => {
    const wrapper = mount(AddCharacterDialog, {
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[aria-label="角色名称"]').exists()).toBe(true)
    expect(wrapper.find('input[aria-label="别名（用逗号分隔，可选）"]').exists()).toBe(true)
    expect(wrapper.find('textarea[aria-label="角色描述（可选）"]').exists()).toBe(true)
  })

  it('exposes explicit names for add-form fields', () => {
    const wrapper = mount(AddFormDialog, {
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[aria-label="形态名称"]').exists()).toBe(true)
    expect(wrapper.find('textarea[aria-label="形态描述（可选）"]').exists()).toBe(true)
  })

  it('exposes explicit names for edit-character fields', () => {
    const wrapper = mount(EditCharacterDialog, {
      props: {
        character: {
          name: 'Saber',
          aliases: ['骑士王'],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      },
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[aria-label="角色名称"]').exists()).toBe(true)
    expect(wrapper.find('input[aria-label="别名（用逗号分隔）"]').exists()).toBe(true)
  })

  it('exposes explicit names for edit-form fields', () => {
    const wrapper = mount(EditFormDialog, {
      props: {
        form: {
          form_id: 'form_1',
          form_name: '常服',
          description: '日常服装',
          reference_image: '',
          enabled: true,
        },
      },
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[aria-label="形态名称"]').exists()).toBe(true)
    expect(wrapper.find('textarea[aria-label="形态描述"]').exists()).toBe(true)
  })

  it('exposes an explicit name for orthographic source image upload', () => {
    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: 'Saber',
        formId: 'form_1',
        formName: '常服',
        bookId: 'book-1',
      },
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[type="file"]').attributes('aria-label')).toBe('上传 Saber 常服 三视图源图')
  })
})
