import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it } from 'vitest'

import AddCharacterDialog from '@/components/insight/continuation/AddCharacterDialog.vue'
import AddFormDialog from '@/components/insight/continuation/AddFormDialog.vue'
import EditCharacterDialog from '@/components/insight/continuation/EditCharacterDialog.vue'
import EditFormDialog from '@/components/insight/continuation/EditFormDialog.vue'
import OrthographicDialog from '@/components/insight/continuation/OrthographicDialog.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

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

  it('renders dialog footer actions through the product action row', () => {
    const wrapper = mount(AddCharacterDialog, {
      global: {
        stubs: dialogStubs,
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('justify')).toBe('end')
    expect(wrapper.find('.continuation-dialog-actions').exists()).toBe(false)
  })

  it('renders dialog fields through the shared UiField primitive', () => {
    const wrapper = mount(AddCharacterDialog, {
      global: {
        stubs: dialogStubs,
      },
    })

    const fields = wrapper.findAllComponents(UiField)
    expect(fields.map(field => field.props('label'))).toEqual([
      '角色名称',
      '别名（用逗号分隔，可选）',
      '角色描述（可选）',
    ])
    expect(fields[0]?.props('variant')).toBe('dialog')
    expect(fields[0]?.props('required')).toBe(true)
    expect(fields[0]?.props('error')).toBe('')
    expect(wrapper.find('.continuation-dialog-field__label').exists()).toBe(false)
  })

  it('links add-character dialog labels to stable input controls', () => {
    const wrapper = mount(AddCharacterDialog, {
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.findAllComponents(UiField).map(field => field.props('controlId'))).toEqual([
      'continuationAddCharacterName',
      'continuationAddCharacterAliases',
      'continuationAddCharacterDescription',
    ])
    expect(wrapper.get('#continuationAddCharacterName').attributes('aria-label')).toBe('角色名称')
    expect(wrapper.get('#continuationAddCharacterAliases').attributes('aria-label')).toBe('别名（用逗号分隔，可选）')
    expect(wrapper.get('#continuationAddCharacterDescription').attributes('aria-label')).toBe('角色描述（可选）')
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

  it('links edit and form dialog labels to stable input controls', () => {
    const editCharacter = mount(EditCharacterDialog, {
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
    const addForm = mount(AddFormDialog, {
      global: {
        stubs: dialogStubs,
      },
    })
    const editForm = mount(EditFormDialog, {
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

    expect(editCharacter.findAllComponents(UiField).map(field => field.props('controlId'))).toEqual([
      'continuationEditCharacterName',
      'continuationEditCharacterAliases',
    ])
    expect(editCharacter.get('#continuationEditCharacterName').attributes('aria-label')).toBe('角色名称')
    expect(editCharacter.get('#continuationEditCharacterAliases').attributes('aria-label')).toBe('别名（用逗号分隔）')

    expect(addForm.findAllComponents(UiField).map(field => field.props('controlId'))).toEqual([
      'continuationAddFormName',
      'continuationAddFormDescription',
    ])
    expect(addForm.get('#continuationAddFormName').attributes('aria-label')).toBe('形态名称')
    expect(addForm.get('#continuationAddFormDescription').attributes('aria-label')).toBe('形态描述（可选）')

    expect(editForm.findAllComponents(UiField).map(field => field.props('controlId'))).toEqual([
      'continuationEditFormName',
      'continuationEditFormDescription',
    ])
    expect(editForm.get('#continuationEditFormName').attributes('aria-label')).toBe('形态名称')
    expect(editForm.get('#continuationEditFormDescription').attributes('aria-label')).toBe('形态描述')
  })

  it('uses the shared panel textarea contract for dialog description fields', () => {
    const addCharacter = mount(AddCharacterDialog, {
      global: {
        stubs: dialogStubs,
      },
    })
    const addForm = mount(AddFormDialog, {
      global: {
        stubs: dialogStubs,
      },
    })
    const editForm = mount(EditFormDialog, {
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

    expect(addCharacter.getComponent(UiTextarea).props('variant')).toBe('panel')
    expect(addForm.getComponent(UiTextarea).props('variant')).toBe('panel')
    expect(editForm.getComponent(UiTextarea).props('variant')).toBe('panel')
  })

  it('exposes an explicit name for orthographic source image upload', () => {
    const wrapper = mount(OrthographicDialog, {
      props: {
        characterName: 'Saber',
        formId: 'form_1',
        formName: '常服',
        bookId: 'book-1',
        isGenerating: false,
        resultImagePath: null,
      },
      global: {
        stubs: dialogStubs,
      },
    })

    expect(wrapper.find('input[type="file"]').attributes('aria-label')).toBe('上传 Saber 常服 三视图源图')
  })
})
