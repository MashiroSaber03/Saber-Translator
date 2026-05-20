/* eslint-disable vue/one-component-per-file */

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'

vi.mock('@/api/config', () => ({
  getFontList: async () => ({ fonts: [] }),
  uploadFont: async () => ({ success: true }),
  getTranslateWorkflowPreferences: async () => ({
    success: true,
    preferences: {
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'translate-current',
    },
  }),
  saveTranslateWorkflowPreferences: async () => ({ success: true }),
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: [String, Number] as PropType<string | number | undefined>,
        default: undefined,
      },
      options: {
        type: Array as PropType<Array<{ label: string; value: string | number }>>,
        default: () => [],
      },
    },
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        (props.options || []).map((option: any) => h('option', { value: option.value }, option.label))
      )
    },
  }),
}))

vi.mock('@/components/common/CollapsiblePanel.vue', () => ({
  default: defineComponent({
    props: {
      title: {
        type: String,
        default: '',
      },
    },
    setup(props, { slots }) {
      return () => h('section', [h('h3', props.title), slots.default?.()])
    },
  }),
}))

vi.mock('@/components/translate/PageSelectionModal.vue', () => ({
  default: defineComponent({
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
      selectedPages: {
        type: Array as PropType<number[]>,
        default: () => [],
      },
    },
    emits: ['update:modelValue', 'confirm'],
    setup(props, { emit }) {
      return () => props.modelValue
        ? h('div', { class: 'page-selection-modal-stub' }, [
            h('button', {
              class: 'confirm-selection',
              onClick: () => emit('confirm', [1, 3, 8, 10]),
            }, 'confirm'),
            h('button', {
              class: 'close-selection',
              onClick: () => emit('update:modelValue', false),
            }, 'close'),
          ])
        : null
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'
import { useImageStore } from '@/stores/imageStore'

describe('SettingsSidebar page selection workflow', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const imageStore = useImageStore()
    imageStore.clearImages()
    imageStore.addImage('001.png', 'data:image/png;base64,aaa')
    imageStore.addImage('002.png', 'data:image/png;base64,bbb')
    imageStore.addImage('003.png', 'data:image/png;base64,ccc')
    imageStore.addImage('004.png', 'data:image/png;base64,ddd')
    imageStore.addImage('005.png', 'data:image/png;base64,eee')
    imageStore.addImage('006.png', 'data:image/png;base64,fff')
    imageStore.addImage('007.png', 'data:image/png;base64,ggg')
    imageStore.addImage('008.png', 'data:image/png;base64,hhh')
    imageStore.addImage('009.png', 'data:image/png;base64,iii')
    imageStore.addImage('010.png', 'data:image/png;base64,jjj')
  })

  it('opens page selection modal and emits selected pages for batch workflow', async () => {
    const wrapper = mount(SettingsSidebar)

    const selects = wrapper.findAll('select')
    const workflowModeSelect = selects[selects.length - 1]
    expect(workflowModeSelect).toBeTruthy()
    await workflowModeSelect!.setValue('translate-batch')
    await workflowModeSelect!.trigger('change')

    const enableCheckbox = wrapper.find('.page-selection-toggle-compact input[type="checkbox"]')
    await enableCheckbox.setValue(true)

    await wrapper.find('.page-selection-open-btn').trigger('click')
    expect(wrapper.find('.page-selection-modal-stub').exists()).toBe(true)

    await wrapper.find('.confirm-selection').trigger('click')

    const runButton = wrapper.find('#runWorkflowButton')
    await runButton.trigger('click')

    expect(wrapper.emitted('runWorkflow')?.[0]).toEqual([
      {
        mode: 'translate-batch',
        pageSelection: {
          pages: [1, 3, 8, 10],
        },
      },
    ])
  })
})
