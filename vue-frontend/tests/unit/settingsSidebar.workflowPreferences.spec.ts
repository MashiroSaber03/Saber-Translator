/* eslint-disable vue/one-component-per-file */

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'

const { getPreferencesMock, savePreferencesMock } = vi.hoisted(() => ({
  getPreferencesMock: vi.fn(),
  savePreferencesMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getFontList: async () => ({ fonts: [] }),
  uploadFont: async () => ({ success: true }),
  getTranslateWorkflowPreferences: getPreferencesMock,
  saveTranslateWorkflowPreferences: savePreferencesMock,
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
    setup(props, { attrs, emit }) {
      return () => h(
        'select',
        {
          ...attrs,
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
    setup() {
      return () => null
    },
  }),
}))

import SettingsSidebar from '@/components/translate/SettingsSidebar.vue'

describe('SettingsSidebar workflow preferences', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    getPreferencesMock.mockReset()
    savePreferencesMock.mockReset()
    getPreferencesMock.mockResolvedValue({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: false,
        lastWorkflowMode: 'translate-current',
      },
    })
    savePreferencesMock.mockResolvedValue({ success: true })
  })

  it('keeps the default workflow mode when remembering is disabled', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('translate-current')
    expect((wrapper.find('#rememberWorkflowModeCheckbox').element as HTMLInputElement).checked).toBe(false)
  })

  it('restores a remembered dangerous workflow mode', async () => {
    getPreferencesMock.mockResolvedValue({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })

    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('clear-all')
    expect((wrapper.find('#rememberWorkflowModeCheckbox').element as HTMLInputElement).checked).toBe(true)
  })

  it('saves the last workflow mode immediately when the dropdown changes', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    await wrapper.find('#workflowModeSelect').setValue('hq-batch')

    expect(savePreferencesMock).toHaveBeenCalledWith({
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'hq-batch',
    })
  })

  it('saves the remember switch immediately with the current workflow mode', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    await wrapper.find('#rememberWorkflowModeCheckbox').setValue(true)

    expect(savePreferencesMock).toHaveBeenCalledWith({
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'translate-current',
    })
  })

  it('keeps the selected workflow mode in the UI even when saving fails', async () => {
    savePreferencesMock.mockRejectedValueOnce(new Error('network down'))
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const wrapper = mount(SettingsSidebar)
      await flushPromises()

      await wrapper.find('#workflowModeSelect').setValue('hq-batch')
      await flushPromises()

      expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('hq-batch')
    } finally {
      warnSpy.mockRestore()
    }
  })

  it('serializes rapid workflow preference saves so the latest value wins', async () => {
    let resolveFirstSave!: () => void
    savePreferencesMock
      .mockImplementationOnce(() => new Promise(resolve => {
        resolveFirstSave = () => resolve({ success: true })
      }))
      .mockResolvedValueOnce({ success: true })

    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    await wrapper.find('#workflowModeSelect').setValue('hq-batch')
    await wrapper.find('#workflowModeSelect').setValue('clear-all')

    expect(savePreferencesMock).toHaveBeenCalledTimes(1)
    expect(savePreferencesMock).toHaveBeenNthCalledWith(1, {
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'hq-batch',
    })

    resolveFirstSave()
    await flushPromises()

    expect(savePreferencesMock).toHaveBeenCalledTimes(2)
    expect(savePreferencesMock).toHaveBeenNthCalledWith(2, {
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'clear-all',
    })
  })

  it('does not let late preference loading overwrite a manual mode change', async () => {
    let resolvePreferences!: (value: unknown) => void
    getPreferencesMock.mockReturnValue(new Promise(resolve => {
      resolvePreferences = resolve
    }))

    const wrapper = mount(SettingsSidebar)

    await wrapper.find('#workflowModeSelect').setValue('proofread-batch')
    resolvePreferences({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('proofread-batch')
  })

  it('does not let late preference loading overwrite after the remember switch changes', async () => {
    let resolvePreferences!: (value: unknown) => void
    getPreferencesMock.mockReturnValue(new Promise(resolve => {
      resolvePreferences = resolve
    }))

    const wrapper = mount(SettingsSidebar)

    await wrapper.find('#rememberWorkflowModeCheckbox').setValue(true)
    resolvePreferences({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('translate-current')
    expect((wrapper.find('#rememberWorkflowModeCheckbox').element as HTMLInputElement).checked).toBe(true)
  })
})
