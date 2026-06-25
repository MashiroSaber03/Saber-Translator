import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, type PropType } from 'vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

const { getFontListMock, getPreferencesMock, savePreferencesMock } = vi.hoisted(() => ({
  getFontListMock: vi.fn(),
  getPreferencesMock: vi.fn(),
  savePreferencesMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getFontList: getFontListMock,
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
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    setActivePinia(createPinia())
    getFontListMock.mockReset()
    getPreferencesMock.mockReset()
    savePreferencesMock.mockReset()
    getFontListMock.mockResolvedValue({ fonts: [] })
    getPreferencesMock.mockResolvedValue({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: false,
        lastWorkflowMode: 'translate-current',
      },
    })
    savePreferencesMock.mockResolvedValue({ success: true })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps the default workflow mode when remembering is disabled', async () => {
    const wrapper = mount(SettingsSidebar)
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('translate-current')
    expect(wrapper.findComponent(UiCheckbox).props('modelValue')).toBe(false)
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
    expect(wrapper.findComponent(UiCheckbox).props('modelValue')).toBe(true)
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

    wrapper.findComponent(UiCheckbox).vm.$emit('change', true)
    await flushPromises()

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

    wrapper.findComponent(UiCheckbox).vm.$emit('change', true)
    resolvePreferences({
      success: true,
      preferences: {
        rememberWorkflowModeEnabled: true,
        lastWorkflowMode: 'clear-all',
      },
    })
    await flushPromises()

    expect((wrapper.find('#workflowModeSelect').element as HTMLSelectElement).value).toBe('translate-current')
    expect(wrapper.findComponent(UiCheckbox).props('modelValue')).toBe(true)
  })

  it('registers outside-click handling before async font loading settles', () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')
    getFontListMock.mockReturnValue(new Promise(() => {}))

    const wrapper = mount(SettingsSidebar)

    expect(addEventListenerSpy).toHaveBeenCalledWith('click', expect.any(Function))

    wrapper.unmount()
    expect(removeEventListenerSpy).toHaveBeenCalledWith('click', expect.any(Function))

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
  })

  it('updates auto font size without routine console noise', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    try {
      const wrapper = mount(SettingsSidebar)

      await wrapper.find('.auto-fontsize-toggle').trigger('click')

      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }
  })
})
