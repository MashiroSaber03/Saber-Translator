import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h, reactive } from 'vue'

const {
  updateTranslationServiceMock,
  updateAiVisionOcrMock,
  toastInfoMock,
  getPromptsMock,
} = vi.hoisted(() => ({
  updateTranslationServiceMock: vi.fn(),
  updateAiVisionOcrMock: vi.fn(),
  toastInfoMock: vi.fn(),
  getPromptsMock: vi.fn(),
}))

const settingsState = reactive({
  translation: {
    isJsonMode: false,
  },
  aiVisionOcr: {
    isJsonMode: false,
    promptMode: 'paddleocr_vl' as 'normal' | 'json' | 'paddleocr_vl',
  },
})

vi.mock('@/stores/settingsStore', () => ({
  useSettingsStore: () => ({
    settings: settingsState,
    updateTranslationService: updateTranslationServiceMock,
    updateAiVisionOcr: updateAiVisionOcrMock,
  }),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: toastInfoMock,
  }),
}))

vi.mock('@/api/config', () => ({
  configApi: {
    getPrompts: getPromptsMock,
    getPromptContent: vi.fn(),
    savePrompt: vi.fn(),
    deletePrompt: vi.fn(),
    getTextboxPrompts: vi.fn(),
    getTextboxPromptContent: vi.fn(),
    saveTextboxPrompt: vi.fn(),
    deleteTextboxPrompt: vi.fn(),
  },
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    name: 'CustomSelectStub',
    props: ['modelValue', 'options'],
    emits: ['update:modelValue', 'change'],
    setup(props, { emit }) {
      return () =>
        h(
          'select',
          {
            value: props.modelValue,
            onChange: (event: Event) => {
              const value = (event.target as HTMLSelectElement).value
              emit('update:modelValue', value)
              emit('change', value)
            },
          },
          (props.options || []).map((option: any) =>
            h('option', { value: option.value }, option.label),
          ),
        )
    },
  }),
}))

import PromptLibrary from '@/components/settings/PromptLibrary.vue'

describe('PromptLibrary', () => {
  beforeEach(() => {
    updateTranslationServiceMock.mockReset()
    updateAiVisionOcrMock.mockReset()
    toastInfoMock.mockReset()
    getPromptsMock.mockReset()
    getPromptsMock.mockResolvedValue({ prompt_names: [] })

    settingsState.translation.isJsonMode = false
    settingsState.aiVisionOcr.isJsonMode = false
    settingsState.aiVisionOcr.promptMode = 'paddleocr_vl'
  })

  it('preserves ai vision paddleocr_vl prompt mode instead of collapsing it to normal/json', async () => {
    const wrapper = mount(PromptLibrary)
    await flushPromises()

    const selects = wrapper.findAll('select')
    expect(selects.length).toBeGreaterThanOrEqual(1)

    const typeSelect = selects[0]
    await typeSelect.setValue('ai_vision_ocr')
    await flushPromises()

    const refreshedSelects = wrapper.findAll('select')
    expect(refreshedSelects.length).toBeGreaterThanOrEqual(2)

    const modeSelect = refreshedSelects[1]
    expect(modeSelect.element.value).toBe('paddleocr_vl')

    await modeSelect.setValue('paddleocr_vl')

    expect(updateAiVisionOcrMock).toHaveBeenLastCalledWith({
      isJsonMode: false,
      promptMode: 'paddleocr_vl',
    })
    expect(toastInfoMock).toHaveBeenCalledWith('已切换到OCR模型提示词模式')
  })
})
