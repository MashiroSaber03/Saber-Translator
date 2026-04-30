import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

const {
  fetchModelsMock,
  testAiTranslateConnectionMock,
  testOllamaConnectionMock,
  testSakuraConnectionMock,
} = vi.hoisted(() => ({
  fetchModelsMock: vi.fn(),
  testAiTranslateConnectionMock: vi.fn(),
  testOllamaConnectionMock: vi.fn(),
  testSakuraConnectionMock: vi.fn(),
}))

vi.mock('@/api/config', async () => {
  const actual = await vi.importActual<typeof import('@/api/config')>('@/api/config')
  return {
    ...actual,
    configApi: {
      ...actual.configApi,
      fetchModels: fetchModelsMock,
      testAiTranslateConnection: testAiTranslateConnectionMock,
      testOllamaConnection: testOllamaConnectionMock,
      testSakuraConnection: testSakuraConnectionMock,
    },
  }
})

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  }),
}))

vi.mock('@/components/common/CustomSelect.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'options'],
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          class: 'custom-select-stub',
          value: props.modelValue,
          onChange: (event: Event) => emit('change', (event.target as HTMLSelectElement).value),
        },
        (props.options || []).map((option: { label: string; value: string }) =>
          h('option', { value: option.value }, option.label)
        )
      )
    },
  }),
}))

vi.mock('@/components/settings/SavedPromptsPicker.vue', () => ({
  default: defineComponent({
    setup() {
      return () => h('div', 'SavedPromptsPicker stub')
    },
  }),
}))

import TranslationSettings from '@/components/settings/TranslationSettings.vue'
import { useSettingsStore } from '@/stores/settingsStore'

describe('TranslationSettings', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()

    fetchModelsMock.mockReset()
    testAiTranslateConnectionMock.mockReset()
    testOllamaConnectionMock.mockReset()
    testSakuraConnectionMock.mockReset()

    fetchModelsMock.mockResolvedValue({
      success: true,
      models: [{ id: 'llama3.2', name: 'llama3.2' }],
    })
    testAiTranslateConnectionMock.mockResolvedValue({
      success: true,
      message: '连接成功',
    })
  })

  it('uses the shared model-fetch endpoint for ollama local model discovery', async () => {
    const store = useSettingsStore()
    store.settings.translation.provider = 'ollama'
    store.settings.translation.modelName = 'llama3.2'

    const wrapper = mount(TranslationSettings)
    const fetchButtons = wrapper.findAll('.fetch-models-btn')

    expect(fetchButtons.length).toBeGreaterThan(0)
    await fetchButtons[fetchButtons.length - 1]!.trigger('click')
    await flushPromises()

    expect(fetchModelsMock).toHaveBeenCalledWith('ollama', '', '')
    expect(testOllamaConnectionMock).not.toHaveBeenCalled()
    const optionTexts = wrapper.findAll('.custom-select-stub option').map(option => option.text())
    expect(optionTexts).toContain('llama3.2')
    expect(optionTexts).not.toContain('[object Object]')
  })

  it('uses the shared AI translate connection test for ollama local providers', async () => {
    const store = useSettingsStore()
    store.settings.translation.provider = 'ollama'
    store.settings.translation.modelName = 'llama3.2'

    const wrapper = mount(TranslationSettings)
    const testButton = wrapper.find('.settings-test-btn')

    await testButton.trigger('click')
    await flushPromises()

    expect(testAiTranslateConnectionMock).toHaveBeenCalledWith({
      provider: 'ollama',
      apiKey: '',
      modelName: 'llama3.2',
      baseUrl: '',
    })
    expect(testOllamaConnectionMock).not.toHaveBeenCalled()
  })
})
