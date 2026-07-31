import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import UiField from '@/components/ui/UiField.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

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

vi.mock('@/api/v2/diagnostics', () => ({
  fetchModels: fetchModelsMock,
  testAiTranslateConnection: testAiTranslateConnectionMock,
  testOllamaConnection: testOllamaConnectionMock,
  testSakuraConnection: testSakuraConnectionMock,
  testBaiduTranslateConnection: vi.fn(),
  testYoudaoTranslateConnection: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
  }),
}))

vi.mock('@/components/ui/UiCombobox.vue', () => ({
  default: defineComponent({
    props: ['modelValue', 'options'],
    emits: ['change'],
    setup(props, { emit }) {
      return () => h(
        'select',
        {
          class: 'ui-combobox-stub',
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
import { useSettingsStore } from '@/stores/settings'

const componentSourcePath = resolve(process.cwd(), 'src/components/settings/TranslationSettings.vue')

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

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
    const modelPickers = wrapper.findAllComponents(UiModelPicker)

    expect(modelPickers.length).toBeGreaterThan(0)
    modelPickers[modelPickers.length - 1]!.vm.$emit('fetch')
    await flushPromises()

    expect(fetchModelsMock).toHaveBeenCalledWith('ollama', '', '')
    expect(testOllamaConnectionMock).not.toHaveBeenCalled()
    const optionTexts = wrapper.findAll('.ui-combobox-stub option').map(option => option.text())
    expect(optionTexts).toContain('llama3.2')
    expect(optionTexts).not.toContain('[object Object]')
  })

  it('uses the shared AI translate connection test for ollama local providers', async () => {
    const store = useSettingsStore()
    store.settings.translation.provider = 'ollama'
    store.settings.translation.modelName = 'llama3.2'

    const wrapper = mount(TranslationSettings)
    const testButton = wrapper.findAll('button').find(button => button.text().includes('测试连接'))
    expect(testButton).toBeTruthy()

    await testButton!.trigger('click')
    await flushPromises()

    expect(testAiTranslateConnectionMock).toHaveBeenCalledWith({
      provider: 'ollama',
      apiKey: '',
      modelName: 'llama3.2',
      baseUrl: '',
    })
    expect(testOllamaConnectionMock).not.toHaveBeenCalled()
  })

  it('switches translation mode without routine console output', async () => {
    const store = useSettingsStore()
    store.settings.translation.translationMode = 'batch'

    const wrapper = mount(TranslationSettings)
    const modeSelect = wrapper.findAllComponents(UiSelect)
      .find(select =>
        (select.props('options') || [])
          .some((option: { value: string | number }) => option.value === 'single')
      )
    expect(modeSelect).toBeTruthy()

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    try {
      modeSelect!.vm.$emit('update:modelValue', 'single')
      modeSelect!.vm.$emit('change', 'single')
      await flushPromises()
      expect(logSpy).not.toHaveBeenCalled()
    } finally {
      logSpy.mockRestore()
    }

    expect(store.settings.translation.translationMode).toBe('single')
  })

  it('restores prompt mode, translation mode, and prompt when switching providers', async () => {
    const store = useSettingsStore()
    const cachedOptions = JSON.parse(
      JSON.stringify(store.settings.translation.openaiOptions),
    ) as typeof store.settings.translation.openaiOptions
    cachedOptions.request.forceJsonOutput = true
    store.settings.translation.singleJsonPrompt = 'cached single JSON prompt'
    store.providerConfigs.translation.deepseek = {
      apiKey: '',
      modelName: 'deepseek-chat',
      customBaseUrl: '',
      openaiOptions: cachedOptions,
      translationMode: 'single',
    }

    const wrapper = mount(TranslationSettings)
    const providerSelect = wrapper.findAllComponents(UiSelect)
      .find(select =>
        (select.props('options') || [])
          .some((option: { value: string | number }) => option.value === 'deepseek')
      )
    expect(providerSelect).toBeTruthy()

    providerSelect!.vm.$emit('change', 'deepseek')
    await flushPromises()

    expect(store.settings.translation.provider).toBe('deepseek')
    expect(store.settings.translation.translationMode).toBe('single')
    expect(store.settings.translation.openaiOptions.request.forceJsonOutput).toBe(true)
    expect(wrapper.get<HTMLTextAreaElement>('#settingsPromptContent').element.value)
      .toBe('cached single JSON prompt')
    const selectedValues = wrapper.findAllComponents(UiSelect)
      .map(select => select.props('modelValue'))
    expect(selectedValues).toContain('single')
    expect(selectedValues).toContain('json')
  })

  it('uses fixed select primitives for provider and prompt mode fields', () => {
    const wrapper = mount(TranslationSettings)

    const selects = wrapper.findAllComponents(UiSelect)
    const optionValues = selects.map(select =>
      (select.props('options') || []).map((option: { value: string | number }) => option.value)
    )

    expect(optionValues).toContainEqual(expect.arrayContaining(['siliconflow', 'custom']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['batch', 'single']))
    expect(optionValues).toContainEqual(expect.arrayContaining(['normal', 'json']))
  })

  it('routes labels and hints through typed UiField props', () => {
    const source = readFileSync(componentSourcePath, 'utf8')
    const wrapper = mount(TranslationSettings)
    const fields = wrapper.findAllComponents(UiField)
    const fieldByLabel = (label: string) => fields.find(field => field.props('label') === label)
    const fieldByControlId = (controlId: string) =>
      fields.find(field => field.props('controlId') === controlId)

    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('class="ui-form-hint"')

    expect(fieldByLabel('翻译服务商')?.props('controlId')).toBe('settingsModelProvider')
    expect(fieldByControlId('settingsApiKey')?.props('label')).toBeTruthy()
    expect(fieldByLabel('Base URL')?.props('controlId')).toBe('settingsCustomBaseUrl')
    expect(fieldByControlId('settingsModelName')?.props('label')).toBeTruthy()
    expect(fieldByControlId('settingsLocalModelName')?.props('label')).toBe('模型名称')
    expect(fieldByLabel('RPM限制')?.props('hint')).toBe('每分钟请求数，0表示无限制')
    expect(fieldByLabel('重试次数')?.props('hint')).toBe('业务重试：空结果/结构解析失败')
    expect(fieldByLabel('传输重试')?.props('hint')).toBe('网络超时/429/5xx')
    expect(fieldByLabel('翻译模式')?.props('hint')).toContain('整页批量翻译')
    expect(fieldByLabel('翻译提示词')?.props('controlId')).toBe('settingsPromptContent')
    expect(fieldByLabel('文本框提示词')?.props('controlId')).toBe('settingsTextboxPromptContent')
  })

  it('ignores stale model-list responses after provider changes', async () => {
    const store = useSettingsStore()
    store.settings.translation.provider = 'siliconflow'
    store.settings.translation.apiKey = 'model-key'
    const pendingModels = deferred<{ success: boolean; models: Array<{ id: string; name: string }> }>()
    fetchModelsMock.mockReturnValueOnce(pendingModels.promise)

    const wrapper = mount(TranslationSettings)
    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')

    const providerSelect = wrapper.findAllComponents(UiSelect)
      .find(select =>
        (select.props('options') || [])
          .some((option: { value: string | number }) => option.value === 'deepseek')
      )
    expect(providerSelect).toBeTruthy()
    providerSelect!.vm.$emit('update:modelValue', 'deepseek')
    providerSelect!.vm.$emit('change', 'deepseek')
    await flushPromises()

    pendingModels.resolve({
      success: true,
      models: [{ id: 'stale-model', name: 'stale-model' }],
    })
    await flushPromises()

    const optionTexts = wrapper.findAll('.ui-combobox-stub option').map(option => option.text())
    expect(optionTexts).not.toContain('stale-model')
  })
})
