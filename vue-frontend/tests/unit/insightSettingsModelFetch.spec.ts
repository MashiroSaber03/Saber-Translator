import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import type { Component } from 'vue'

import VlmSettingsTab from '@/components/insight/settings/VlmSettingsTab.vue'
import LlmSettingsTab from '@/components/insight/settings/LlmSettingsTab.vue'
import EmbeddingSettingsTab from '@/components/insight/settings/EmbeddingSettingsTab.vue'
import RerankerSettingsTab from '@/components/insight/settings/RerankerSettingsTab.vue'
import ImageGenSettingsTab from '@/components/insight/settings/ImageGenSettingsTab.vue'
import InsightModelProviderSection from '@/components/insight/settings/InsightModelProviderSection.vue'
import InsightSettingsPanel from '@/components/insight/settings/InsightSettingsPanel.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useInsightStore } from '@/stores/insightStore'

const { fetchModelsMock } = vi.hoisted(() => ({
  fetchModelsMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  fetchModels: fetchModelsMock,
  testVlmConnection: vi.fn(),
  testLlmConnection: vi.fn(),
  testEmbeddingConnection: vi.fn(),
  testRerankerConnection: vi.fn(),
}))

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

function latestConfig<T>(wrapper: ReturnType<typeof mount>): T {
  const latestEvent = wrapper.emitted('update:config')?.at(-1)
  if (!latestEvent) throw new Error('Missing update:config event')
  return latestEvent[0] as T
}

type SettingsFetchCase = {
  name: string
  component: Component
  switchProvider: string
  configureStore: (store: ReturnType<typeof useInsightStore>) => void
  expectedInitialProvider: string
  expectedCredentialId: string
  expectedProviderInputId: string
  expectedModelInputId: string
  expectedBaseUrlInputId: string
  expectedModelPlaceholder: string
  expectedNumberFields: Array<{
    inputId: string
    min: number
    max: number
    step?: number
  }>
}

const settingsFetchCases: SettingsFetchCase[] = [
  {
    name: 'VLM',
    component: VlmSettingsTab,
    switchProvider: 'ollama',
    expectedInitialProvider: 'gemini',
    expectedCredentialId: 'insight-vlm-api-key',
    expectedProviderInputId: 'insight-vlm-provider',
    expectedModelInputId: 'insight-vlm-model',
    expectedBaseUrlInputId: 'insight-vlm-base-url',
    expectedModelPlaceholder: '例如: gemini-2.0-flash',
    expectedNumberFields: [
      { inputId: 'insight-vlm-rpm-limit', min: 1, max: 100 },
      { inputId: 'insight-vlm-transport-retries', min: 0, max: 10 },
      { inputId: 'insight-vlm-business-retries', min: 0, max: 10 },
      { inputId: 'insight-vlm-temperature', min: 0, max: 1, step: 0.1 },
      { inputId: 'insight-vlm-image-max-size', min: 0, max: 4096, step: 128 },
    ],
    configureStore: (store) => {
      store.updateVlmConfig({
        provider: 'gemini',
        apiKey: 'model-key',
        model: 'gemini-2.0-flash',
        baseUrl: '',
      })
    },
  },
  {
    name: 'LLM',
    component: LlmSettingsTab,
    switchProvider: 'ollama',
    expectedInitialProvider: 'gemini',
    expectedCredentialId: 'insight-llm-api-key',
    expectedProviderInputId: 'insight-llm-provider',
    expectedModelInputId: 'insight-llm-model',
    expectedBaseUrlInputId: 'insight-llm-base-url',
    expectedModelPlaceholder: '例如: gpt-4o-mini',
    expectedNumberFields: [
      { inputId: 'insight-llm-rpm-limit', min: 0, max: 100 },
      { inputId: 'insight-llm-transport-retries', min: 0, max: 10 },
      { inputId: 'insight-llm-business-retries', min: 0, max: 10 },
    ],
    configureStore: (store) => {
      store.updateLlmConfig({
        provider: 'gemini',
        apiKey: 'model-key',
        model: 'gemini-2.0-flash',
        baseUrl: '',
      })
    },
  },
  {
    name: 'Embedding',
    component: EmbeddingSettingsTab,
    switchProvider: 'ollama',
    expectedInitialProvider: 'openai',
    expectedCredentialId: 'insight-embedding-api-key',
    expectedProviderInputId: 'insight-embedding-provider',
    expectedModelInputId: 'insight-embedding-model',
    expectedBaseUrlInputId: 'insight-embedding-base-url',
    expectedModelPlaceholder: '例如: text-embedding-3-small',
    expectedNumberFields: [
      { inputId: 'insight-embedding-rpm-limit', min: 0, max: 1000 },
      { inputId: 'insight-embedding-transport-retries', min: 0, max: 100 },
      { inputId: 'insight-embedding-business-retries', min: 0, max: 100 },
      { inputId: 'insight-embedding-timeout-seconds', min: 0, max: 3600, step: 1 },
    ],
    configureStore: (store) => {
      store.updateEmbeddingConfig({
        provider: 'openai',
        apiKey: 'model-key',
        model: 'text-embedding-3-small',
        baseUrl: '',
      })
    },
  },
  {
    name: 'Reranker',
    component: RerankerSettingsTab,
    switchProvider: 'jina',
    expectedInitialProvider: 'qwen',
    expectedCredentialId: 'reranker-api-key',
    expectedProviderInputId: 'reranker-provider',
    expectedModelInputId: 'reranker-model',
    expectedBaseUrlInputId: 'reranker-base-url',
    expectedModelPlaceholder: '例如: jina-reranker-v2-base-multilingual',
    expectedNumberFields: [
      { inputId: 'reranker-top-k', min: 1, max: 20 },
      { inputId: 'reranker-transport-retries', min: 0, max: 100 },
      { inputId: 'reranker-business-retries', min: 0, max: 100 },
      { inputId: 'reranker-timeout-seconds', min: 0, max: 3600, step: 1 },
    ],
    configureStore: (store) => {
      store.updateRerankerConfig({
        provider: 'qwen',
        apiKey: 'model-key',
        model: 'qwen-rerank',
        baseUrl: '',
      })
    },
  },
]

const settingsFormIdCases = [
  ...settingsFetchCases,
  {
    name: 'ImageGen',
    component: ImageGenSettingsTab,
    configureStore: (store: ReturnType<typeof useInsightStore>) => {
      store.updateImageGenConfig({
        provider: 'gpt2api',
        apiKey: 'model-key',
        model: 'gpt-image-2',
        baseUrl: 'http://127.0.0.1:17200',
      })
    },
    expectedProviderInputId: 'insight-imagegen-provider',
    expectedModelInputId: 'insight-imagegen-model',
    expectedBaseUrlInputId: 'insight-imagegen-base-url',
  },
]

describe('Insight settings model fetch ownership', () => {
  beforeEach(() => {
    localStorage.clear()
    fetchModelsMock.mockReset()
  })

  it.each(settingsFetchCases)('ignores stale $name model-list responses after provider changes', async (testCase) => {
    const deferred = createDeferred({
      success: true,
      models: [{ id: 'previous-provider-model', name: 'Previous Provider Model' }],
    })
    fetchModelsMock.mockReturnValueOnce(deferred.promise)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    wrapper.getComponent(UiModelPicker).vm.$emit('fetch')
    await flushPromises()
    expect(fetchModelsMock).toHaveBeenCalledWith(testCase.expectedInitialProvider, 'model-key', undefined)

    const providerSelect = wrapper.findComponent(UiSelect)
    providerSelect.vm.$emit('update:modelValue', testCase.switchProvider)
    providerSelect.vm.$emit('change', testCase.switchProvider)

    deferred.resolve({
      success: true,
      models: [{ id: 'previous-provider-model', name: 'Previous Provider Model' }],
    })
    await flushPromises()

    const emittedMessages = wrapper.emitted('showMessage') ?? []
    expect(emittedMessages).not.toContainEqual(['获取到 1 个模型', 'success'])
    expect(wrapper.text()).not.toContain('Previous Provider Model')
  })

  it.each(settingsFetchCases)('renders fetched $name models through the shared model picker', async (testCase) => {
    fetchModelsMock.mockResolvedValueOnce({
      success: true,
      models: [{ id: 'shared-model', name: 'Shared Model' }],
    })

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    const modelPicker = wrapper.getComponent(UiModelPicker)
    await modelPicker.vm.$emit('fetch')
    await flushPromises()

    const updatedModelPicker = wrapper.getComponent(UiModelPicker)
    expect(updatedModelPicker.props('modelCount')).toBe(1)
    expect(updatedModelPicker.props('options')).toEqual([
      { label: '-- 选择模型 --', value: '' },
      { label: 'Shared Model', value: 'shared-model' },
    ])

    await updatedModelPicker.vm.$emit('change', 'shared-model')
    await flushPromises()
    expect(latestConfig<{ model: string }>(wrapper).model).toBe('shared-model')
  })

  it('uses the current checkbox primitive for LLM boolean options', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(LlmSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiCheckbox)).toHaveLength(2)
  })

  it('uses the current checkbox primitive for VLM boolean options', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(VlmSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiCheckbox)).toHaveLength(2)
  })

  it('derives LLM provider options from chat-capable providers', () => {
    const typesSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/types.ts'),
      'utf8',
    )
    const llmSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/LlmSettingsTab.vue'),
      'utf8',
    )

    expect(typesSource).toContain("export const LLM_PROVIDER_OPTIONS = getProvidersForCapability('chat')")
    expect(llmSource).toContain('LLM_PROVIDER_OPTIONS')
    expect(llmSource).not.toContain('VLM_PROVIDER_OPTIONS')
  })

  it.each(settingsFetchCases)('uses the settings field primitive for $name form layout', (testCase) => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiField).length).toBeGreaterThan(0)
    expect(wrapper.find('.insight-settings-field').exists()).toBe(false)
  })

  it.each(settingsFetchCases)('uses the shared Insight settings shell and model-provider section for $name', (testCase) => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.getComponent(InsightSettingsPanel).props('description')).toEqual(expect.any(String))

    const section = wrapper.getComponent(InsightModelProviderSection)
    expect(section.props('provider')).toBe(testCase.expectedInitialProvider)
    expect(section.props('credentialId')).toBe(testCase.expectedCredentialId)
    expect(section.props('modelInputId')).toBe(testCase.expectedModelInputId)
    expect(section.props('modelPlaceholder')).toBe(testCase.expectedModelPlaceholder)
    expect(section.props('fetchVariant')).toBe('primary')
    expect(section.props('showTest')).toBe(true)
  })

  it.each(settingsFormIdCases)('passes stable shared form ids for $name model-provider settings', (testCase) => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    const section = wrapper.getComponent(InsightModelProviderSection)
    expect(section.props('providerInputId')).toBe(testCase.expectedProviderInputId)
    expect(section.props('modelInputId')).toBe(testCase.expectedModelInputId)
    expect(section.props('baseUrlInputId')).toBe(testCase.expectedBaseUrlInputId)
  })

  it('binds shared Insight model-provider labels to their primitive controls', () => {
    const wrapper = mount(InsightModelProviderSection, {
      props: {
        provider: 'custom',
        providerOptions: [{ label: 'Custom', value: 'custom' }],
        apiKey: 'secret',
        model: 'shared-model',
        baseUrl: 'http://localhost:11434',
        showApiKey: true,
        credentialId: 'shared-api-key',
        providerInputId: 'shared-provider',
        modelInputId: 'shared-model',
        modelPlaceholder: '模型名称',
        showBaseUrl: true,
        baseUrlInputId: 'shared-base-url',
      },
    })

    expect(wrapper.findAllComponents(UiField).map(field => ({
      label: field.props('label'),
      controlId: field.props('controlId'),
    }))).toEqual([
      { label: '服务商', controlId: 'shared-provider' },
      { label: 'API Key', controlId: 'shared-api-key' },
      { label: '模型', controlId: 'shared-model' },
      { label: 'Base URL', controlId: 'shared-base-url' },
    ])
    expect(wrapper.getComponent(UiSelect).attributes('id')).toBe('shared-provider')
    expect(wrapper.getComponent(UiPasswordField).props('inputId')).toBe('shared-api-key')
    expect(wrapper.getComponent(UiModelPicker).props('inputId')).toBe('shared-model')

    const baseUrlInput = wrapper.findAllComponents(UiInput)
      .find(input => input.props('modelValue') === 'http://localhost:11434')
    expect(baseUrlInput?.attributes('id')).toBe('shared-base-url')
  })

  it.each(settingsFetchCases)('uses the shared credential field for $name API keys', async (testCase) => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    const credential = wrapper.getComponent(UiPasswordField)
    expect(credential.props('placeholder')).toBe('输入 API Key')

    credential.vm.$emit('update:modelValue', 'updated-key')

    await flushPromises()
    expect(latestConfig<{ apiKey: string }>(wrapper).apiKey).toBe('updated-key')
  })

  it.each(settingsFetchCases)('uses the shared number-field primitive for $name numeric settings', (testCase) => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
      },
    })

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields.map(field => ({
      inputId: field.props('inputId'),
      min: field.props('min'),
      max: field.props('max'),
      step: field.props('step'),
    }))).toEqual(testCase.expectedNumberFields.map(field => ({
      inputId: field.inputId,
      min: field.min,
      max: field.max,
      step: field.step ?? 1,
    })))
  })

  it('uses emitted drafts instead of exposed instance methods for settings tabs', () => {
    const settingsTabFiles = [
      'src/components/insight/settings/VlmSettingsTab.vue',
      'src/components/insight/settings/LlmSettingsTab.vue',
      'src/components/insight/settings/BatchSettingsTab.vue',
      'src/components/insight/settings/EmbeddingSettingsTab.vue',
      'src/components/insight/settings/RerankerSettingsTab.vue',
      'src/components/insight/settings/ImageGenSettingsTab.vue',
      'src/components/insight/settings/PromptsSettingsTab.vue',
    ]

    for (const filePath of settingsTabFiles) {
      const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
      expect(source, filePath).not.toContain('defineExpose')
    }
  })

  it('keeps settings tab draft helpers named after internal draft ownership', () => {
    const settingsTabFiles = [
      'src/components/insight/settings/VlmSettingsTab.vue',
      'src/components/insight/settings/LlmSettingsTab.vue',
      'src/components/insight/settings/BatchSettingsTab.vue',
      'src/components/insight/settings/EmbeddingSettingsTab.vue',
      'src/components/insight/settings/RerankerSettingsTab.vue',
      'src/components/insight/settings/ImageGenSettingsTab.vue',
      'src/components/insight/settings/PromptsSettingsTab.vue',
    ]

    for (const filePath of settingsTabFiles) {
      const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
      expect(source, filePath).not.toMatch(/\bfunction\s+(getConfig|getCustomPrompts|syncFromStore)\b/)
    }
  })

  it('keeps provider cache switching behind typed store methods instead of nested config writes', () => {
    const settingsTabFiles = [
      'src/components/insight/settings/VlmSettingsTab.vue',
      'src/components/insight/settings/LlmSettingsTab.vue',
      'src/components/insight/settings/EmbeddingSettingsTab.vue',
      'src/components/insight/settings/RerankerSettingsTab.vue',
      'src/components/insight/settings/ImageGenSettingsTab.vue',
    ]

    for (const filePath of settingsTabFiles) {
      const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')
      expect(source, filePath).not.toMatch(/\binsightStore\.config\.(vlm|llm|embedding|reranker|imageGen)\.[A-Za-z0-9_.]+\s=/)
    }
  })

  it('keeps the prompts library header responsive inside narrow settings modals', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/settings/PromptsSettingsTab.vue'),
      'utf8',
    )
    const headerBlock = source.match(/\.prompts-settings-tab__library-header \{(?<body>[\s\S]*?)\n\}/)
      ?.groups?.body ?? ''
    const titleBlock = source.match(/\.prompts-settings-tab__library-title \{(?<body>[\s\S]*?)\n\}/)
      ?.groups?.body ?? ''

    expect(headerBlock).toContain('flex-wrap: wrap')
    expect(headerBlock).toContain('gap:')
    expect(titleBlock).toContain('min-width: 0')
    expect(titleBlock).toContain('overflow-wrap: anywhere')
  })
})
