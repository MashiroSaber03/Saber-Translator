import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ImageGenSettingsTab from './ImageGenSettingsTab.vue'
import InsightModelProviderSection from './InsightModelProviderSection.vue'
import InsightSettingsPanel from './InsightSettingsPanel.vue'
import { IMAGE_GEN_PROVIDER_OPTIONS } from './types'
import { useInsightStore } from '@/stores/insightStore'
import UiField from '@/components/ui/UiField.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

function latestConfig<T>(wrapper: ReturnType<typeof mount>): T {
  const latestEvent = wrapper.emitted('update:config')?.at(-1)
  if (!latestEvent) throw new Error('Missing update:config event')
  return latestEvent[0] as T
}

describe('ImageGenSettingsTab', () => {
  it('renders the image generation provider as a selector', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const providerSelect = wrapper.findComponent(UiSelect)

    expect(providerSelect.exists()).toBe(true)
    expect(providerSelect.props('options')).toEqual(IMAGE_GEN_PROVIDER_OPTIONS)
    expect(latestConfig<{ provider: string }>(wrapper).provider).toBe('gpt2api')
  })

  it('syncs provider changes from the store', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.updateImageGenConfig({
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://gateway.example.com/v1',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.setProps({ syncRequestId: 1 })
    await flushPromises()

    expect(latestConfig(wrapper)).toEqual({
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: 'https://gateway.example.com/v1',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })
  })

  it('uses the shared model picker as a model-name input without fetch controls', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const modelPicker = wrapper.getComponent(UiModelPicker)
    expect(modelPicker.props('showFetch')).toBe(false)
    expect(modelPicker.props('placeholder')).toBe('例如: gpt-image-2')
  })

  it('uses the settings field primitive for form layout', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.findAllComponents(UiField).length).toBeGreaterThan(0)
    expect(wrapper.find('.insight-settings-field').exists()).toBe(false)
  })

  it('uses the shared Insight settings shell and model-provider section', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.getComponent(InsightSettingsPanel).props('description')).toContain('生图模型')

    const section = wrapper.getComponent(InsightModelProviderSection)
    expect(section.props('provider')).toBe('gpt2api')
    expect(section.props('credentialId')).toBe('insight-imagegen-api-key')
    expect(section.props('modelPlaceholder')).toBe('例如: gpt-image-2')
    expect(section.props('showFetch')).toBe(false)
    expect(section.props('showTest')).toBe(false)
  })

  it('uses the shared credential field for image generation API keys', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.updateImageGenConfig({
      provider: 'gpt2api',
      apiKey: 'image-key',
      model: 'gpt-image-2',
      baseUrl: '',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const credential = wrapper.getComponent(UiPasswordField)
    expect(credential.props('placeholder')).toBe('输入 API Key')

    credential.vm.$emit('update:modelValue', 'updated-image-key')
    await flushPromises()

    expect(latestConfig<{ apiKey: string }>(wrapper).apiKey).toBe('updated-image-key')
  })

  it('uses the shared number-field primitive for runtime retry settings', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(ImageGenSettingsTab, {
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
    }))).toEqual([
      { inputId: 'insight-imagegen-transport-retries', min: 0, step: 1 },
      { inputId: 'insight-imagegen-business-retries', min: 0, step: 1 },
      { inputId: 'insight-imagegen-timeout-seconds', min: 0, max: undefined, step: 0.1 },
    ])
  })

  it('shows a non-blocking warning when newapi is selected without a model', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.updateImageGenConfig({
      provider: 'newapi',
      apiKey: 'image-key',
      model: '',
      baseUrl: 'https://newapi.example.com/v1',
      transportRetries: 5,
      businessRetries: 6,
      timeoutSeconds: 7,
    })

    const wrapper = mount(ImageGenSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.setProps({ syncRequestId: 1 })
    await flushPromises()

    expect(wrapper.text()).toContain('当前服务商需要手动填写模型名')
    expect(latestConfig<{ model: string }>(wrapper).model).toBe('')
  })
})

