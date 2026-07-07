import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import RerankerSettingsTab from './RerankerSettingsTab.vue'
import { useInsightStore } from '@/stores/insightStore'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { RERANKER_PROVIDER_OPTIONS } from './types'

function latestConfig<T>(wrapper: ReturnType<typeof mount>): T {
  const latestEvent = wrapper.emitted('update:config')?.at(-1)
  if (!latestEvent) throw new Error('Missing update:config event')
  return latestEvent[0] as T
}

const { testRerankerConnection } = vi.hoisted(() => ({
  testRerankerConnection: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  testRerankerConnection,
  fetchModels: vi.fn(),
}))

describe('RerankerSettingsTab', () => {
  beforeEach(() => {
    localStorage.clear()
    testRerankerConnection.mockReset()
    testRerankerConnection.mockResolvedValue({ success: true })
  })

  it('syncs runtime retry settings from the store', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.updateRerankerConfig({
      provider: 'jina',
      apiKey: 'rerank-key',
      model: 'jina-reranker-v2-base-multilingual',
      baseUrl: 'https://rerank.example.com/v1',
      topK: 6,
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })

    const wrapper = mount(RerankerSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.setProps({ syncRequestId: 1 })
    await flushPromises()

    expect(latestConfig(wrapper)).toEqual({
      provider: 'jina',
      apiKey: 'rerank-key',
      model: 'jina-reranker-v2-base-multilingual',
      baseUrl: 'https://rerank.example.com/v1',
      topK: 6,
      transportRetries: 7,
      businessRetries: 8,
      timeoutSeconds: 9,
    })
  })

  it('uses the shared select primitive for the fixed reranker provider list', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(RerankerSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const selects = wrapper.findAllComponents(UiSelect)
    expect(selects).toHaveLength(1)
    expect(selects[0]!.props('options')).toEqual(RERANKER_PROVIDER_OPTIONS)
  })

  it('uses the shared model picker for reranker model fetch controls', () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(RerankerSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    const modelPicker = wrapper.getComponent(UiModelPicker)
    expect(modelPicker.props('inputId')).toBe('reranker-model')
    expect(modelPicker.props('placeholder')).toBe('例如: jina-reranker-v2-base-multilingual')
  })

  it('passes retry and timeout settings to the reranker connection test API', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(RerankerSettingsTab, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.get('#reranker-api-key').setValue('rerank-key')
    await wrapper.get('input#reranker-model').setValue('jina-reranker-v2-base-multilingual')
    await wrapper.get('#reranker-top-k').setValue('6')
    await wrapper.get('#reranker-transport-retries').setValue('7')
    await wrapper.get('#reranker-business-retries').setValue('8')
    await wrapper.get('#reranker-timeout-seconds').setValue('9')
    const testButton = wrapper.findAll('button').find(button => button.text().includes('测试连接'))
    if (!testButton) throw new Error('Missing reranker connection test button')
    await testButton.trigger('click')

    expect(testRerankerConnection).toHaveBeenCalledWith({
      provider: 'jina',
      api_key: 'rerank-key',
      model: 'jina-reranker-v2-base-multilingual',
      base_url: undefined,
      transport_retries: 7,
      business_retries: 8,
      timeout_seconds: 9,
    })
  })
})
