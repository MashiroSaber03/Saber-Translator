import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import RerankerSettingsTab from './RerankerSettingsTab.vue'
import { useInsightStore } from '@/stores/insightStore'

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

  it('syncs runtime retry settings from the store', () => {
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
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            props: ['modelValue', 'options'],
            template: '<div class="custom-select-stub" />',
          },
        },
      },
    })

    wrapper.vm.syncFromStore()

    expect(wrapper.vm.getConfig()).toEqual({
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

  it('passes retry and timeout settings to the reranker connection test API', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const wrapper = mount(RerankerSettingsTab, {
      global: {
        plugins: [pinia],
        stubs: {
          CustomSelect: {
            name: 'CustomSelect',
            props: ['modelValue', 'options'],
            template: '<div class="custom-select-stub" />',
          },
        },
      },
    })

    const inputs = wrapper.findAll('input')
    await inputs[0].setValue('rerank-key')
    await inputs[1].setValue('jina-reranker-v2-base-multilingual')
    await inputs[2].setValue('6')
    await inputs[3].setValue('7')
    await inputs[4].setValue('8')
    await inputs[5].setValue('9')
    const buttons = wrapper.findAll('button.btn-secondary')
    await buttons[buttons.length - 1].trigger('click')

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
