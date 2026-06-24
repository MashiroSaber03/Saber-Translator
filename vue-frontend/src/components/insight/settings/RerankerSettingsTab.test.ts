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

    await wrapper.get('[data-testid="reranker-api-key"]').setValue('rerank-key')
    await wrapper.get('[data-testid="reranker-model"]').setValue('jina-reranker-v2-base-multilingual')
    await wrapper.get('[data-testid="reranker-top-k"]').setValue('6')
    await wrapper.get('[data-testid="reranker-transport-retries"]').setValue('7')
    await wrapper.get('[data-testid="reranker-business-retries"]').setValue('8')
    await wrapper.get('[data-testid="reranker-timeout-seconds"]').setValue('9')
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
