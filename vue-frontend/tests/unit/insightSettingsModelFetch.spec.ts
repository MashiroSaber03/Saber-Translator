import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import type { Component } from 'vue'

import VlmSettingsTab from '@/components/insight/settings/VlmSettingsTab.vue'
import LlmSettingsTab from '@/components/insight/settings/LlmSettingsTab.vue'
import EmbeddingSettingsTab from '@/components/insight/settings/EmbeddingSettingsTab.vue'
import RerankerSettingsTab from '@/components/insight/settings/RerankerSettingsTab.vue'
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

type SettingsFetchCase = {
  name: string
  component: Component
  switchProvider: string
  configureStore: (store: ReturnType<typeof useInsightStore>) => void
  expectedInitialProvider: string
}

const settingsFetchCases: SettingsFetchCase[] = [
  {
    name: 'VLM',
    component: VlmSettingsTab,
    switchProvider: 'ollama',
    expectedInitialProvider: 'gemini',
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

describe('Insight settings model fetch ownership', () => {
  beforeEach(() => {
    localStorage.clear()
    fetchModelsMock.mockReset()
  })

  it.each(settingsFetchCases)('ignores stale $name model-list responses after provider changes', async (testCase) => {
    const deferred = createDeferred({
      success: true,
      models: [{ id: 'old-provider-model', name: 'Old Provider Model' }],
    })
    fetchModelsMock.mockReturnValueOnce(deferred.promise)

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    testCase.configureStore(store)

    const wrapper = mount(testCase.component, {
      global: {
        plugins: [pinia],
        stubs: {
          CustomSelect: {
            props: ['modelValue', 'options'],
            emits: ['update:modelValue', 'change'],
            template: `<button class="provider-stub" type="button" @click="$emit('update:modelValue', '${testCase.switchProvider}'); $emit('change')">switch provider</button>`,
          },
        },
      },
    })

    const fetchButton = wrapper.findAll('button').find(button => button.text().includes('获取模型'))
    expect(fetchButton).toBeTruthy()

    await fetchButton!.trigger('click')
    expect(fetchModelsMock).toHaveBeenCalledWith(testCase.expectedInitialProvider, 'model-key', undefined)

    await wrapper.find('.provider-stub').trigger('click')
    deferred.resolve({
      success: true,
      models: [{ id: 'old-provider-model', name: 'Old Provider Model' }],
    })
    await flushPromises()

    const emittedMessages = wrapper.emitted('showMessage') ?? []
    expect(emittedMessages).not.toContainEqual(['获取到 1 个模型', 'success'])
    expect(wrapper.text()).not.toContain('Old Provider Model')
  })
})
