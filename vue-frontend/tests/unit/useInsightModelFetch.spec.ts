import { describe, expect, it, vi } from 'vitest'
import { flushPromises } from '@vue/test-utils'
import { ref } from 'vue'

import { useInsightModelFetch } from '@/components/insight/settings/useInsightModelFetch'

const { fetchModelsMock } = vi.hoisted(() => ({
  fetchModelsMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  fetchModels: fetchModelsMock,
}))

function createDeferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolver) => {
    resolve = resolver
  })
  return { promise, resolve }
}

describe('useInsightModelFetch', () => {
  it('blocks model fetches until required credentials are present', async () => {
    fetchModelsMock.mockReset()
    const messages: Array<[string, 'success' | 'error']> = []
    const modelFetch = useInsightModelFetch({
      provider: ref('openai'),
      apiKey: ref(''),
      baseUrl: ref(''),
      model: ref(''),
      requiresApiKey: () => true,
      emitMessage: (message, type) => messages.push([message, type]),
    })

    await modelFetch.fetchModels()

    expect(fetchModelsMock).not.toHaveBeenCalled()
    expect(messages).toEqual([['请先填写 API Key', 'error']])
    expect(modelFetch.isFetchingModels.value).toBe(false)
  })

  it('ignores stale model-list responses after invalidation', async () => {
    fetchModelsMock.mockReset()
    const deferred = createDeferred({
      success: true,
      models: [{ id: 'stale-model', name: 'Stale Model' }],
    })
    fetchModelsMock.mockReturnValueOnce(deferred.promise)
    const provider = ref('gemini')
    const messages: Array<[string, 'success' | 'error']> = []
    const modelFetch = useInsightModelFetch({
      provider,
      apiKey: ref('model-key'),
      baseUrl: ref(''),
      model: ref(''),
      emitMessage: (message, type) => messages.push([message, type]),
    })

    const request = modelFetch.fetchModels()
    provider.value = 'ollama'
    modelFetch.invalidateModelFetch()
    deferred.resolve({
      success: true,
      models: [{ id: 'stale-model', name: 'Stale Model' }],
    })
    await request
    await flushPromises()

    expect(modelFetch.isFetchingModels.value).toBe(false)
    expect(modelFetch.modelCount.value).toBe(0)
    expect(modelFetch.modelOptions.value).toEqual([])
    expect(messages).toEqual([])
  })

  it('exposes fetched model options and applies selections to the model ref', async () => {
    fetchModelsMock.mockReset()
    fetchModelsMock.mockResolvedValueOnce({
      success: true,
      models: [{ id: 'shared-model', name: 'Shared Model' }],
    })
    const model = ref('')
    const messages: Array<[string, 'success' | 'error']> = []
    const modelFetch = useInsightModelFetch({
      provider: ref('gemini'),
      apiKey: ref('model-key'),
      baseUrl: ref(''),
      model,
      emitMessage: (message, type) => messages.push([message, type]),
    })

    await modelFetch.fetchModels()

    expect(fetchModelsMock).toHaveBeenCalledWith('gemini', 'model-key', undefined)
    expect(modelFetch.modelCount.value).toBe(1)
    expect(modelFetch.modelOptions.value).toEqual([
      { label: '-- 选择模型 --', value: '' },
      { label: 'Shared Model', value: 'shared-model' },
    ])
    expect(messages).toEqual([['获取到 1 个模型', 'success']])

    modelFetch.selectModel('shared-model')

    expect(model.value).toBe('shared-model')
  })
})
