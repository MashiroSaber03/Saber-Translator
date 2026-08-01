import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { useAiModelDiscovery } from '@/composables/useAiModelDiscovery'
import type { FetchModelsResponse } from '@/types'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolvePromise => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('useAiModelDiscovery', () => {
  it('validates the current provider credentials before requesting models', async () => {
    const provider = ref('custom')
    const apiKey = ref('')
    const baseUrl = ref('')
    const notify = vi.fn()
    const fetcher = vi.fn()
    const discovery = useAiModelDiscovery({
      source: () => ({ provider: provider.value, apiKey: apiKey.value, baseUrl: baseUrl.value }),
      fetcher,
      notify,
    })

    await discovery.fetchModels()
    expect(fetcher).not.toHaveBeenCalled()
    expect(notify).toHaveBeenLastCalledWith('请先填写 API Key', 'warning')

    apiKey.value = 'secret'
    await discovery.fetchModels()
    expect(fetcher).not.toHaveBeenCalled()
    expect(notify).toHaveBeenLastCalledWith('自定义服务需要先填写 Base URL', 'warning')
  })

  it('stores discovered models and reports the current result', async () => {
    const notify = vi.fn()
    const fetcher = vi.fn().mockResolvedValue({
      models: [
        { id: 'model-a', name: 'Model A' },
        { id: 'model-b', name: 'Model B' },
      ],
    } satisfies FetchModelsResponse)
    const discovery = useAiModelDiscovery({
      source: () => ({ provider: 'openai', apiKey: 'secret', baseUrl: '' }),
      fetcher,
      notify,
      emptyBaseUrl: '',
    })

    await discovery.fetchModels()

    expect(fetcher).toHaveBeenCalledWith('openai', 'secret', '')
    expect(discovery.models.value).toEqual([
      { id: 'model-a', name: 'Model A' },
      { id: 'model-b', name: 'Model B' },
    ])
    expect(discovery.isFetchingModels.value).toBe(false)
    expect(notify).toHaveBeenCalledWith('获取到 2 个模型', 'success')
  })

  it('allows model discovery through a credential already stored by the backend', async () => {
    const fetcher = vi.fn().mockResolvedValue({
      models: [{ id: 'stored-model', name: 'Stored Model' }],
    } satisfies FetchModelsResponse)
    const discovery = useAiModelDiscovery({
      source: () => ({
        provider: 'deepseek',
        apiKey: '',
        baseUrl: '',
        hasStoredCredential: true,
      }),
      fetcher,
      notify: vi.fn(),
    })

    await discovery.fetchModels()

    expect(fetcher).toHaveBeenCalledWith('deepseek', '', undefined)
    expect(discovery.models.value).toEqual([
      { id: 'stored-model', name: 'Stored Model' },
    ])
  })

  it('ignores a response after the provider snapshot changes', async () => {
    const provider = ref('openai')
    const request = deferred<FetchModelsResponse>()
    const notify = vi.fn()
    const discovery = useAiModelDiscovery({
      source: () => ({ provider: provider.value, apiKey: 'secret', baseUrl: '' }),
      fetcher: vi.fn(() => request.promise),
      notify,
    })

    const pending = discovery.fetchModels()
    provider.value = 'deepseek'
    request.resolve({ models: [{ id: 'stale', name: 'Stale' }] })
    await pending

    expect(discovery.models.value).toEqual([])
    expect(notify).not.toHaveBeenCalledWith('获取到 1 个模型', 'success')
    expect(discovery.isFetchingModels.value).toBe(false)
  })

  it('invalidates an active request and clears its model options', async () => {
    const request = deferred<FetchModelsResponse>()
    const discovery = useAiModelDiscovery({
      source: () => ({ provider: 'openai', apiKey: 'secret', baseUrl: '' }),
      fetcher: vi.fn(() => request.promise),
      notify: vi.fn(),
    })
    discovery.models.value = [{ id: 'existing', name: 'Existing' }]

    const pending = discovery.fetchModels()
    discovery.invalidate()
    request.resolve({ models: [{ id: 'stale', name: 'Stale' }] })
    await pending

    expect(discovery.models.value).toEqual([])
    expect(discovery.isFetchingModels.value).toBe(false)
  })
})
