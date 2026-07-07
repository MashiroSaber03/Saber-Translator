import { computed, ref, type Ref } from 'vue'

import * as insightApi from '@/api/insight'
import { providerRequiresApiKey } from '@/config/aiProviders'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'
import { SUPPORTED_FETCH_PROVIDERS, type ModelInfo } from './types'

type MessageType = 'success' | 'error'

type ModelFetchOptions = {
  provider: Ref<string>
  apiKey: Ref<string>
  baseUrl: Ref<string>
  model: Ref<string>
  requiresApiKey?: (provider: string) => boolean
  formatFetchError?: (error: unknown) => string
  emitMessage: (message: string, type: MessageType) => void
}

const DEFAULT_FETCH_ERROR = '获取模型列表失败'

export function useInsightModelFetch(options: ModelFetchOptions) {
  const models = ref<ModelInfo[]>([])
  const modelSelectVisible = ref(false)
  const isFetchingModels = ref(false)
  const modelFetchGuard = useLatestRequestGuard()
  const requiresApiKey = options.requiresApiKey ?? providerRequiresApiKey

  const modelOptions = computed(() => {
    if (!modelSelectVisible.value || models.value.length === 0) return []
    return [
      { label: '-- 选择模型 --', value: '' },
      ...models.value.map(item => ({
        label: item.name || item.id,
        value: item.id,
      })),
    ]
  })

  const modelCount = computed(() => models.value.length)

  function resetModelOptions(): void {
    models.value = []
    modelSelectVisible.value = false
  }

  function invalidateModelFetch(): void {
    modelFetchGuard.invalidate()
    isFetchingModels.value = false
    resetModelOptions()
  }

  async function fetchModels(): Promise<void> {
    const provider = options.provider.value
    const apiKey = options.apiKey.value
    const baseUrl = options.baseUrl.value || undefined

    if (requiresApiKey(provider) && !apiKey) {
      options.emitMessage('请先填写 API Key', 'error')
      return
    }

    if (!SUPPORTED_FETCH_PROVIDERS.includes(provider)) {
      options.emitMessage(`${provider} 不支持自动获取模型列表`, 'error')
      return
    }

    if (provider === 'custom' && !baseUrl) {
      options.emitMessage('自定义服务需要先填写 Base URL', 'error')
      return
    }

    isFetchingModels.value = true
    const requestId = modelFetchGuard.next()
    const isCurrentRequest = () => modelFetchGuard.isCurrent(requestId, () => (
      options.provider.value === provider &&
      options.apiKey.value === apiKey &&
      (options.baseUrl.value || undefined) === baseUrl
    ))

    try {
      const response = await insightApi.fetchModels(provider, apiKey, baseUrl)
      if (!isCurrentRequest()) return

      if (response.success && response.models?.length) {
        models.value = response.models
        modelSelectVisible.value = true
        options.emitMessage(`获取到 ${response.models.length} 个模型`, 'success')
      } else {
        options.emitMessage(response.message || '未获取到模型列表', 'error')
        modelSelectVisible.value = false
      }
    } catch (error) {
      if (isCurrentRequest()) {
        options.emitMessage(options.formatFetchError?.(error) ?? DEFAULT_FETCH_ERROR, 'error')
        modelSelectVisible.value = false
      }
    } finally {
      if (modelFetchGuard.isCurrent(requestId)) {
        isFetchingModels.value = false
      }
    }
  }

  function selectModel(modelId: string | number): void {
    const nextModel = String(modelId)
    if (nextModel) {
      options.model.value = nextModel
    }
  }

  return {
    isFetchingModels,
    modelOptions,
    modelCount,
    resetModelOptions,
    invalidateModelFetch,
    fetchModels,
    selectModel,
  }
}
