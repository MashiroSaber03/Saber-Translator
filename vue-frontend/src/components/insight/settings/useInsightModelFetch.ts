import { computed, ref, type Ref } from 'vue'

import * as insightApi from '@/api/insight'
import { useAiModelDiscovery } from '@/composables/useAiModelDiscovery'
import { SUPPORTED_FETCH_PROVIDERS } from './types'

type MessageType = 'success' | 'error'

type ModelFetchOptions = {
  domain: 'insight_chat' | 'insight_embedding' | 'insight_reranker' | 'insight_vlm'
  provider: Ref<string>
  apiKey: Ref<string>
  baseUrl: Ref<string>
  model: Ref<string>
  requiresApiKey?: (provider: string) => boolean
  emitMessage: (message: string, type: MessageType) => void
}

function defaultFetchError(error: unknown): string {
  return '获取模型列表失败: ' + (error instanceof Error ? error.message : '网络错误')
}

export function useInsightModelFetch(options: ModelFetchOptions) {
  const modelSelectVisible = ref(false)
  const discovery = useAiModelDiscovery({
    source: () => ({
      provider: options.provider.value,
      apiKey: options.apiKey.value,
      baseUrl: options.baseUrl.value,
    }),
    fetcher: (provider, apiKey, baseUrl) => insightApi.fetchModels(
      provider,
      apiKey,
      baseUrl || undefined,
      options.domain,
    ),
    notify: (message, type) => options.emitMessage(message, type === 'warning' ? 'error' : type),
    supportsProvider: provider => SUPPORTED_FETCH_PROVIDERS.includes(provider),
    requiresApiKey: options.requiresApiKey,
    validationTone: 'error',
    emptyTone: 'error',
    providerLabel: provider => provider,
    errorMessage: defaultFetchError,
  })
  const { isFetchingModels } = discovery

  const modelOptions = computed(() => {
    if (!modelSelectVisible.value || discovery.models.value.length === 0) return []
    return [
      { label: '-- 选择模型 --', value: '' },
      ...discovery.models.value.map(item => ({
        label: item.name || item.id,
        value: item.id,
      })),
    ]
  })

  const modelCount = computed(() => discovery.models.value.length)

  function resetModelOptions(): void {
    discovery.clearModels()
    modelSelectVisible.value = false
  }

  function invalidateModelFetch(): void {
    discovery.invalidate()
    resetModelOptions()
  }

  async function fetchModels(): Promise<void> {
    const result = await discovery.fetchModels()
    modelSelectVisible.value = Boolean(result?.length)
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
