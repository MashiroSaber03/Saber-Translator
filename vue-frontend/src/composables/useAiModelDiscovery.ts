import { getCurrentScope, onScopeDispose, ref } from 'vue'

import { fetchModels as fetchV2Models } from '@/api/v2/diagnostics'
import {
  getProviderDisplayName,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import type { FetchModelsResponse, ModelInfoItem } from '@/types'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'

export type AiModelDiscoveryMessageTone = 'success' | 'warning' | 'error'

export interface AiModelDiscoverySource {
  provider: string
  apiKey?: string
  baseUrl?: string
  hasStoredCredential?: boolean
}

export interface AiModelDiscoveryOptions {
  source: () => AiModelDiscoverySource
  notify: (message: string, tone: AiModelDiscoveryMessageTone) => void
  fetcher?: (provider: string, apiKey: string, baseUrl?: string) => Promise<FetchModelsResponse>
  supportsProvider?: (provider: string) => boolean
  requiresApiKey?: (provider: string) => boolean
  requiresBaseUrl?: (provider: string) => boolean
  providerLabel?: (provider: string) => string
  validationTone?: AiModelDiscoveryMessageTone
  emptyTone?: AiModelDiscoveryMessageTone
  successMessage?: (count: number) => string
  emptyMessage?: () => string
  errorMessage?: (error: unknown) => string
  emptyBaseUrl?: string
}

interface AiModelDiscoverySnapshot {
  provider: string
  apiKey: string
  baseUrl: string
  hasStoredCredential: boolean
}

function snapshotSource(source: AiModelDiscoverySource): AiModelDiscoverySnapshot {
  return {
    provider: source.provider.trim(),
    apiKey: source.apiKey?.trim() ?? '',
    baseUrl: source.baseUrl?.trim() ?? '',
    hasStoredCredential: source.hasStoredCredential === true,
  }
}

function sameSnapshot(left: AiModelDiscoverySnapshot, right: AiModelDiscoverySnapshot): boolean {
  return left.provider === right.provider
    && left.apiKey === right.apiKey
    && left.baseUrl === right.baseUrl
    && left.hasStoredCredential === right.hasStoredCredential
}

export function useAiModelDiscovery(options: AiModelDiscoveryOptions) {
  const models = ref<ModelInfoItem[]>([])
  const isFetchingModels = ref(false)
  const requestGuard = useLatestRequestGuard()
  const fetcher = options.fetcher ?? fetchV2Models
  const supportsProvider = options.supportsProvider
    ?? ((provider: string) => providerSupportsCapability(provider, 'modelFetch'))
  const requiresApiKey = options.requiresApiKey ?? providerRequiresApiKey
  const requiresBaseUrl = options.requiresBaseUrl ?? providerRequiresBaseUrl
  const providerLabel = options.providerLabel ?? getProviderDisplayName
  const validationTone = options.validationTone ?? 'warning'
  const emptyTone = options.emptyTone ?? 'warning'

  function clearModels(): void {
    models.value = []
  }

  function invalidate(): void {
    requestGuard.invalidate()
    isFetchingModels.value = false
    clearModels()
  }

  function isCurrentRequest(requestId: number, snapshot: AiModelDiscoverySnapshot): boolean {
    return requestGuard.isCurrent(requestId, () => sameSnapshot(snapshotSource(options.source()), snapshot))
  }

  async function fetchModels(): Promise<ModelInfoItem[] | null> {
    const snapshot = snapshotSource(options.source())
    if (
      requiresApiKey(snapshot.provider)
      && !snapshot.apiKey
      && !snapshot.hasStoredCredential
    ) {
      options.notify('请先填写 API Key', validationTone)
      return null
    }
    if (!supportsProvider(snapshot.provider)) {
      options.notify(`${providerLabel(snapshot.provider)} 不支持自动获取模型列表`, validationTone)
      return null
    }
    if (requiresBaseUrl(snapshot.provider) && !snapshot.baseUrl) {
      options.notify('自定义服务需要先填写 Base URL', validationTone)
      return null
    }

    const requestId = requestGuard.next()
    isFetchingModels.value = true
    try {
      const response = await fetcher(
        snapshot.provider,
        snapshot.apiKey,
        snapshot.baseUrl || options.emptyBaseUrl,
      )
      if (!isCurrentRequest(requestId, snapshot)) return null

      if (response.models.length) {
        models.value = response.models
        options.notify(
          options.successMessage?.(response.models.length) ?? `获取到 ${response.models.length} 个模型`,
          'success',
        )
        return response.models
      }

      clearModels()
      options.notify(
        options.emptyMessage?.() ?? '未获取到可用模型',
        emptyTone,
      )
      return []
    } catch (error) {
      if (!isCurrentRequest(requestId, snapshot)) return null
      clearModels()
      options.notify(
        options.errorMessage?.(error) ?? (error instanceof Error ? error.message : '获取模型列表失败'),
        'error',
      )
      return null
    } finally {
      if (requestGuard.isCurrent(requestId)) {
        isFetchingModels.value = false
      }
    }
  }

  if (getCurrentScope()) onScopeDispose(invalidate)

  return {
    models,
    isFetchingModels,
    clearModels,
    invalidate,
    fetchModels,
  }
}
