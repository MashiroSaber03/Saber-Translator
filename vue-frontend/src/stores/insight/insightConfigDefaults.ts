import { getProviderBaseUrl, getProviderDefaultModel, normalizeProviderId } from '@/config/aiProviders'
import type { StoreImageGenConfig, StoreRerankerConfig } from '@/types/insight'

export function normalizeInsightRerankerConfig(
  source?: Partial<StoreRerankerConfig> | null,
  previous?: StoreRerankerConfig,
): StoreRerankerConfig {
  const provider = normalizeProviderId(source?.provider || previous?.provider || 'jina') || 'jina'
  return {
    provider,
    apiKey: source?.apiKey ?? previous?.apiKey ?? '',
    model: source?.model ?? previous?.model ?? 'jina-reranker-v2-base-multilingual',
    baseUrl: source?.baseUrl ?? previous?.baseUrl ?? '',
    topK: source?.topK ?? previous?.topK ?? 5,
    transportRetries: source?.transportRetries ?? previous?.transportRetries ?? 10,
    businessRetries: source?.businessRetries ?? previous?.businessRetries ?? 10,
    timeoutSeconds: source?.timeoutSeconds ?? previous?.timeoutSeconds ?? 0,
  }
}

export function normalizeInsightImageGenConfig(
  source?: Partial<StoreImageGenConfig> | null,
  previous?: StoreImageGenConfig,
): StoreImageGenConfig {
  const normalizedProvider = normalizeProviderId(source?.provider || previous?.provider || 'gpt2api') || 'gpt2api'
  const previousProvider = normalizeProviderId(previous?.provider || '') || 'gpt2api'
  const providerChanged = normalizedProvider !== previousProvider
  const providerDefaultModel = getProviderDefaultModel(normalizedProvider, 'imageGen')
  const defaultModel = providerDefaultModel || (normalizedProvider === 'gpt2api' ? 'gpt-image-2' : '')
  const defaultBaseUrl = getProviderBaseUrl(normalizedProvider, 'imageGen')
  const base = previous ?? {
    provider: normalizedProvider,
    apiKey: '',
    model: defaultModel,
    baseUrl: defaultBaseUrl,
    transportRetries: 10,
    businessRetries: 10,
    timeoutSeconds: 0,
  }
  const model = source?.model ?? (providerChanged ? providerDefaultModel : base.model || defaultModel)
  const baseUrl = source?.baseUrl ?? (providerChanged ? defaultBaseUrl : (base.baseUrl || defaultBaseUrl))
  const businessRetries = source?.businessRetries ?? base.businessRetries ?? 10

  return {
    provider: normalizedProvider,
    apiKey: source?.apiKey ?? base.apiKey,
    model,
    baseUrl,
    transportRetries: source?.transportRetries ?? base.transportRetries ?? 10,
    businessRetries,
    timeoutSeconds: source?.timeoutSeconds ?? base.timeoutSeconds ?? 0,
  }
}
