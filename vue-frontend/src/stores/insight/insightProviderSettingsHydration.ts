import { deserializeOpenAICompatibleOptionsFromApi } from '@/utils/openaiOptions'
import type { ProviderConfigsCache } from './useInsightConfigManager'

type ProviderSettingsPayload = Record<string, Record<string, Record<string, unknown>>>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readProviderGroup(
  payload: ProviderSettingsPayload,
  key: keyof ProviderSettingsPayload,
): Record<string, Record<string, unknown>> | undefined {
  const group = payload[key]
  return isRecord(group) ? group as Record<string, Record<string, unknown>> : undefined
}

export function applyInsightProviderSettingsFromApi(
  providerConfigs: ProviderConfigsCache,
  wirePayload: unknown,
): boolean {
  if (!isRecord(wirePayload)) return false
  const settings = wirePayload as ProviderSettingsPayload

  const vlmProvider = readProviderGroup(settings, 'vlmProvider')
  if (vlmProvider) {
    for (const [provider, config] of Object.entries(vlmProvider)) {
      providerConfigs.vlm[provider] = {
        apiKey: (config.api_key as string) || '',
        model: (config.model as string) || '',
        baseUrl: (config.base_url as string) || '',
        openaiOptions: deserializeOpenAICompatibleOptionsFromApi(config.openai_options, {
          request: { forceJsonOutput: false, temperature: 0.3 },
          execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
        }),
        imageMaxSize: (config.image_max_size as number) ?? 1280,
      }
    }
  }

  const llmProvider = readProviderGroup(settings, 'llmProvider')
  if (llmProvider) {
    for (const [provider, config] of Object.entries(llmProvider)) {
      providerConfigs.llm[provider] = {
        apiKey: (config.api_key as string) || '',
        model: (config.model as string) || '',
        baseUrl: (config.base_url as string) || '',
        openaiOptions: deserializeOpenAICompatibleOptionsFromApi(config.openai_options, {
          request: { forceJsonOutput: false },
          execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
        }),
      }
    }
  }

  const embeddingProvider = readProviderGroup(settings, 'embeddingProvider')
  if (embeddingProvider) {
    for (const [provider, config] of Object.entries(embeddingProvider)) {
      providerConfigs.embedding[provider] = {
        apiKey: (config.api_key as string) || '',
        model: (config.model as string) || '',
        baseUrl: (config.base_url as string) || '',
        rpmLimit: (config.rpm_limit as number) ?? 0,
        transportRetries: (config.transport_retries as number) ?? 10,
        businessRetries: (config.business_retries as number) ?? 10,
        timeoutSeconds: (config.timeout_seconds as number) ?? 0,
      }
    }
  }

  const rerankerProvider = readProviderGroup(settings, 'rerankerProvider')
  if (rerankerProvider) {
    for (const [provider, config] of Object.entries(rerankerProvider)) {
      providerConfigs.reranker[provider] = {
        apiKey: (config.api_key as string) || '',
        model: (config.model as string) || '',
        baseUrl: (config.base_url as string) || '',
        topK: (config.top_k as number) ?? 5,
        transportRetries: (config.transport_retries as number) ?? 10,
        businessRetries: (config.business_retries as number) ?? 10,
        timeoutSeconds: (config.timeout_seconds as number) ?? 0,
      }
    }
  }

  const imageGenProvider = readProviderGroup(settings, 'imageGenProvider')
  if (imageGenProvider) {
    for (const [provider, config] of Object.entries(imageGenProvider)) {
      providerConfigs.imageGen[provider] = {
        apiKey: (config.api_key as string) || '',
        model: (config.model as string) || '',
        baseUrl: (config.base_url as string) || '',
        transportRetries: (config.transport_retries as number) ?? 10,
        businessRetries: (config.business_retries as number) ?? 10,
        timeoutSeconds: (config.timeout_seconds as number) ?? 0,
      }
    }
  }

  return true
}
