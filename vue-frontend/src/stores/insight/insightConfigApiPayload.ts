import type {
  BatchConfig,
  StoreEmbeddingConfig,
  StoreImageGenConfig,
  StoreInsightConfig,
  StoreLlmConfig,
  StoreOpenAICompatibleOptions,
  StoreRerankerConfig,
  StoreVlmConfig,
} from '@/types/insight'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'
import type { ProviderConfigsCache } from './useInsightConfigManager'

function serializeInsightOpenAiOptions(
  options: StoreOpenAICompatibleOptions,
): ReturnType<typeof serializeOpenAICompatibleOptionsForApi> {
  return serializeOpenAICompatibleOptionsForApi(options)
}

function serializeActiveVlmConfigForApi(config: StoreVlmConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    api_key: config.apiKey,
    model: config.model,
    base_url: config.baseUrl || null,
    openai_options: serializeInsightOpenAiOptions(config.openaiOptions),
    image_max_size: config.imageMaxSize,
  }
}

function serializeActiveLlmConfigForApi(config: StoreLlmConfig): Record<string, unknown> {
  return {
    use_same_as_vlm: config.useSameAsVlm,
    provider: config.provider,
    api_key: config.apiKey,
    model: config.model,
    base_url: config.baseUrl || null,
    openai_options: serializeInsightOpenAiOptions(config.openaiOptions),
  }
}

function serializeEmbeddingConfigForApi(config: StoreEmbeddingConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    api_key: config.apiKey,
    model: config.model,
    base_url: config.baseUrl || null,
    rpm_limit: config.rpmLimit,
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function serializeRerankerConfigForApi(config: StoreRerankerConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    api_key: config.apiKey,
    model: config.model,
    base_url: config.baseUrl || null,
    top_k: config.topK,
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function serializeImageGenConfigForApi(config: StoreImageGenConfig): Record<string, unknown> {
  return {
    provider: config.provider,
    api_key: config.apiKey,
    model: config.model,
    base_url: config.baseUrl || null,
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function serializeBatchConfigForApi(config: BatchConfig): Record<string, unknown> {
  return {
    pages_per_batch: config.pagesPerBatch,
    context_batch_count: config.contextBatchCount,
    architecture_preset: config.architecturePreset,
    custom_layers: config.customLayers.map(layer => ({
      name: layer.name,
      units_per_group: layer.units,
      align_to_chapter: layer.align,
    })),
  }
}

function serializeVlmProviderConfigForApi(config: Partial<StoreVlmConfig>): Record<string, unknown> {
  return {
    api_key: config.apiKey || '',
    model: config.model || '',
    base_url: config.baseUrl || '',
    openai_options: serializeInsightOpenAiOptions(config.openaiOptions as StoreOpenAICompatibleOptions),
    image_max_size: config.imageMaxSize ?? 1280,
  }
}

function serializeLlmProviderConfigForApi(config: Partial<StoreLlmConfig>): Record<string, unknown> {
  return {
    api_key: config.apiKey || '',
    model: config.model || '',
    base_url: config.baseUrl || '',
    openai_options: serializeInsightOpenAiOptions(config.openaiOptions as StoreOpenAICompatibleOptions),
  }
}

function serializeEmbeddingProviderConfigForApi(config: Partial<StoreEmbeddingConfig>): Record<string, unknown> {
  return {
    api_key: config.apiKey || '',
    model: config.model || '',
    base_url: config.baseUrl || '',
    rpm_limit: config.rpmLimit ?? 0,
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function serializeRerankerProviderConfigForApi(config: Partial<StoreRerankerConfig>): Record<string, unknown> {
  return {
    api_key: config.apiKey || '',
    model: config.model || '',
    base_url: config.baseUrl || '',
    top_k: config.topK ?? 5,
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function serializeImageGenProviderConfigForApi(config: Partial<StoreImageGenConfig>): Record<string, unknown> {
  return {
    api_key: config.apiKey || '',
    model: config.model || '',
    base_url: config.baseUrl || '',
    transport_retries: config.transportRetries ?? 10,
    business_retries: config.businessRetries ?? 10,
    timeout_seconds: config.timeoutSeconds ?? 0,
  }
}

function mapProviderConfig<T>(
  cache: Record<string, T>,
  mapper: (config: T) => Record<string, unknown>,
): Record<string, Record<string, unknown>> {
  return Object.fromEntries(
    Object.entries(cache).map(([provider, config]) => [provider, mapper(config)]),
  )
}

export function buildInsightConfigApiPayload(
  config: StoreInsightConfig,
  providerConfigs: ProviderConfigsCache,
): Record<string, unknown> {
  return {
    vlm: serializeActiveVlmConfigForApi(config.vlm),
    chat_llm: serializeActiveLlmConfigForApi(config.llm),
    embedding: serializeEmbeddingConfigForApi(config.embedding),
    reranker: serializeRerankerConfigForApi(config.reranker),
    image_gen: serializeImageGenConfigForApi(config.imageGen),
    analysis: { batch: serializeBatchConfigForApi(config.batch) },
    prompts: config.prompts,
    provider_settings: {
      vlmProvider: mapProviderConfig(providerConfigs.vlm, serializeVlmProviderConfigForApi),
      llmProvider: mapProviderConfig(providerConfigs.llm, serializeLlmProviderConfigForApi),
      embeddingProvider: mapProviderConfig(providerConfigs.embedding, serializeEmbeddingProviderConfigForApi),
      rerankerProvider: mapProviderConfig(providerConfigs.reranker, serializeRerankerProviderConfigForApi),
      imageGenProvider: mapProviderConfig(providerConfigs.imageGen, serializeImageGenProviderConfigForApi),
    },
  }
}
