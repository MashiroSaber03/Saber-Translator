import type { StoreInsightConfig } from '@/types/insight'
import { deserializeOpenAICompatibleOptionsFromApi } from '@/utils/openaiOptions'
import {
  normalizeInsightImageGenConfig,
  normalizeInsightRerankerConfig,
} from './insightConfigDefaults'

export function applyActiveInsightConfigFromApi(
  config: StoreInsightConfig,
  apiConfig: Record<string, unknown>,
): void {
  const vlm = apiConfig.vlm as Record<string, unknown> | undefined
  const chatLlm = apiConfig.chat_llm as Record<string, unknown> | undefined
  const embedding = apiConfig.embedding as Record<string, unknown> | undefined
  const reranker = apiConfig.reranker as Record<string, unknown> | undefined
  const batch = (apiConfig.analysis as Record<string, unknown> | undefined)?.batch as Record<string, unknown> | undefined
  const imageGen = apiConfig.image_gen as Record<string, unknown> | undefined

  if (vlm) {
    config.vlm = {
      provider: (vlm.provider as string) || 'gemini',
      apiKey: (vlm.api_key as string) || '',
      model: (vlm.model as string) || '',
      baseUrl: (vlm.base_url as string) || '',
      openaiOptions: deserializeOpenAICompatibleOptionsFromApi(vlm.openai_options, {
        request: { forceJsonOutput: false, temperature: 0.3 },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
      }),
      imageMaxSize: vlm.image_max_size !== undefined && vlm.image_max_size !== null
        ? Number(vlm.image_max_size)
        : 1280,
    }
  }

  if (chatLlm) {
    config.llm = {
      useSameAsVlm: chatLlm.use_same_as_vlm === true,
      provider: (chatLlm.provider as string) || config.vlm.provider,
      apiKey: (chatLlm.api_key as string) || config.vlm.apiKey,
      model: (chatLlm.model as string) || config.vlm.model,
      baseUrl: (chatLlm.base_url as string) || config.vlm.baseUrl || '',
      openaiOptions: deserializeOpenAICompatibleOptionsFromApi(chatLlm.openai_options, {
        request: { forceJsonOutput: false },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
      }),
    }
  }

  if (embedding) {
    config.embedding = {
      provider: (embedding.provider as string) || 'openai',
      apiKey: (embedding.api_key as string) || '',
      model: (embedding.model as string) || '',
      baseUrl: (embedding.base_url as string) || '',
      rpmLimit: (embedding.rpm_limit as number) ?? 0,
      transportRetries: (embedding.transport_retries as number) ?? 10,
      businessRetries: (embedding.business_retries as number) ?? 10,
      timeoutSeconds: (embedding.timeout_seconds as number) ?? 0,
    }
  }

  if (reranker) {
    config.reranker = normalizeInsightRerankerConfig({
      provider: (reranker.provider as string) || 'jina',
      apiKey: (reranker.api_key as string) || '',
      model: (reranker.model as string) || '',
      baseUrl: (reranker.base_url as string) || '',
      topK: (reranker.top_k as number) || 5,
      transportRetries: (reranker.transport_retries as number) ?? 10,
      businessRetries: (reranker.business_retries as number) ?? 10,
      timeoutSeconds: (reranker.timeout_seconds as number) ?? 0,
    }, config.reranker)
  }

  if (batch) {
    const customLayers = batch.custom_layers as Array<Record<string, unknown>> | undefined
    config.batch = {
      pagesPerBatch: (batch.pages_per_batch as number) || 5,
      contextBatchCount: (batch.context_batch_count as number) ?? 3,
      architecturePreset: (batch.architecture_preset as string) || 'standard',
      customLayers: customLayers?.map(layer => ({
        name: (layer.name as string) || '',
        units: (layer.units_per_group as number) || 1,
        align: (layer.align_to_chapter as boolean) || false,
      })) || [],
    }
  }

  if (imageGen) {
    config.imageGen = normalizeInsightImageGenConfig({
      provider: imageGen.provider as string | undefined,
      apiKey: (imageGen.api_key as string) || '',
      model: imageGen.model as string | undefined,
      baseUrl: (imageGen.base_url as string) || '',
      transportRetries: (imageGen.transport_retries as number) ?? 10,
      businessRetries: (imageGen.business_retries as number) ?? 10,
      timeoutSeconds: (imageGen.timeout_seconds as number) ?? 0,
    }, config.imageGen)
  }
}
