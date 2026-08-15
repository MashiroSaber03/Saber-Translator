import type {
  InsightEmbeddingProviderDraft,
  InsightImageGenProviderDraft,
  InsightLlmProviderDraft,
  InsightProviderDrafts,
  InsightRerankerProviderDraft,
  InsightVlmProviderDraft,
  StoreEmbeddingConfig,
  StoreImageGenConfig,
  StoreLlmConfig,
  StoreRerankerConfig,
  StoreVlmConfig,
} from '@/types/insight'
import { getProviderBaseUrl, getProviderDefaultModel } from '@/config/aiProviders'
import { cloneOpenAiOptions } from '@/utils/openaiOptions'
import type { Ref } from 'vue'

function createProviderManager<TConfig, TDraft>(
  cache: () => Record<string, TDraft>,
  createDraft: (config: TConfig) => TDraft,
  applyDraft: (config: TConfig, draft: TDraft) => void,
  createDefaultDraft: (provider: string) => TDraft
) {
  function save(provider: string, currentConfig: TConfig): void {
    if (!provider) return
    cache()[provider] = createDraft(currentConfig)
  }

  function restore(provider: string, currentConfig: TConfig): void {
    if (!provider) return
    applyDraft(currentConfig, cache()[provider] ?? createDefaultDraft(provider))
  }

  return {
    save,
    restore,

    switch(previousProvider: string, newProvider: string, currentConfig: TConfig): void {
      if (previousProvider === newProvider) return
      save(previousProvider, currentConfig)
      restore(newProvider, currentConfig)
    },
  }
}

export function useInsightConfigManager(providerConfigs: Ref<InsightProviderDrafts>) {
  const vlmManager = createProviderManager<StoreVlmConfig, InsightVlmProviderDraft>(
    () => providerConfigs.value.vlm,
    config => ({
      apiKey: config.apiKey,
      model: config.model,
      baseUrl: config.baseUrl,
      openaiOptions: cloneOpenAiOptions(config.openaiOptions),
      imageMaxSize: config.imageMaxSize,
    }),
    (config, draft) => {
      config.apiKey = draft.apiKey
      config.model = draft.model
      config.baseUrl = draft.baseUrl
      config.openaiOptions = cloneOpenAiOptions(draft.openaiOptions)
      config.imageMaxSize = draft.imageMaxSize
    },
    () => ({
      apiKey: '',
      model: '',
      baseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: false, temperature: 0.3 },
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 0,
        },
      },
      imageMaxSize: 0,
    })
  )

  const llmManager = createProviderManager<StoreLlmConfig, InsightLlmProviderDraft>(
    () => providerConfigs.value.llm,
    config => ({
      apiKey: config.apiKey,
      model: config.model,
      baseUrl: config.baseUrl,
      openaiOptions: cloneOpenAiOptions(config.openaiOptions),
    }),
    (config, draft) => {
      config.apiKey = draft.apiKey
      config.model = draft.model
      config.baseUrl = draft.baseUrl
      config.openaiOptions = cloneOpenAiOptions(draft.openaiOptions)
    },
    () => ({
      apiKey: '',
      model: '',
      baseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: false },
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 0,
        },
      },
    })
  )

  const embeddingManager = createProviderManager<
    StoreEmbeddingConfig,
    InsightEmbeddingProviderDraft
  >(
    () => providerConfigs.value.embedding,
    ({ provider: _provider, ...draft }) => draft,
    (config, draft) => Object.assign(config, draft),
    () => ({
      apiKey: '',
      model: '',
      baseUrl: '',
      rpmLimit: 0,
      transportRetries: 1,
      businessRetries: 0,
      timeoutSeconds: 0,
    })
  )

  const rerankerManager = createProviderManager<StoreRerankerConfig, InsightRerankerProviderDraft>(
    () => providerConfigs.value.reranker,
    ({ provider: _provider, ...draft }) => draft,
    (config, draft) => Object.assign(config, draft),
    () => ({
      apiKey: '',
      model: '',
      baseUrl: '',
      transportRetries: 1,
      businessRetries: 0,
      timeoutSeconds: 0,
    })
  )

  const imageGenManager = createProviderManager<StoreImageGenConfig, InsightImageGenProviderDraft>(
    () => providerConfigs.value.imageGen,
    ({ provider: _provider, ...draft }) => draft,
    (config, draft) => Object.assign(config, draft),
    provider => ({
      apiKey: '',
      model: getProviderDefaultModel(provider, 'imageGen'),
      baseUrl: getProviderBaseUrl(provider, 'imageGen'),
      transportRetries: 1,
      businessRetries: 0,
      timeoutSeconds: 0,
    })
  )

  return {
    vlmManager,
    llmManager,
    embeddingManager,
    rerankerManager,
    imageGenManager,
  }
}
