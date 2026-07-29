import type { OpenAICompatibleOptions } from '@/types/settings'
import { cloneOpenAiOptions } from '@/utils/openaiOptions'
import type { Ref } from 'vue'

const DEFAULT_VLM_OPENAI_OPTIONS = {
  request: { forceJsonOutput: false, temperature: 0.3 },
  execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
} satisfies OpenAICompatibleOptions
const DEFAULT_LLM_OPENAI_OPTIONS = {
  request: { forceJsonOutput: false },
  execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 },
} satisfies OpenAICompatibleOptions

interface ProviderFieldMap {
  apiKey: string
  model: string
  baseUrl: string
  [key: string]: unknown
}

interface VlmFields extends ProviderFieldMap {
  openaiOptions: OpenAICompatibleOptions
  imageMaxSize: number
}

interface LlmFields extends ProviderFieldMap {
  openaiOptions: OpenAICompatibleOptions
}

interface EmbeddingFields extends ProviderFieldMap {
  rpmLimit: number
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

interface RerankerFields extends ProviderFieldMap {
  topK: number
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

interface ImageGenFields extends ProviderFieldMap {
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

export interface ProviderConfigsCache {
  vlm: Record<string, Partial<VlmFields>>
  llm: Record<string, Partial<LlmFields>>
  embedding: Record<string, Partial<EmbeddingFields>>
  reranker: Record<string, Partial<RerankerFields>>
  imageGen: Record<string, Partial<ImageGenFields>>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasNumberField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'number' && Number.isFinite(value[key])
}

function hasBooleanField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'boolean'
}

function isOpenAiOptions(value: unknown): value is OpenAICompatibleOptions {
  if (!isRecord(value) || !isRecord(value.request) || !isRecord(value.execution)) return false
  if (!hasBooleanField(value.request, 'forceJsonOutput')) return false
  if (value.request.temperature !== undefined && !hasNumberField(value.request, 'temperature')) return false
  if (value.request.extraBody !== undefined && !isRecord(value.request.extraBody)) return false
  return (
    hasBooleanField(value.execution, 'useStream') &&
    hasNumberField(value.execution, 'rpmLimit') &&
    hasNumberField(value.execution, 'transportRetries') &&
    hasNumberField(value.execution, 'businessRetries')
  )
}

function cloneOpenAiOptionsOrDefault(value: unknown, fallback: OpenAICompatibleOptions): OpenAICompatibleOptions {
  return cloneOpenAiOptions(isOpenAiOptions(value) ? value : fallback)
}

export function useInsightConfigManager(
  providerConfigs: Ref<ProviderConfigsCache>
) {
  function createProviderManager<T extends ProviderFieldMap>(
    configType: 'vlm' | 'llm' | 'embedding' | 'reranker' | 'imageGen',
    fieldExtractor: (config: Record<string, unknown>) => Partial<T>,
    fieldApplier: (config: Record<string, unknown>, cached: Partial<T>) => void,
    defaultFields: Partial<T>
  ) {
    return {
      save(provider: string, currentConfig: Record<string, unknown>): void {
        if (!provider) return
        const cache = providerConfigs.value[configType] as Record<string, Partial<T>>
        cache[provider] = fieldExtractor(currentConfig)
      },

      restore(provider: string, currentConfig: Record<string, unknown>): void {
        if (!provider) return
        const cache = providerConfigs.value[configType] as Record<string, Partial<T>>
        const cached = cache[provider]
        if (cached) {
          fieldApplier(currentConfig, cached)
        } else {
          fieldApplier(currentConfig, defaultFields)
        }
      },

      switch(
        previousProvider: string,
        newProvider: string,
        currentConfig: Record<string, unknown>
      ): void {
        if (previousProvider === newProvider) return
        this.save(previousProvider, currentConfig)
        this.restore(newProvider, currentConfig)
      }
    }
  }

  const vlmManager = createProviderManager<VlmFields>(
    'vlm',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      openaiOptions: cloneOpenAiOptionsOrDefault(config.openaiOptions, DEFAULT_VLM_OPENAI_OPTIONS),
      imageMaxSize: config.imageMaxSize as number
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.openaiOptions !== undefined) config.openaiOptions = cloneOpenAiOptions(cached.openaiOptions)
      if (cached.imageMaxSize !== undefined) config.imageMaxSize = cached.imageMaxSize
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  const llmManager = createProviderManager<LlmFields>(
    'llm',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      openaiOptions: cloneOpenAiOptionsOrDefault(config.openaiOptions, DEFAULT_LLM_OPENAI_OPTIONS)
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.openaiOptions !== undefined) config.openaiOptions = cloneOpenAiOptions(cached.openaiOptions)
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  const embeddingManager = createProviderManager<EmbeddingFields>(
    'embedding',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      rpmLimit: config.rpmLimit as number,
      transportRetries: config.transportRetries as number,
      businessRetries: config.businessRetries as number,
      timeoutSeconds: config.timeoutSeconds as number
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.rpmLimit !== undefined) config.rpmLimit = cached.rpmLimit
      if (cached.transportRetries !== undefined) config.transportRetries = cached.transportRetries
      if (cached.businessRetries !== undefined) config.businessRetries = cached.businessRetries
      if (cached.timeoutSeconds !== undefined) config.timeoutSeconds = cached.timeoutSeconds
    },
    { apiKey: '', model: '', baseUrl: '', rpmLimit: 0, transportRetries: 10, businessRetries: 10, timeoutSeconds: 0 }
  )

  const rerankerManager = createProviderManager<RerankerFields>(
    'reranker',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      topK: config.topK as number,
      transportRetries: config.transportRetries as number,
      businessRetries: config.businessRetries as number,
      timeoutSeconds: config.timeoutSeconds as number,
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.topK !== undefined) config.topK = cached.topK
      if (cached.transportRetries !== undefined) config.transportRetries = cached.transportRetries
      if (cached.businessRetries !== undefined) config.businessRetries = cached.businessRetries
      if (cached.timeoutSeconds !== undefined) config.timeoutSeconds = cached.timeoutSeconds
    },
    { apiKey: '', model: '', baseUrl: '', topK: 5, transportRetries: 10, businessRetries: 10, timeoutSeconds: 0 }
  )

  const imageGenManager = createProviderManager<ImageGenFields>(
    'imageGen',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      transportRetries: config.transportRetries as number,
      businessRetries: config.businessRetries as number,
      timeoutSeconds: config.timeoutSeconds as number,
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.transportRetries !== undefined) config.transportRetries = cached.transportRetries
      if (cached.businessRetries !== undefined) config.businessRetries = cached.businessRetries
      if (cached.timeoutSeconds !== undefined) config.timeoutSeconds = cached.timeoutSeconds
    },
    { apiKey: '', model: '', baseUrl: '', transportRetries: 10, businessRetries: 10, timeoutSeconds: 0 }
  )

  return {
    vlmManager,
    llmManager,
    embeddingManager,
    rerankerManager,
    imageGenManager
  }
}
