/**
 * Insight 配置管理 Composable
 *
 * 统一管理 VLM/LLM/Embedding/Reranker/ImageGen 五种服务商配置的保存/恢复
 */

import type { Ref } from 'vue'

/** localStorage 存储键 */
const STORAGE_KEY = 'insight_provider_configs'
const INSIGHT_PROVIDER_CONFIG_SCHEMA_VERSION = 1

/** 服务商配置字段映射 */
interface ProviderFieldMap {
  apiKey: string
  model: string
  baseUrl: string
  [key: string]: unknown
}

/** VLM 配置字段 */
interface VlmFields extends ProviderFieldMap {
  openaiOptions: {
    request: {
      forceJsonOutput: boolean
      temperature?: number
      extraBody?: Record<string, unknown>
    }
    execution: {
      useStream: boolean
      rpmLimit: number
      transportRetries: number
      businessRetries: number
    }
  }
  imageMaxSize: number
}

/** LLM 配置字段 */
interface LlmFields extends ProviderFieldMap {
  openaiOptions: {
    request: {
      forceJsonOutput: boolean
      temperature?: number
      extraBody?: Record<string, unknown>
    }
    execution: {
      useStream: boolean
      rpmLimit: number
      transportRetries: number
      businessRetries: number
    }
  }
}

/** Embedding 配置字段 */
interface EmbeddingFields extends ProviderFieldMap {
  rpmLimit: number
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

/** Reranker 配置字段 */
interface RerankerFields extends ProviderFieldMap {
  topK: number
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

/** ImageGen 配置字段 */
interface ImageGenFields extends ProviderFieldMap {
  transportRetries: number
  businessRetries: number
  timeoutSeconds: number
}

/** 服务商配置缓存结构 */
export interface ProviderConfigsCache {
  vlm: Record<string, Partial<VlmFields>>
  llm: Record<string, Partial<LlmFields>>
  embedding: Record<string, Partial<EmbeddingFields>>
  reranker: Record<string, Partial<RerankerFields>>
  imageGen: Record<string, Partial<ImageGenFields>>
}

type ProviderConfigsStoragePayload = ProviderConfigsCache & {
  insightProviderConfigSchemaVersion: typeof INSIGHT_PROVIDER_CONFIG_SCHEMA_VERSION
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasStringField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'string'
}

function hasNumberField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'number' && Number.isFinite(value[key])
}

function hasBooleanField(value: Record<string, unknown>, key: string): boolean {
  return typeof value[key] === 'boolean'
}

function isOpenAiOptions(value: unknown): boolean {
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

function isProviderGroup<T extends Record<string, unknown>>(
  value: unknown,
  isConfig: (config: Record<string, unknown>) => config is T,
): value is Record<string, T> {
  if (!isRecord(value)) return false
  return Object.values(value).every(config => isRecord(config) && isConfig(config))
}

function hasBaseProviderFields(value: Record<string, unknown>): boolean {
  return hasStringField(value, 'apiKey') && hasStringField(value, 'model') && hasStringField(value, 'baseUrl')
}

function isVlmConfig(value: Record<string, unknown>): value is VlmFields {
  return hasBaseProviderFields(value) && isOpenAiOptions(value.openaiOptions) && hasNumberField(value, 'imageMaxSize')
}

function isLlmConfig(value: Record<string, unknown>): value is LlmFields {
  return hasBaseProviderFields(value) && isOpenAiOptions(value.openaiOptions)
}

function isEmbeddingConfig(value: Record<string, unknown>): value is EmbeddingFields {
  return (
    hasBaseProviderFields(value) &&
    hasNumberField(value, 'rpmLimit') &&
    hasNumberField(value, 'transportRetries') &&
    hasNumberField(value, 'businessRetries') &&
    hasNumberField(value, 'timeoutSeconds')
  )
}

function isRerankerConfig(value: Record<string, unknown>): value is RerankerFields {
  return (
    hasBaseProviderFields(value) &&
    hasNumberField(value, 'topK') &&
    hasNumberField(value, 'transportRetries') &&
    hasNumberField(value, 'businessRetries') &&
    hasNumberField(value, 'timeoutSeconds')
  )
}

function isImageGenConfig(value: Record<string, unknown>): value is ImageGenFields {
  return (
    hasBaseProviderFields(value) &&
    hasNumberField(value, 'transportRetries') &&
    hasNumberField(value, 'businessRetries') &&
    hasNumberField(value, 'timeoutSeconds')
  )
}

function parseProviderConfigsStorage(value: unknown): ProviderConfigsCache | null {
  if (!isRecord(value)) return null
  if (value.insightProviderConfigSchemaVersion !== INSIGHT_PROVIDER_CONFIG_SCHEMA_VERSION) return null
  if (!isProviderGroup(value.vlm, isVlmConfig)) return null
  if (!isProviderGroup(value.llm, isLlmConfig)) return null
  if (!isProviderGroup(value.embedding, isEmbeddingConfig)) return null
  if (!isProviderGroup(value.reranker, isRerankerConfig)) return null
  if (!isProviderGroup(value.imageGen, isImageGenConfig)) return null
  return {
    vlm: value.vlm,
    llm: value.llm,
    embedding: value.embedding,
    reranker: value.reranker,
    imageGen: value.imageGen,
  }
}

/**
 * 创建配置管理器
 */
export function useInsightConfigManager(
  providerConfigs: Ref<ProviderConfigsCache>
) {
  /**
   * 保存配置缓存到 localStorage
   */
  function saveToStorage(): void {
    const payload: ProviderConfigsStoragePayload = {
      insightProviderConfigSchemaVersion: INSIGHT_PROVIDER_CONFIG_SCHEMA_VERSION,
      ...providerConfigs.value,
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }

  /**
   * 从 localStorage 加载配置缓存
   */
  function loadFromStorage(): void {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      try {
        const parsed = parseProviderConfigsStorage(JSON.parse(stored) as unknown)
        if (parsed) providerConfigs.value = parsed
      } catch {
        return
      }
    }
  }

  /**
   * 创建通用的服务商配置管理器
   */
  function createProviderManager<T extends ProviderFieldMap>(
    configType: 'vlm' | 'llm' | 'embedding' | 'reranker' | 'imageGen',
    fieldExtractor: (config: Record<string, unknown>) => Partial<T>,
    fieldApplier: (config: Record<string, unknown>, cached: Partial<T>) => void,
    defaultFields: Partial<T>
  ) {
    return {
      /**
       * 保存当前服务商配置到缓存
       */
      save(provider: string, currentConfig: Record<string, unknown>): void {
        if (!provider) return
        const cache = providerConfigs.value[configType] as Record<string, Partial<T>>
        cache[provider] = fieldExtractor(currentConfig)
        saveToStorage()
      },

      /**
       * 从缓存恢复服务商配置
       */
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

      /**
       * 切换服务商时保存当前配置，并恢复目标服务商配置。
       */
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

  // VLM 配置管理器
  const vlmManager = createProviderManager<VlmFields>(
    'vlm',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      openaiOptions: JSON.parse(JSON.stringify(config.openaiOptions || {
        request: { forceJsonOutput: false, temperature: 0.3 },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
      })),
      imageMaxSize: config.imageMaxSize as number
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.openaiOptions !== undefined) config.openaiOptions = JSON.parse(JSON.stringify(cached.openaiOptions))
      if (cached.imageMaxSize !== undefined) config.imageMaxSize = cached.imageMaxSize
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  // LLM 配置管理器
  const llmManager = createProviderManager<LlmFields>(
    'llm',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      openaiOptions: JSON.parse(JSON.stringify(config.openaiOptions || {
        request: { forceJsonOutput: false },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
      }))
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.openaiOptions !== undefined) config.openaiOptions = JSON.parse(JSON.stringify(cached.openaiOptions))
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  // Embedding 配置管理器
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

  // Reranker 配置管理器
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

  // ImageGen 配置管理器
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
    saveToStorage,
    loadFromStorage,
    vlmManager,
    llmManager,
    embeddingManager,
    rerankerManager,
    imageGenManager
  }
}
