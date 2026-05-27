/**
 * Insight 配置管理 Composable
 *
 * 统一管理 VLM/LLM/Embedding/Reranker 四种服务商配置的保存/恢复
 */

import type { Ref } from 'vue'
import { normalizeOpenAiOptions } from '@/utils/openaiOptions'

/** localStorage 存储键 */
const STORAGE_KEY = 'insight_provider_configs'

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
}

/** Reranker 配置字段 */
interface RerankerFields extends ProviderFieldMap {
  topK: number
}

/** 服务商配置缓存结构 */
export interface ProviderConfigsCache {
  vlm: Record<string, Partial<VlmFields>>
  llm: Record<string, Partial<LlmFields>>
  embedding: Record<string, Partial<EmbeddingFields>>
  reranker: Record<string, Partial<RerankerFields>>
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
    localStorage.setItem(STORAGE_KEY, JSON.stringify(providerConfigs.value))
  }

  /**
   * 从 localStorage 加载配置缓存
   */
  function loadFromStorage(): void {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        providerConfigs.value = {
          vlm: parsed.vlm || {},
          llm: parsed.llm || {},
          embedding: parsed.embedding || {},
          reranker: parsed.reranker || {}
        }
        for (const config of Object.values(providerConfigs.value.vlm)) {
          config.openaiOptions = normalizeOpenAiOptions(config.openaiOptions, {
            forceJsonOutput: (config as Record<string, unknown>).forceJson,
            temperature: (config as Record<string, unknown>).temperature,
            extraBody: (config as Record<string, unknown>).extraBody,
            useStream: (config as Record<string, unknown>).useStream,
            rpmLimit: (config as Record<string, unknown>).rpmLimit,
            transportRetries: (config as Record<string, unknown>).transportRetries,
            businessRetries: (config as Record<string, unknown>).businessRetries
          }, {
            request: { forceJsonOutput: false, temperature: 0.3 },
            execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
          })
        }
        for (const config of Object.values(providerConfigs.value.llm)) {
          config.openaiOptions = normalizeOpenAiOptions(config.openaiOptions, {
            forceJsonOutput: (config as Record<string, unknown>).forceJson,
            extraBody: (config as Record<string, unknown>).extraBody,
            useStream: (config as Record<string, unknown>).useStream,
            rpmLimit: (config as Record<string, unknown>).rpmLimit,
            transportRetries: (config as Record<string, unknown>).transportRetries,
            businessRetries: (config as Record<string, unknown>).businessRetries
          }, {
            request: { forceJsonOutput: false },
            execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
          })
        }
      } catch (e) {
        console.error('[Insight] 加载服务商配置缓存失败:', e)
      }
    }
  }

  /**
   * 创建通用的服务商配置管理器
   */
  function createProviderManager<T extends ProviderFieldMap>(
    configType: 'vlm' | 'llm' | 'embedding' | 'reranker',
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
       * 切换服务商（先保存旧的，再恢复新的）
       */
      switch(
        oldProvider: string,
        newProvider: string,
        currentConfig: Record<string, unknown>
      ): void {
        if (oldProvider === newProvider) return
        this.save(oldProvider, currentConfig)
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
      rpmLimit: config.rpmLimit as number
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.rpmLimit !== undefined) config.rpmLimit = cached.rpmLimit
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  // Reranker 配置管理器
  const rerankerManager = createProviderManager<RerankerFields>(
    'reranker',
    (config) => ({
      apiKey: config.apiKey as string,
      model: config.model as string,
      baseUrl: config.baseUrl as string,
      topK: config.topK as number
    }),
    (config, cached) => {
      if (cached.apiKey !== undefined) config.apiKey = cached.apiKey
      if (cached.model !== undefined) config.model = cached.model
      if (cached.baseUrl !== undefined) config.baseUrl = cached.baseUrl
      if (cached.topK !== undefined) config.topK = cached.topK
    },
    { apiKey: '', model: '', baseUrl: '' }
  )

  return {
    saveToStorage,
    loadFromStorage,
    vlmManager,
    llmManager,
    embeddingManager,
    rerankerManager
  }
}
