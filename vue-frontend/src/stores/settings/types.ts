/**
 * Settings Store 类型定义
 * 包含所有设置模块共享的类型定义
 */

import type { NoThinkingMethod } from '@/types/settings'

// ============================================================
// 服务商配置缓存类型定义
// ============================================================

/** 翻译服务配置缓存项 */
export interface TranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  rpmLimit?: number
  maxRetries?: number
  isJsonMode?: boolean
  translationMode?: 'batch' | 'single'
}

/** 高质量翻译服务配置缓存项 */
export interface HqTranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  batchSize?: number
  sessionReset?: number
  rpmLimit?: number
  maxRetries?: number
  lowReasoning?: boolean
  noThinkingMethod?: NoThinkingMethod
  forceJsonOutput?: boolean
  useStream?: boolean
  prompt?: string
}

/** AI视觉OCR服务配置缓存项 */
export interface AiVisionOcrProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  prompt?: string
  rpmLimit?: number
  isJsonMode?: boolean
  /** 最小图片尺寸 */
  minImageSize?: number
}

/** 服务商配置缓存结构 */
export interface ProviderConfigsCache {
  translation: Record<string, TranslationProviderConfig>
  hqTranslation: Record<string, HqTranslationProviderConfig>
  aiVisionOcr: Record<string, AiVisionOcrProviderConfig>
}
