/**
 * Settings Store 类型定义
 * 包含所有设置模块共享的类型定义
 */

// ============================================================
// 服务商配置缓存类型定义
// ============================================================

/** 翻译服务配置缓存项 */
export interface TranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  openaiOptions?: {
    request?: {
      forceJsonOutput?: boolean
    }
    execution?: {
      useStream?: boolean
      rpmLimit?: number
      transportRetries?: number
      businessRetries?: number
    }
  }
  translationMode?: 'batch' | 'single'
}

/** 高质量翻译服务配置缓存项 */
export interface HqTranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  batchSize?: number
  openaiOptions?: {
    request?: {
      forceJsonOutput?: boolean
      temperature?: number
    }
    execution?: {
      useStream?: boolean
      rpmLimit?: number
      transportRetries?: number
      businessRetries?: number
    }
  }
  prompt?: string
}

/** AI视觉OCR服务配置缓存项 */
export interface AiVisionOcrProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  prompt?: string
  promptMode?: 'normal' | 'json' | 'paddleocr_vl'
  openaiOptions?: {
    request?: {
      forceJsonOutput?: boolean
    }
    execution?: {
      useStream?: boolean
      rpmLimit?: number
      transportRetries?: number
      businessRetries?: number
    }
  }
  /** 最小图片尺寸 */
  minImageSize?: number
}

/** 服务商配置缓存结构 */
export interface ProviderConfigsCache {
  translation: Record<string, TranslationProviderConfig>
  hqTranslation: Record<string, HqTranslationProviderConfig>
  aiVisionOcr: Record<string, AiVisionOcrProviderConfig>
}
