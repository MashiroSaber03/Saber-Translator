/**
 * OCR 识别设置模块
 * 对应设置模态窗的 "OCR识别" Tab
 */

import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  TranslationSettings,
  BaiduOcrSettings,
  PaddleOcrVlSettings,
  AiVisionOcrSettings,
  HybridOcrSettings,
  OcrEngine
} from '@/types/settings'
import type { ProviderConfigsCache, AiVisionOcrProviderConfig } from '../types'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT
} from '@/constants'
import { normalizeHybridOcrConfig } from '@/utils/hybridOcr'

/**
 * 创建 OCR 设置模块
 */
export function useOcrSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void
) {
  type AiVisionOcrUiUpdates = Partial<AiVisionOcrSettings> & {
    rpmLimit?: number
    transportRetries?: number
    businessRetries?: number
    forceJsonOutput?: boolean
    useStream?: boolean
    extraBody?: Record<string, unknown>
  }
  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前OCR引擎 */
  const ocrEngine = computed(() => settings.value.ocrEngine)

  /** 当前源语言 */
  const sourceLanguage = computed(() => settings.value.sourceLanguage)

  // ============================================================
  // OCR 设置方法
  // ============================================================

  /**
   * 设置OCR引擎
   * @param engine - OCR引擎类型
   */
  function setOcrEngine(engine: OcrEngine): void {
    const normalized = normalizeHybridOcrConfig(engine, settings.value.hybridOcr)
    settings.value.ocrEngine = normalized.primaryEngine
    settings.value.hybridOcr = normalized.hybrid
    saveToStorage()
  }

  /**
   * 设置源语言
   * @param language - 源语言代码
   */
  function setSourceLanguage(language: string): void {
    settings.value.sourceLanguage = language
    saveToStorage()
  }

  /**
   * 更新百度OCR设置
   * @param updates - 要更新的设置
   */
  function updateBaiduOcr(updates: Partial<BaiduOcrSettings>): void {
    Object.assign(settings.value.baiduOcr, updates)
    saveToStorage()
  }

  /**
   * 更新PaddleOCR-VL设置
   * @param updates - 要更新的设置
   */
  function updatePaddleOcrVl(updates: Partial<PaddleOcrVlSettings>): void {
    Object.assign(settings.value.paddleOcrVl, updates)
    saveToStorage()
  }

  /**
   * 更新AI视觉OCR设置
   * @param updates - 要更新的设置
   */
  function updateAiVisionOcr(updates: AiVisionOcrUiUpdates): void {
    const {
      rpmLimit,
      transportRetries,
      businessRetries,
      forceJsonOutput,
      useStream,
      extraBody,
      ...ocrUpdates
    } = updates

    Object.assign(settings.value.aiVisionOcr, ocrUpdates)
    if (rpmLimit !== undefined) settings.value.aiVisionOcr.openaiOptions.execution.rpmLimit = rpmLimit
    if (transportRetries !== undefined) settings.value.aiVisionOcr.openaiOptions.execution.transportRetries = transportRetries
    if (businessRetries !== undefined) settings.value.aiVisionOcr.openaiOptions.execution.businessRetries = businessRetries
    if (forceJsonOutput !== undefined) {
      settings.value.aiVisionOcr.openaiOptions.request.forceJsonOutput = forceJsonOutput
      settings.value.aiVisionOcr.promptMode = forceJsonOutput ? 'json' : 'normal'
    }
    if (useStream !== undefined) settings.value.aiVisionOcr.openaiOptions.execution.useStream = useStream
    if (Object.prototype.hasOwnProperty.call(updates, 'extraBody')) {
      settings.value.aiVisionOcr.openaiOptions.request.extraBody = extraBody
    }
    saveToStorage()
  }

  /**
   * 更新混合OCR设置
   */
  function updateHybridOcr(updates: Partial<HybridOcrSettings>): void {
    const enablingHybrid = Boolean(updates.enabled) && !settings.value.hybridOcr.enabled
    const normalized = normalizeHybridOcrConfig(
      settings.value.ocrEngine,
      {
        ...settings.value.hybridOcr,
        ...updates
      },
      {
        preferRecommendedOrder: enablingHybrid
      }
    )
    settings.value.ocrEngine = normalized.primaryEngine
    settings.value.hybridOcr = normalized.hybrid
    saveToStorage()
  }

  /**
   * 设置AI视觉OCR服务商
   * @param provider - 服务商名称
   */
  function setAiVisionOcrProvider(provider: string): void {
    provider = normalizeProviderId(provider)
    const previousProvider = settings.value.aiVisionOcr.provider
    if (previousProvider === provider) return

    // 保存当前服务商配置
    saveAiVisionOcrProviderConfig(previousProvider)

    // 切换服务商
    settings.value.aiVisionOcr.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreAiVisionOcrProviderConfig(provider)

    saveToStorage()
  }

  /**
   * 设置AI视觉OCR提示词模式
   * 切换时自动更新当前提示词内容为对应模式的默认提示词
   * @param mode - 提示词模式
   */
  function setAiVisionOcrPromptMode(mode: boolean | 'normal' | 'json' | 'paddleocr_vl'): void {
    const normalizedMode = typeof mode === 'boolean' ? (mode ? 'json' : 'normal') : mode

    settings.value.aiVisionOcr.promptMode = normalizedMode
    settings.value.aiVisionOcr.openaiOptions.request.forceJsonOutput = normalizedMode === 'json'

    if (normalizedMode === 'json') {
      settings.value.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_JSON_PROMPT
    } else if (normalizedMode === 'normal') {
      settings.value.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_PROMPT
    }

    saveToStorage()
  }

  // ============================================================
  // AI视觉OCR 服务商配置缓存方法
  // ============================================================

  /**
   * 保存AI视觉OCR服务商配置到缓存
   * @param provider - 服务商名称
   */
  function saveAiVisionOcrProviderConfig(provider: string): void {
    if (!provider) return
    provider = normalizeProviderId(provider)

    const config: AiVisionOcrProviderConfig = {
      apiKey: settings.value.aiVisionOcr.apiKey,
      modelName: settings.value.aiVisionOcr.modelName,
      customBaseUrl: settings.value.aiVisionOcr.customBaseUrl,
      prompt: settings.value.aiVisionOcr.prompt,
      promptMode: settings.value.aiVisionOcr.promptMode,
      openaiOptions: JSON.parse(JSON.stringify(settings.value.aiVisionOcr.openaiOptions)),
      minImageSize: settings.value.aiVisionOcr.minImageSize
    }

    providerConfigs.value.aiVisionOcr[provider] = config
    saveProviderConfigsToStorage()
  }

  /**
   * 恢复AI视觉OCR服务商配置从缓存
   * @param provider - 服务商名称
   */
  function restoreAiVisionOcrProviderConfig(provider: string): void {
    if (!provider) return
    provider = normalizeProviderId(provider)

    const cached = providerConfigs.value.aiVisionOcr[provider]
    if (cached) {
      if (cached.apiKey !== undefined) settings.value.aiVisionOcr.apiKey = cached.apiKey
      if (cached.modelName !== undefined) settings.value.aiVisionOcr.modelName = cached.modelName
      if (cached.customBaseUrl !== undefined) settings.value.aiVisionOcr.customBaseUrl = cached.customBaseUrl
      if (cached.prompt !== undefined) settings.value.aiVisionOcr.prompt = cached.prompt
      if (cached.promptMode !== undefined) settings.value.aiVisionOcr.promptMode = cached.promptMode
      if (cached.openaiOptions !== undefined) settings.value.aiVisionOcr.openaiOptions = JSON.parse(JSON.stringify(cached.openaiOptions))
      if (cached.minImageSize !== undefined) settings.value.aiVisionOcr.minImageSize = cached.minImageSize
    } else {
      // 无缓存时清空配置
      settings.value.aiVisionOcr.apiKey = ''
      settings.value.aiVisionOcr.modelName = ''
      settings.value.aiVisionOcr.customBaseUrl = ''
    }
  }

  return {
    // 计算属性
    ocrEngine,
    sourceLanguage,

    // 方法
    setOcrEngine,
    setSourceLanguage,
    updateBaiduOcr,
    updatePaddleOcrVl,
    updateAiVisionOcr,
    updateHybridOcr,
    setAiVisionOcrProvider,
    setAiVisionOcrPromptMode,
    saveAiVisionOcrProviderConfig,
    restoreAiVisionOcrProviderConfig
  }
}
