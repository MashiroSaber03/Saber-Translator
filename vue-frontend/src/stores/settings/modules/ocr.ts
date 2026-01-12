/**
 * OCR 识别设置模块
 * 对应设置模态窗的 "OCR识别" Tab
 */

import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  BaiduOcrSettings,
  AiVisionOcrSettings,
  OcrEngine
} from '@/types/settings'
import type { ProviderConfigsCache, AiVisionOcrProviderConfig } from '../types'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT
} from '@/constants'

/**
 * 创建 OCR 设置模块
 */
export function useOcrSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void
) {
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
    settings.value.ocrEngine = engine
    saveToStorage()
    console.log(`OCR引擎已设置为: ${engine}`)
  }

  /**
   * 设置源语言
   * @param language - 源语言代码
   */
  function setSourceLanguage(language: string): void {
    settings.value.sourceLanguage = language
    saveToStorage()
    console.log(`源语言已设置为: ${language}`)
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
   * 更新AI视觉OCR设置
   * @param updates - 要更新的设置
   */
  function updateAiVisionOcr(updates: Partial<AiVisionOcrSettings>): void {
    Object.assign(settings.value.aiVisionOcr, updates)
    saveToStorage()
  }

  /**
   * 设置AI视觉OCR服务商
   * @param provider - 服务商名称
   */
  function setAiVisionOcrProvider(provider: string): void {
    const oldProvider = settings.value.aiVisionOcr.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveAiVisionOcrProviderConfig(oldProvider)

    // 切换服务商
    settings.value.aiVisionOcr.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreAiVisionOcrProviderConfig(provider)

    saveToStorage()
    console.log(`AI视觉OCR服务商已切换为: ${provider}`)
  }

  /**
   * 设置AI视觉OCR提示词模式
   * 切换时自动更新当前提示词内容为对应模式的默认提示词
   * @param isJsonMode - 是否为JSON格式模式
   */
  function setAiVisionOcrPromptMode(isJsonMode: boolean): void {
    // 更新模式状态
    settings.value.aiVisionOcr.isJsonMode = isJsonMode

    // 根据模式切换默认提示词
    const defaultPrompt = isJsonMode ? DEFAULT_AI_VISION_OCR_JSON_PROMPT : DEFAULT_AI_VISION_OCR_PROMPT
    settings.value.aiVisionOcr.prompt = defaultPrompt

    saveToStorage()
    console.log(`AI视觉OCR提示词模式已切换为: ${isJsonMode ? 'JSON格式' : '普通模式'}`)
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

    const config: AiVisionOcrProviderConfig = {
      apiKey: settings.value.aiVisionOcr.apiKey,
      modelName: settings.value.aiVisionOcr.modelName,
      customBaseUrl: settings.value.aiVisionOcr.customBaseUrl,
      prompt: settings.value.aiVisionOcr.prompt,
      rpmLimit: settings.value.aiVisionOcr.rpmLimit,
      isJsonMode: settings.value.aiVisionOcr.isJsonMode,
      minImageSize: settings.value.aiVisionOcr.minImageSize
    }

    providerConfigs.value.aiVisionOcr[provider] = config
    saveProviderConfigsToStorage()
    console.log(`[Settings] 保存AI视觉OCR服务商配置: ${provider}`, config)
  }

  /**
   * 恢复AI视觉OCR服务商配置从缓存
   * @param provider - 服务商名称
   */
  function restoreAiVisionOcrProviderConfig(provider: string): void {
    if (!provider) return

    const cached = providerConfigs.value.aiVisionOcr[provider]
    if (cached) {
      if (cached.apiKey !== undefined) settings.value.aiVisionOcr.apiKey = cached.apiKey
      if (cached.modelName !== undefined) settings.value.aiVisionOcr.modelName = cached.modelName
      if (cached.customBaseUrl !== undefined) settings.value.aiVisionOcr.customBaseUrl = cached.customBaseUrl
      if (cached.prompt !== undefined) settings.value.aiVisionOcr.prompt = cached.prompt
      if (cached.rpmLimit !== undefined) settings.value.aiVisionOcr.rpmLimit = cached.rpmLimit
      if (cached.isJsonMode !== undefined) settings.value.aiVisionOcr.isJsonMode = cached.isJsonMode
      if (cached.minImageSize !== undefined) settings.value.aiVisionOcr.minImageSize = cached.minImageSize
      console.log(`[Settings] 恢复AI视觉OCR服务商配置: ${provider}`, cached)
    } else {
      // 无缓存时清空配置
      settings.value.aiVisionOcr.apiKey = ''
      settings.value.aiVisionOcr.modelName = ''
      settings.value.aiVisionOcr.customBaseUrl = ''
      console.log(`[Settings] ${provider} 无缓存配置，使用默认值`)
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
    updateAiVisionOcr,
    setAiVisionOcrProvider,
    setAiVisionOcrPromptMode,
    saveAiVisionOcrProviderConfig,
    restoreAiVisionOcrProviderConfig
  }
}
