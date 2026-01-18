/**
 * 翻译服务设置模块
 * 对应设置模态窗的 "翻译服务" Tab
 */

import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  TranslationServiceSettings,
  TranslationProvider
} from '@/types/settings'
import type { ProviderConfigsCache, TranslationProviderConfig } from '../types'

/**
 * 创建翻译服务设置模块
 */
export function useTranslationSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void
) {
  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前翻译服务商 */
  const translationProvider = computed(() => settings.value.translation.provider)

  // ============================================================
  // 翻译服务设置方法
  // ============================================================

  /**
   * 设置翻译服务商
   * @param provider - 服务商类型
   */
  function setTranslationProvider(provider: TranslationProvider): void {
    const oldProvider = settings.value.translation.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveTranslationProviderConfig(oldProvider)

    // 切换服务商
    settings.value.translation.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreTranslationProviderConfig(provider)

    saveToStorage()
  }

  /**
   * 更新翻译服务设置
   * @param updates - 要更新的设置
   */
  function updateTranslationService(updates: Partial<TranslationServiceSettings>): void {
    Object.assign(settings.value.translation, updates)
    saveToStorage()
  }

  /**
   * 设置翻译提示词
   * @param prompt - 提示词内容
   */
  function setTranslatePrompt(prompt: string): void {
    settings.value.translatePrompt = prompt
    saveToStorage()
  }

  /**
   * 设置翻译提示词模式
   * 切换时从对应的存储字段加载提示词（不会丢失用户修改）
   * @param isJsonMode - 是否为JSON格式模式
   */
  function setTranslatePromptMode(isJsonMode: boolean): void {
    // 更新模式状态
    settings.value.translation.isJsonMode = isJsonMode

    // 根据翻译模式和提示词模式，从对应的存储字段加载提示词
    const isSingleMode = settings.value.translation.translationMode === 'single'
    let prompt: string
    if (isSingleMode) {
      prompt = isJsonMode
        ? settings.value.translation.singleJsonPrompt
        : settings.value.translation.singleNormalPrompt
    } else {
      prompt = isJsonMode
        ? settings.value.translation.batchJsonPrompt
        : settings.value.translation.batchNormalPrompt
    }
    settings.value.translatePrompt = prompt

    saveToStorage()
    console.log(`翻译提示词模式已切换为: ${isJsonMode ? 'JSON格式' : '普通模式'}`)
  }

  // ============================================================
  // 翻译服务商配置缓存方法
  // ============================================================

  /**
   * 保存翻译服务商配置到缓存
   * @param provider - 服务商名称
   */
  function saveTranslationProviderConfig(provider: string): void {
    if (!provider) return

    const config: TranslationProviderConfig = {
      apiKey: settings.value.translation.apiKey,
      modelName: settings.value.translation.modelName,
      customBaseUrl: settings.value.translation.customBaseUrl,
      rpmLimit: settings.value.translation.rpmLimit,
      maxRetries: settings.value.translation.maxRetries,
      isJsonMode: settings.value.translation.isJsonMode,
      translationMode: settings.value.translation.translationMode
    }

    providerConfigs.value.translation[provider] = config
    saveProviderConfigsToStorage()
    console.log(`[Settings] 保存翻译服务商配置: ${provider}`, config)
  }

  /**
   * 恢复翻译服务商配置从缓存
   * @param provider - 服务商名称
   */
  function restoreTranslationProviderConfig(provider: string): void {
    if (!provider) return

    const cached = providerConfigs.value.translation[provider]
    if (cached) {
      // 恢复缓存的配置
      if (cached.apiKey !== undefined) settings.value.translation.apiKey = cached.apiKey
      if (cached.modelName !== undefined) settings.value.translation.modelName = cached.modelName
      if (cached.customBaseUrl !== undefined) settings.value.translation.customBaseUrl = cached.customBaseUrl
      if (cached.rpmLimit !== undefined) settings.value.translation.rpmLimit = cached.rpmLimit
      if (cached.maxRetries !== undefined) settings.value.translation.maxRetries = cached.maxRetries
      if (cached.isJsonMode !== undefined) settings.value.translation.isJsonMode = cached.isJsonMode
      if (cached.translationMode !== undefined) settings.value.translation.translationMode = cached.translationMode
      console.log(`[Settings] 恢复翻译服务商配置: ${provider}`, cached)
    } else {
      // 无缓存时清空配置（保留默认值）
      settings.value.translation.apiKey = ''
      settings.value.translation.modelName = ''
      settings.value.translation.customBaseUrl = ''
      console.log(`[Settings] ${provider} 无缓存配置，使用默认值`)
    }
  }

  return {
    // 计算属性
    translationProvider,

    // 方法
    setTranslationProvider,
    updateTranslationService,
    setTranslatePrompt,
    setTranslatePromptMode,
    saveTranslationProviderConfig,
    restoreTranslationProviderConfig
  }
}
