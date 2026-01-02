/**
 * 高质量翻译设置模块
 * 对应设置模态窗的 "高质量翻译" Tab
 */

import { computed, type Ref } from 'vue'
import type {
  TranslationSettings,
  HqTranslationSettings,
  HqTranslationProvider,
  NoThinkingMethod
} from '@/types/settings'
import type { ProviderConfigsCache, HqTranslationProviderConfig } from '../types'

/**
 * 创建高质量翻译设置模块
 */
export function useHqTranslationSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void
) {
  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前高质量翻译服务商 */
  const hqProvider = computed(() => settings.value.hqTranslation.provider)

  // ============================================================
  // 高质量翻译设置方法
  // ============================================================

  /**
   * 设置高质量翻译服务商
   * @param provider - 服务商类型
   */
  function setHqProvider(provider: HqTranslationProvider): void {
    const oldProvider = settings.value.hqTranslation.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveHqProviderConfig(oldProvider)

    // 切换服务商
    settings.value.hqTranslation.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreHqProviderConfig(provider)

    saveToStorage()
    console.log(`高质量翻译服务商已切换为: ${provider}`)
  }

  /**
   * 更新高质量翻译设置
   * @param updates - 要更新的设置
   */
  function updateHqTranslation(updates: Partial<HqTranslationSettings>): void {
    Object.assign(settings.value.hqTranslation, updates)
    saveToStorage()
  }

  /**
   * 设置高质量翻译流式调用开关
   * @param useStream - 是否使用流式调用
   */
  function setHqUseStream(useStream: boolean): void {
    settings.value.hqTranslation.useStream = useStream
    saveToStorage()
  }

  /**
   * 设置高质量翻译取消思考方法
   * @param method - 取消思考方法
   */
  function setHqNoThinkingMethod(method: NoThinkingMethod): void {
    settings.value.hqTranslation.noThinkingMethod = method
    saveToStorage()
  }

  /**
   * 设置高质量翻译强制JSON输出
   * @param forceJson - 是否强制JSON输出
   */
  function setHqForceJsonOutput(forceJson: boolean): void {
    settings.value.hqTranslation.forceJsonOutput = forceJson
    saveToStorage()
  }

  // ============================================================
  // 高质量翻译服务商配置缓存方法
  // ============================================================

  /**
   * 保存高质量翻译服务商配置到缓存
   * @param provider - 服务商名称
   */
  function saveHqProviderConfig(provider: string): void {
    if (!provider) return

    const config: HqTranslationProviderConfig = {
      apiKey: settings.value.hqTranslation.apiKey,
      modelName: settings.value.hqTranslation.modelName,
      customBaseUrl: settings.value.hqTranslation.customBaseUrl,
      batchSize: settings.value.hqTranslation.batchSize,
      sessionReset: settings.value.hqTranslation.sessionReset,
      rpmLimit: settings.value.hqTranslation.rpmLimit,
      maxRetries: settings.value.hqTranslation.maxRetries,
      lowReasoning: settings.value.hqTranslation.lowReasoning,
      noThinkingMethod: settings.value.hqTranslation.noThinkingMethod,
      forceJsonOutput: settings.value.hqTranslation.forceJsonOutput,
      useStream: settings.value.hqTranslation.useStream,
      prompt: settings.value.hqTranslation.prompt
    }

    providerConfigs.value.hqTranslation[provider] = config
    saveProviderConfigsToStorage()
    console.log(`[Settings] 保存高质量翻译服务商配置: ${provider}`, config)
  }

  /**
   * 恢复高质量翻译服务商配置从缓存
   * @param provider - 服务商名称
   */
  function restoreHqProviderConfig(provider: string): void {
    if (!provider) return

    const cached = providerConfigs.value.hqTranslation[provider]
    if (cached) {
      if (cached.apiKey !== undefined) settings.value.hqTranslation.apiKey = cached.apiKey
      if (cached.modelName !== undefined) settings.value.hqTranslation.modelName = cached.modelName
      if (cached.customBaseUrl !== undefined) settings.value.hqTranslation.customBaseUrl = cached.customBaseUrl
      if (cached.batchSize !== undefined) settings.value.hqTranslation.batchSize = cached.batchSize
      if (cached.sessionReset !== undefined) settings.value.hqTranslation.sessionReset = cached.sessionReset
      if (cached.rpmLimit !== undefined) settings.value.hqTranslation.rpmLimit = cached.rpmLimit
      if (cached.maxRetries !== undefined) settings.value.hqTranslation.maxRetries = cached.maxRetries
      if (cached.lowReasoning !== undefined) settings.value.hqTranslation.lowReasoning = cached.lowReasoning
      if (cached.noThinkingMethod !== undefined) settings.value.hqTranslation.noThinkingMethod = cached.noThinkingMethod
      if (cached.forceJsonOutput !== undefined) settings.value.hqTranslation.forceJsonOutput = cached.forceJsonOutput
      if (cached.useStream !== undefined) settings.value.hqTranslation.useStream = cached.useStream
      if (cached.prompt !== undefined) settings.value.hqTranslation.prompt = cached.prompt
      console.log(`[Settings] 恢复高质量翻译服务商配置: ${provider}`, cached)
    } else {
      // 无缓存时清空配置
      settings.value.hqTranslation.apiKey = ''
      settings.value.hqTranslation.modelName = ''
      settings.value.hqTranslation.customBaseUrl = ''
      console.log(`[Settings] ${provider} 无缓存配置，使用默认值`)
    }
  }

  return {
    // 计算属性
    hqProvider,

    // 方法
    setHqProvider,
    updateHqTranslation,
    setHqUseStream,
    setHqNoThinkingMethod,
    setHqForceJsonOutput,
    saveHqProviderConfig,
    restoreHqProviderConfig
  }
}
