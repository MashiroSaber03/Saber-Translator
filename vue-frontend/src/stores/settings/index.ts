/**
 * 设置状态管理 Store（模块化重构版）
 * 管理翻译设置、OCR设置、高质量翻译设置、AI校对设置等
 * 支持 localStorage 持久化
 * 
 * 模块结构：
 * - ocr.ts: OCR识别设置
 * - translation.ts: 翻译服务设置
 * - detection.ts: 检测设置
 * - hqTranslation.ts: 高质量翻译设置
 * - proofreading.ts: AI校对设置
 * - prompts.ts: 提示词管理
 * - misc.ts: 更多设置（PDF、调试、文字样式等）
 * - modelHistory.ts: 模型历史记录
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  TranslationSettings,
  OcrEngine,
  TextDetector,
  TranslationProvider,
  HqTranslationProvider,
  NoThinkingMethod
} from '@/types/settings'
import {
  STORAGE_KEY_TRANSLATION_SETTINGS,
  STORAGE_KEY_PROVIDER_CONFIGS,
  DEFAULT_RPM_TRANSLATION,
  DEFAULT_RPM_AI_VISION_OCR,
  DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
  DEFAULT_TRANSLATION_MAX_RETRIES,
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_MAX_RETRIES
} from '@/constants'

import type { ProviderConfigsCache } from './types'
import { createDefaultSettings } from './defaults'

// 导入各功能模块
import {
  useOcrSettings,
  useTranslationSettings,
  useDetectionSettings,
  useHqTranslationSettings,
  useProofreadingSettings,
  usePromptsSettings,
  useMiscSettings,
  useModelHistory
} from './modules'

// ============================================================
// Store 定义
// ============================================================

export const useSettingsStore = defineStore('settings', () => {
  // ============================================================
  // 核心状态定义
  // ============================================================

  /** 翻译设置 */
  const settings = ref<TranslationSettings>(createDefaultSettings())

  /** 服务商配置分组存储（用于切换服务商时保存/恢复配置） */
  const providerConfigs = ref<ProviderConfigsCache>({
    translation: {},
    hqTranslation: {},
    aiVisionOcr: {}
  })

  // ============================================================
  // localStorage 持久化方法
  // ============================================================

  /**
   * 保存设置到 localStorage
   */
  function saveToStorage(): void {
    try {
      const data = JSON.stringify(settings.value)
      localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, data)
    } catch (error) {
      console.error('保存设置到 localStorage 失败:', error)
    }
  }

  /**
   * 保存服务商配置缓存到 localStorage
   */
  function saveProviderConfigsToStorage(): void {
    try {
      const data = JSON.stringify(providerConfigs.value)
      localStorage.setItem(STORAGE_KEY_PROVIDER_CONFIGS, data)
    } catch (error) {
      console.error('保存服务商配置缓存失败:', error)
    }
  }

  /** 原版 localStorage 存储键（用于兼容迁移） */
  const LEGACY_STORAGE_KEY = 'saber_translator_settings'

  /**
   * 从 localStorage 加载设置
   * 优先读取新 Key，若不存在则尝试读取原版 Key 并迁移
   * 【复刻原版】textStyle（左侧边栏文字设置）不从 localStorage 加载，始终使用默认值
   */
  function loadFromStorage(): void {
    try {
      let data = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)

      // 如果新 Key 不存在，尝试读取原版 Key（兼容迁移）
      if (!data) {
        const legacyData = localStorage.getItem(LEGACY_STORAGE_KEY)
        if (legacyData) {
          console.log('[Settings] 检测到原版设置，正在迁移...')
          data = legacyData
          // 迁移到新 Key
          localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, legacyData)
          console.log('[Settings] 原版设置已迁移到新存储键')
        }
      }

      if (data) {
        const parsed = JSON.parse(data)
        const defaults = createDefaultSettings()
        // 深度合并，确保新增的默认值不会丢失
        settings.value = deepMerge(defaults, parsed)
        // 确保数值类型正确
        ensureNumericTypes()

        // 【复刻原版】左侧边栏文字设置始终使用默认值，不从 localStorage 恢复
        settings.value.textStyle = { ...defaults.textStyle }

        console.log('已从 localStorage 加载设置（textStyle 使用默认值）')
      }
    } catch (error) {
      console.error('从 localStorage 加载设置失败:', error)
    }
  }

  /**
   * 从 localStorage 加载服务商配置缓存
   */
  function loadProviderConfigsFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_PROVIDER_CONFIGS)
      if (data) {
        const parsed = JSON.parse(data)
        // 确保结构完整
        providerConfigs.value = {
          translation: parsed.translation || {},
          hqTranslation: parsed.hqTranslation || {},
          aiVisionOcr: parsed.aiVisionOcr || {}
        }
        console.log('已从 localStorage 加载服务商配置缓存')
      }
    } catch (error) {
      console.error('加载服务商配置缓存失败:', error)
    }
  }

  /**
   * 确保设置中的数值类型正确
   * 注意：textStyle 不在这里处理，因为它会被重置为默认值（复刻原版行为）
   */
  function ensureNumericTypes(): void {
    const be = settings.value.boxExpand
    be.ratio = Number(be.ratio) || 1.0
    be.top = Number(be.top) || 0
    be.bottom = Number(be.bottom) || 0
    be.left = Number(be.left) || 0
    be.right = Number(be.right) || 0

    const pm = settings.value.preciseMask
    pm.dilateSize = Number(pm.dilateSize) || 5
    pm.boxExpandRatio = Number(pm.boxExpandRatio) || 1.0

    const tr = settings.value.translation
    tr.rpmLimit = Number(tr.rpmLimit) || DEFAULT_RPM_TRANSLATION
    tr.maxRetries = Number(tr.maxRetries) || DEFAULT_TRANSLATION_MAX_RETRIES

    const hq = settings.value.hqTranslation
    hq.batchSize = Number(hq.batchSize) || 10
    hq.sessionReset = Number(hq.sessionReset) || 50
    hq.rpmLimit = Number(hq.rpmLimit) || 60
    hq.maxRetries = Number(hq.maxRetries) || DEFAULT_HQ_TRANSLATION_MAX_RETRIES

    const av = settings.value.aiVisionOcr
    av.rpmLimit = Number(av.rpmLimit) || DEFAULT_RPM_AI_VISION_OCR
    // 对于 minImageSize，0 是合法值（表示禁用自动放大），所以不能用 || 操作符
    if (av.minImageSize === undefined || av.minImageSize === null || isNaN(Number(av.minImageSize))) {
      av.minImageSize = DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE
    } else {
      av.minImageSize = Number(av.minImageSize)
    }

    const pr = settings.value.proofreading
    pr.maxRetries = Number(pr.maxRetries) || DEFAULT_PROOFREADING_MAX_RETRIES

    // 迁移旧版服务商名称
    if ((tr.provider as string) === 'baidu') {
      tr.provider = 'baidu_translate'
    }
    if ((tr.provider as string) === 'youdao') {
      tr.provider = 'youdao_translate'
    }

    // 迁移缓存的配置
    if (providerConfigs.value.translation['baidu']) {
      providerConfigs.value.translation['baidu_translate'] = { ...providerConfigs.value.translation['baidu'] }
      delete providerConfigs.value.translation['baidu']
    }
    if (providerConfigs.value.translation['youdao']) {
      providerConfigs.value.translation['youdao_translate'] = { ...providerConfigs.value.translation['youdao'] }
      delete providerConfigs.value.translation['youdao']
    }
  }

  /**
   * 深度合并对象
   */
  function deepMerge(
    target: TranslationSettings,
    source: Partial<TranslationSettings>
  ): TranslationSettings {
    const result = { ...target }
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        const k = key as keyof TranslationSettings
        const sourceValue = source[k]
        const targetValue = result[k]
        if (
          sourceValue !== null &&
          sourceValue !== undefined &&
          typeof sourceValue === 'object' &&
          !Array.isArray(sourceValue) &&
          targetValue !== null &&
          typeof targetValue === 'object' &&
          !Array.isArray(targetValue)
        ) {
          ; (result as Record<string, unknown>)[k] = {
            ...(targetValue as unknown as Record<string, unknown>),
            ...(sourceValue as unknown as Record<string, unknown>)
          }
        } else if (sourceValue !== undefined) {
          ; (result as Record<string, unknown>)[k] = sourceValue
        }
      }
    }
    return result
  }

  // ============================================================
  // 初始化各功能模块
  // ============================================================

  // OCR 设置模块
  const ocrModule = useOcrSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage
  )

  // 翻译服务设置模块
  const translationModule = useTranslationSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage
  )

  // 检测设置模块
  const detectionModule = useDetectionSettings(settings, saveToStorage)

  // 高质量翻译设置模块
  const hqTranslationModule = useHqTranslationSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage
  )

  // AI校对设置模块
  const proofreadingModule = useProofreadingSettings(settings, saveToStorage)

  // 提示词管理模块
  const promptsModule = usePromptsSettings(settings, saveToStorage)

  // 更多设置模块
  const miscModule = useMiscSettings(settings, saveToStorage)

  // 模型历史记录模块
  const modelHistoryModule = useModelHistory()

  // ============================================================
  // 兼容旧接口的方法
  // ============================================================

  /**
   * 保存服务商配置（兼容旧接口）
   */
  function saveProviderConfig(category: string, provider: string): void {
    if (category === 'translation') {
      translationModule.saveTranslationProviderConfig(provider)
    } else if (category === 'hqTranslation') {
      hqTranslationModule.saveHqProviderConfig(provider)
    } else if (category === 'aiVisionOcr') {
      ocrModule.saveAiVisionOcrProviderConfig(provider)
    }
  }

  /**
   * 恢复服务商配置（兼容旧接口）
   */
  function restoreProviderConfig(category: string, provider: string): void {
    if (category === 'translation') {
      translationModule.restoreTranslationProviderConfig(provider)
    } else if (category === 'hqTranslation') {
      hqTranslationModule.restoreHqProviderConfig(provider)
    } else if (category === 'aiVisionOcr') {
      ocrModule.restoreAiVisionOcrProviderConfig(provider)
    }
  }

  // ============================================================
  // 初始化方法
  // ============================================================

  /**
   * 清理旧版本的主题设置（兼容性处理）
   */
  function cleanupLegacyThemeSettings(): void {
    try {
      localStorage.removeItem('theme')
      if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-theme', 'light')
        document.body.setAttribute('data-theme', 'light')
      }
    } catch (error) {
      console.warn('清理旧版主题设置失败:', error)
    }
  }

  /**
   * 初始化设置（从 localStorage 加载）
   */
  function initSettings(): void {
    cleanupLegacyThemeSettings()
    loadFromStorage()
    modelHistoryModule.loadCustomFontPresetsFromStorage()
    loadProviderConfigsFromStorage()
    modelHistoryModule.loadModelHistoryFromStorage()
  }

  /**
   * 重置所有设置为默认值
   */
  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
    saveToStorage()
    console.log('设置已重置为默认值')
  }

  // ============================================================
  // 后端同步方法（保持原有实现，导入较长，单独放在这里）
  // ============================================================

  /**
   * 从后端加载用户设置
   */
  async function loadFromBackend(): Promise<boolean> {
    try {
      console.log('[Settings] 开始从后端加载设置...')
      const { getUserSettings } = await import('@/api/config')
      const response = await getUserSettings()

      if (response.success && response.settings) {
        const backendSettings = response.settings
        console.log('[Settings] 从后端加载设置:', backendSettings)
        applyBackendSettings(backendSettings)

        // 【复刻原版】左侧边栏文字设置始终使用默认值，不从后端恢复
        const defaults = createDefaultSettings()
        settings.value.textStyle = { ...defaults.textStyle }

        saveToStorage()
        saveProviderConfigsToStorage()
        console.log('[Settings] 后端设置已应用（textStyle 使用默认值）')
        return true
      } else {
        console.warn('[Settings] 后端无设置数据，使用 localStorage 或默认值')
        return false
      }
    } catch (error) {
      console.error('[Settings] 从后端加载设置失败:', error)
      return false
    }
  }

  /**
   * 将后端设置应用到当前设置
   */
  function applyBackendSettings(backendSettings: Record<string, unknown>): void {
    const parseNum = (val: unknown, defaultVal: number): number => {
      if (val === undefined || val === null || val === '') return defaultVal
      const num = typeof val === 'string' ? parseFloat(val) : Number(val)
      return isNaN(num) ? defaultVal : num
    }

    // OCR 设置
    if (backendSettings.ocrEngine) {
      settings.value.ocrEngine = backendSettings.ocrEngine as OcrEngine
    }
    if (backendSettings.sourceLanguage) {
      settings.value.sourceLanguage = backendSettings.sourceLanguage as string
    }
    if (backendSettings.textDetector) {
      settings.value.textDetector = backendSettings.textDetector as TextDetector
    }

    // 百度 OCR 设置
    if (backendSettings.baiduApiKey) {
      settings.value.baiduOcr.apiKey = backendSettings.baiduApiKey as string
    }
    if (backendSettings.baiduSecretKey) {
      settings.value.baiduOcr.secretKey = backendSettings.baiduSecretKey as string
    }
    if (backendSettings.baiduVersion) {
      settings.value.baiduOcr.version = backendSettings.baiduVersion as string
    }
    if (backendSettings.baiduSourceLanguage) {
      settings.value.baiduOcr.sourceLanguage = backendSettings.baiduSourceLanguage as string
    }

    // AI 视觉 OCR 设置
    if (backendSettings.aiVisionProvider) {
      settings.value.aiVisionOcr.provider = backendSettings.aiVisionProvider as string
    }
    if (backendSettings.aiVisionApiKey) {
      settings.value.aiVisionOcr.apiKey = backendSettings.aiVisionApiKey as string
    }
    if (backendSettings.aiVisionModelName) {
      settings.value.aiVisionOcr.modelName = backendSettings.aiVisionModelName as string
    }
    if (backendSettings.aiVisionOcrPrompt) {
      settings.value.aiVisionOcr.prompt = backendSettings.aiVisionOcrPrompt as string
    }
    if (backendSettings.customAiVisionBaseUrl) {
      settings.value.aiVisionOcr.customBaseUrl = backendSettings.customAiVisionBaseUrl as string
    }
    if (backendSettings.rpmAiVisionOcr !== undefined) {
      settings.value.aiVisionOcr.rpmLimit = parseNum(backendSettings.rpmAiVisionOcr, DEFAULT_RPM_AI_VISION_OCR)
    }
    if (backendSettings.aiVisionPromptModeSelect === 'json') {
      settings.value.aiVisionOcr.isJsonMode = true
    }
    if (backendSettings.aiVisionMinImageSize !== undefined) {
      settings.value.aiVisionOcr.minImageSize = parseNum(backendSettings.aiVisionMinImageSize, DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE)
    }

    // 翻译服务设置
    if (backendSettings.modelProvider) {
      settings.value.translation.provider = backendSettings.modelProvider as TranslationProvider
    }
    if (backendSettings.apiKey) {
      settings.value.translation.apiKey = backendSettings.apiKey as string
    }
    if (backendSettings.modelName) {
      settings.value.translation.modelName = backendSettings.modelName as string
    }
    if (backendSettings.customBaseUrl) {
      settings.value.translation.customBaseUrl = backendSettings.customBaseUrl as string
    }
    if (backendSettings.rpmTranslation !== undefined) {
      settings.value.translation.rpmLimit = parseNum(backendSettings.rpmTranslation, DEFAULT_RPM_TRANSLATION)
    }
    if (backendSettings.translationMaxRetries !== undefined) {
      settings.value.translation.maxRetries = parseNum(backendSettings.translationMaxRetries, DEFAULT_TRANSLATION_MAX_RETRIES)
    }
    if (backendSettings.translatePromptModeSelect === 'json') {
      settings.value.translation.isJsonMode = true
    }

    // 翻译提示词
    if (backendSettings.promptContent) {
      settings.value.translatePrompt = backendSettings.promptContent as string
    }
    if (backendSettings.enableTextboxPrompt !== undefined) {
      settings.value.useTextboxPrompt = backendSettings.enableTextboxPrompt as boolean
    }
    if (backendSettings.textboxPromptContent) {
      settings.value.textboxPrompt = backendSettings.textboxPromptContent as string
    }

    // 高质量翻译设置
    if (backendSettings.hqTranslateProvider) {
      settings.value.hqTranslation.provider = backendSettings.hqTranslateProvider as HqTranslationProvider
    }
    if (backendSettings.hqApiKey) {
      settings.value.hqTranslation.apiKey = backendSettings.hqApiKey as string
    }
    if (backendSettings.hqModelName) {
      settings.value.hqTranslation.modelName = backendSettings.hqModelName as string
    }
    if (backendSettings.hqCustomBaseUrl) {
      settings.value.hqTranslation.customBaseUrl = backendSettings.hqCustomBaseUrl as string
    }
    if (backendSettings.hqBatchSize !== undefined) {
      settings.value.hqTranslation.batchSize = parseNum(backendSettings.hqBatchSize, 3)
    }
    if (backendSettings.hqSessionReset !== undefined) {
      settings.value.hqTranslation.sessionReset = parseNum(backendSettings.hqSessionReset, 20)
    }
    if (backendSettings.hqRpmLimit !== undefined) {
      settings.value.hqTranslation.rpmLimit = parseNum(backendSettings.hqRpmLimit, 7)
    }
    if (backendSettings.hqMaxRetries !== undefined) {
      settings.value.hqTranslation.maxRetries = parseNum(backendSettings.hqMaxRetries, DEFAULT_HQ_TRANSLATION_MAX_RETRIES)
    }
    if (backendSettings.hqPrompt) {
      settings.value.hqTranslation.prompt = backendSettings.hqPrompt as string
    }
    if (backendSettings.hqLowReasoning !== undefined) {
      settings.value.hqTranslation.lowReasoning = backendSettings.hqLowReasoning as boolean
    }
    if (backendSettings.hqNoThinkingMethod) {
      settings.value.hqTranslation.noThinkingMethod = backendSettings.hqNoThinkingMethod as NoThinkingMethod
    }
    if (backendSettings.hqForceJsonOutput !== undefined) {
      settings.value.hqTranslation.forceJsonOutput = backendSettings.hqForceJsonOutput as boolean
    }
    if (backendSettings.hqUseStream !== undefined) {
      settings.value.hqTranslation.useStream = backendSettings.hqUseStream as boolean
    }

    // AI 校对设置
    if (backendSettings.proofreadingEnabled !== undefined) {
      settings.value.proofreading.enabled = backendSettings.proofreadingEnabled as boolean
    }
    if (backendSettings.proofreadingMaxRetries !== undefined) {
      settings.value.proofreading.maxRetries = parseNum(backendSettings.proofreadingMaxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
    }
    if (backendSettings.proofreading && typeof backendSettings.proofreading === 'object') {
      const proofConfig = backendSettings.proofreading as Record<string, unknown>
      if (proofConfig.enabled !== undefined) {
        settings.value.proofreading.enabled = proofConfig.enabled as boolean
      }
      if (proofConfig.maxRetries !== undefined) {
        settings.value.proofreading.maxRetries = parseNum(proofConfig.maxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
      }
      if (Array.isArray(proofConfig.rounds)) {
        settings.value.proofreading.rounds = proofConfig.rounds.map((round: Record<string, unknown>) => ({
          name: (round.name as string) || '轮次',
          provider: ((round.provider as string) || 'siliconflow') as HqTranslationProvider,
          apiKey: (round.apiKey as string) || '',
          modelName: (round.modelName as string) || '',
          customBaseUrl: (round.customBaseUrl as string) || '',
          prompt: (round.prompt as string) || '',
          batchSize: parseNum(round.batchSize, 3),
          sessionReset: parseNum(round.sessionReset, 20),
          rpmLimit: parseNum(round.rpmLimit, 7),
          maxRetries: parseNum(round.maxRetries, DEFAULT_PROOFREADING_MAX_RETRIES),
          lowReasoning: (round.lowReasoning as boolean) || false,
          noThinkingMethod: ((round.noThinkingMethod as string) || 'gemini') as NoThinkingMethod,
          forceJsonOutput: (round.forceJsonOutput as boolean) || false
        }))
      }
    }

    // 文本框扩展设置
    if (backendSettings.boxExpandRatio !== undefined) {
      settings.value.boxExpand.ratio = parseNum(backendSettings.boxExpandRatio, 0)
    }
    if (backendSettings.boxExpandTop !== undefined) {
      settings.value.boxExpand.top = parseNum(backendSettings.boxExpandTop, 0)
    }
    if (backendSettings.boxExpandBottom !== undefined) {
      settings.value.boxExpand.bottom = parseNum(backendSettings.boxExpandBottom, 0)
    }
    if (backendSettings.boxExpandLeft !== undefined) {
      settings.value.boxExpand.left = parseNum(backendSettings.boxExpandLeft, 0)
    }
    if (backendSettings.boxExpandRight !== undefined) {
      settings.value.boxExpand.right = parseNum(backendSettings.boxExpandRight, 0)
    }

    // 精确掩膜设置
    if (backendSettings.usePreciseMask !== undefined) {
      settings.value.preciseMask.enabled = backendSettings.usePreciseMask as boolean
    }
    if (backendSettings.maskDilateSize !== undefined) {
      settings.value.preciseMask.dilateSize = parseNum(backendSettings.maskDilateSize, 5)
    }
    if (backendSettings.maskBoxExpandRatio !== undefined) {
      settings.value.preciseMask.boxExpandRatio = parseNum(backendSettings.maskBoxExpandRatio, 0)
    }

    // PDF 处理方式
    if (backendSettings.pdfProcessingMethod) {
      settings.value.pdfProcessingMethod = backendSettings.pdfProcessingMethod as 'backend' | 'frontend'
    }

    // 调试设置
    if (backendSettings.showDetectionDebug !== undefined) {
      settings.value.showDetectionDebug = backendSettings.showDetectionDebug as boolean
    }

    // 并行翻译设置
    if (backendSettings.parallelEnabled !== undefined) {
      settings.value.parallel.enabled = backendSettings.parallelEnabled as boolean
    }
    if (backendSettings.parallelDeepLearningLockSize !== undefined) {
      settings.value.parallel.deepLearningLockSize = parseNum(backendSettings.parallelDeepLearningLockSize, 1)
    }

    // 服务商配置缓存
    if (backendSettings.providerSettings && typeof backendSettings.providerSettings === 'object') {
      const providerSettings = backendSettings.providerSettings as Record<string, Record<string, Record<string, unknown>>>

      if (providerSettings.modelProvider) {
        for (const [provider, config] of Object.entries(providerSettings.modelProvider)) {
          providerConfigs.value.translation[provider] = {
            apiKey: config.apiKey as string,
            modelName: config.modelName as string,
            customBaseUrl: config.customBaseUrl as string,
            rpmLimit: parseNum(config.rpmTranslation, DEFAULT_RPM_TRANSLATION),
            maxRetries: parseNum(config.translationMaxRetries, DEFAULT_TRANSLATION_MAX_RETRIES)
          }
        }
      }

      if (providerSettings.hqTranslateProvider) {
        for (const [provider, config] of Object.entries(providerSettings.hqTranslateProvider)) {
          providerConfigs.value.hqTranslation[provider] = {
            apiKey: config.hqApiKey as string,
            modelName: config.hqModelName as string,
            customBaseUrl: config.hqCustomBaseUrl as string,
            batchSize: parseNum(config.hqBatchSize, 3),
            sessionReset: parseNum(config.hqSessionReset, 20),
            rpmLimit: parseNum(config.hqRpmLimit, 7),
            maxRetries: parseNum(config.hqMaxRetries, DEFAULT_HQ_TRANSLATION_MAX_RETRIES),
            lowReasoning: config.hqLowReasoning as boolean,
            noThinkingMethod: (config.hqNoThinkingMethod as NoThinkingMethod) || 'gemini',
            forceJsonOutput: config.hqForceJsonOutput as boolean,
            useStream: config.hqUseStream as boolean,
            prompt: config.hqPrompt as string
          }
        }
      }

      if (providerSettings.aiVisionProvider) {
        for (const [provider, config] of Object.entries(providerSettings.aiVisionProvider)) {
          providerConfigs.value.aiVisionOcr[provider] = {
            apiKey: config.aiVisionApiKey as string,
            modelName: config.aiVisionModelName as string,
            customBaseUrl: config.customAiVisionBaseUrl as string,
            prompt: config.aiVisionOcrPrompt as string,
            rpmLimit: parseNum(config.rpmAiVisionOcr, DEFAULT_RPM_AI_VISION_OCR),
            isJsonMode: config.aiVisionPromptModeSelect === 'json',
            minImageSize: parseNum(config.aiVisionMinImageSize, DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE)
          }
        }
      }
    }

    console.log('[Settings] 后端设置映射完成')
  }

  /**
   * 构建服务商分组配置用于保存到后端
   */
  function buildProviderSettingsForBackend(): Record<string, Record<string, Record<string, unknown>>> {
    const modelProviderConfigs: Record<string, Record<string, unknown>> = {}
    const hqTranslateProviderConfigs: Record<string, Record<string, unknown>> = {}
    const aiVisionProviderConfigs: Record<string, Record<string, unknown>> = {}

    for (const [provider, config] of Object.entries(providerConfigs.value.translation)) {
      modelProviderConfigs[provider] = {
        apiKey: config.apiKey || '',
        modelName: config.modelName || '',
        customBaseUrl: config.customBaseUrl || '',
        rpmTranslation: String(config.rpmLimit || 0),
        translationMaxRetries: String(config.maxRetries || 3)
      }
    }

    for (const [provider, config] of Object.entries(providerConfigs.value.hqTranslation)) {
      hqTranslateProviderConfigs[provider] = {
        hqApiKey: config.apiKey || '',
        hqModelName: config.modelName || '',
        hqCustomBaseUrl: config.customBaseUrl || '',
        hqBatchSize: String(config.batchSize || 3),
        hqSessionReset: String(config.sessionReset || 20),
        hqRpmLimit: String(config.rpmLimit || 7),
        hqMaxRetries: String(config.maxRetries || 2),
        hqLowReasoning: config.lowReasoning || false,
        hqNoThinkingMethod: config.noThinkingMethod || 'gemini',
        hqForceJsonOutput: config.forceJsonOutput ?? true,
        hqUseStream: config.useStream || false,
        hqPrompt: config.prompt || ''
      }
    }

    for (const [provider, config] of Object.entries(providerConfigs.value.aiVisionOcr)) {
      aiVisionProviderConfigs[provider] = {
        aiVisionApiKey: config.apiKey || '',
        aiVisionModelName: config.modelName || '',
        customAiVisionBaseUrl: config.customBaseUrl || '',
        aiVisionOcrPrompt: config.prompt || '',
        rpmAiVisionOcr: String(config.rpmLimit || 0),
        aiVisionPromptModeSelect: config.isJsonMode ? 'json' : 'normal',
        aiVisionMinImageSize: String(config.minImageSize ?? DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE)
      }
    }

    return {
      ocrEngine: {},
      aiVisionProvider: aiVisionProviderConfigs,
      modelProvider: modelProviderConfigs,
      hqTranslateProvider: hqTranslateProviderConfigs
    }
  }

  /**
   * 保存设置到后端
   */
  async function saveToBackend(): Promise<boolean> {
    try {
      const { saveUserSettings } = await import('@/api/config')

      // 保存当前所有服务商的配置到缓存
      translationModule.saveTranslationProviderConfig(settings.value.translation.provider)
      hqTranslationModule.saveHqProviderConfig(settings.value.hqTranslation.provider)
      ocrModule.saveAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

      const backendSettings: Record<string, unknown> = {
        // OCR 设置
        ocrEngine: settings.value.ocrEngine,
        sourceLanguage: settings.value.sourceLanguage,
        textDetector: settings.value.textDetector,

        // 百度 OCR
        baiduApiKey: settings.value.baiduOcr.apiKey,
        baiduSecretKey: settings.value.baiduOcr.secretKey,
        baiduVersion: settings.value.baiduOcr.version,
        baiduSourceLanguage: settings.value.baiduOcr.sourceLanguage,

        // AI 视觉 OCR
        aiVisionProvider: settings.value.aiVisionOcr.provider,
        aiVisionApiKey: settings.value.aiVisionOcr.apiKey,
        aiVisionModelName: settings.value.aiVisionOcr.modelName,
        aiVisionOcrPrompt: settings.value.aiVisionOcr.prompt,
        customAiVisionBaseUrl: settings.value.aiVisionOcr.customBaseUrl,
        rpmAiVisionOcr: String(settings.value.aiVisionOcr.rpmLimit),
        aiVisionPromptModeSelect: settings.value.aiVisionOcr.isJsonMode ? 'json' : 'normal',
        aiVisionMinImageSize: String(settings.value.aiVisionOcr.minImageSize),

        // 翻译服务
        modelProvider: settings.value.translation.provider,
        apiKey: settings.value.translation.apiKey,
        modelName: settings.value.translation.modelName,
        customBaseUrl: settings.value.translation.customBaseUrl,
        rpmTranslation: String(settings.value.translation.rpmLimit),
        translationMaxRetries: String(settings.value.translation.maxRetries),
        translatePromptModeSelect: settings.value.translation.isJsonMode ? 'json' : 'normal',

        // 翻译提示词
        promptContent: settings.value.translatePrompt,
        enableTextboxPrompt: settings.value.useTextboxPrompt,
        textboxPromptContent: settings.value.textboxPrompt,

        // 高质量翻译
        hqTranslateProvider: settings.value.hqTranslation.provider,
        hqApiKey: settings.value.hqTranslation.apiKey,
        hqModelName: settings.value.hqTranslation.modelName,
        hqCustomBaseUrl: settings.value.hqTranslation.customBaseUrl,
        hqBatchSize: String(settings.value.hqTranslation.batchSize),
        hqSessionReset: String(settings.value.hqTranslation.sessionReset),
        hqRpmLimit: String(settings.value.hqTranslation.rpmLimit),
        hqMaxRetries: String(settings.value.hqTranslation.maxRetries),
        hqPrompt: settings.value.hqTranslation.prompt,
        hqLowReasoning: settings.value.hqTranslation.lowReasoning,
        hqNoThinkingMethod: settings.value.hqTranslation.noThinkingMethod,
        hqForceJsonOutput: settings.value.hqTranslation.forceJsonOutput,
        hqUseStream: settings.value.hqTranslation.useStream,

        // AI 校对
        proofreadingEnabled: settings.value.proofreading.enabled,
        proofreadingMaxRetries: String(settings.value.proofreading.maxRetries),
        proofreading: {
          enabled: settings.value.proofreading.enabled,
          maxRetries: String(settings.value.proofreading.maxRetries),
          rounds: settings.value.proofreading.rounds.map(round => ({
            name: round.name,
            provider: round.provider,
            apiKey: round.apiKey,
            modelName: round.modelName,
            customBaseUrl: round.customBaseUrl,
            prompt: round.prompt,
            batchSize: round.batchSize,
            sessionReset: round.sessionReset,
            rpmLimit: round.rpmLimit,
            maxRetries: round.maxRetries,
            lowReasoning: round.lowReasoning,
            noThinkingMethod: round.noThinkingMethod,
            forceJsonOutput: round.forceJsonOutput
          }))
        },

        // 文本框扩展
        boxExpandRatio: String(settings.value.boxExpand.ratio),
        boxExpandTop: String(settings.value.boxExpand.top),
        boxExpandBottom: String(settings.value.boxExpand.bottom),
        boxExpandLeft: String(settings.value.boxExpand.left),
        boxExpandRight: String(settings.value.boxExpand.right),

        // 精确掩膜
        usePreciseMask: settings.value.preciseMask.enabled,
        maskDilateSize: String(settings.value.preciseMask.dilateSize),
        maskBoxExpandRatio: String(settings.value.preciseMask.boxExpandRatio || 0),

        // PDF 处理方式
        pdfProcessingMethod: settings.value.pdfProcessingMethod,

        // 调试
        showDetectionDebug: settings.value.showDetectionDebug,

        // 并行翻译设置
        parallelEnabled: settings.value.parallel.enabled,
        parallelDeepLearningLockSize: settings.value.parallel.deepLearningLockSize,

        // 服务商分组配置缓存
        providerSettings: buildProviderSettingsForBackend(),
      }

      const response = await saveUserSettings(backendSettings)

      if (response.success) {
        console.log('[Settings] 设置已保存到后端')
        return true
      } else {
        console.error('[Settings] 保存设置到后端失败:', response)
        return false
      }
    } catch (error) {
      console.error('[Settings] 保存设置到后端出错:', error)
      return false
    }
  }

  // ============================================================
  // 返回 Store
  // ============================================================

  return {
    // 核心状态
    settings,
    providerConfigs,
    customFontPresets: modelHistoryModule.customFontPresets,

    // OCR 模块
    ocrEngine: ocrModule.ocrEngine,
    sourceLanguage: ocrModule.sourceLanguage,
    setOcrEngine: ocrModule.setOcrEngine,
    setSourceLanguage: ocrModule.setSourceLanguage,
    updateBaiduOcr: ocrModule.updateBaiduOcr,
    updateAiVisionOcr: ocrModule.updateAiVisionOcr,
    setAiVisionOcrProvider: ocrModule.setAiVisionOcrProvider,
    setAiVisionOcrPromptMode: ocrModule.setAiVisionOcrPromptMode,
    saveAiVisionOcrProviderConfig: ocrModule.saveAiVisionOcrProviderConfig,
    restoreAiVisionOcrProviderConfig: ocrModule.restoreAiVisionOcrProviderConfig,

    // 翻译服务模块
    translationProvider: translationModule.translationProvider,
    setTranslationProvider: translationModule.setTranslationProvider,
    updateTranslationService: translationModule.updateTranslationService,
    setTranslatePrompt: translationModule.setTranslatePrompt,
    setTranslatePromptMode: translationModule.setTranslatePromptMode,
    saveTranslationProviderConfig: translationModule.saveTranslationProviderConfig,
    restoreTranslationProviderConfig: translationModule.restoreTranslationProviderConfig,

    // 检测设置模块
    setTextDetector: detectionModule.setTextDetector,
    updateBoxExpand: detectionModule.updateBoxExpand,
    updatePreciseMask: detectionModule.updatePreciseMask,

    // 高质量翻译模块
    hqProvider: hqTranslationModule.hqProvider,
    setHqProvider: hqTranslationModule.setHqProvider,
    updateHqTranslation: hqTranslationModule.updateHqTranslation,
    setHqUseStream: hqTranslationModule.setHqUseStream,
    setHqNoThinkingMethod: hqTranslationModule.setHqNoThinkingMethod,
    setHqForceJsonOutput: hqTranslationModule.setHqForceJsonOutput,
    saveHqProviderConfig: hqTranslationModule.saveHqProviderConfig,
    restoreHqProviderConfig: hqTranslationModule.restoreHqProviderConfig,

    // AI校对模块
    isProofreadingEnabled: proofreadingModule.isProofreadingEnabled,
    setProofreadingEnabled: proofreadingModule.setProofreadingEnabled,
    addProofreadingRound: proofreadingModule.addProofreadingRound,
    updateProofreadingRound: proofreadingModule.updateProofreadingRound,
    removeProofreadingRound: proofreadingModule.removeProofreadingRound,
    setProofreadingMaxRetries: proofreadingModule.setProofreadingMaxRetries,

    // 提示词管理模块
    setTextboxPrompt: promptsModule.setTextboxPrompt,
    setUseTextboxPrompt: promptsModule.setUseTextboxPrompt,

    // 更多设置模块
    textStyle: miscModule.textStyle,
    updateSettings: miscModule.updateSettings,
    updateTextStyle: miscModule.updateTextStyle,
    setPdfProcessingMethod: miscModule.setPdfProcessingMethod,
    setShowDetectionDebug: miscModule.setShowDetectionDebug,

    // 模型历史记录模块
    modelHistory: modelHistoryModule.modelHistory,
    addModelToHistory: modelHistoryModule.addModelToHistory,
    getModelHistory: modelHistoryModule.getModelHistory,
    clearModelHistory: modelHistoryModule.clearModelHistory,
    addCustomFontPreset: modelHistoryModule.addCustomFontPreset,
    removeCustomFontPreset: modelHistoryModule.removeCustomFontPreset,

    // 兼容旧接口
    saveProviderConfig,
    restoreProviderConfig,

    // 持久化方法
    saveToStorage,
    loadFromStorage,
    initSettings,
    resetToDefaults,

    // 后端同步方法
    loadFromBackend,
    saveToBackend
  }
})

// 导出类型
export type { ProviderConfigsCache } from './types'
