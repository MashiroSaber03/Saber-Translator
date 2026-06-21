/**
 * 设置状态管理 Store
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
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  TranslationSettings,
  TextDetector,
  TranslationProvider,
  HqTranslationProvider,
  PluginAgentProvider,
} from '@/types/settings'
import {
  STORAGE_KEY_TRANSLATION_SETTINGS,
  STORAGE_KEY_THEME,
  STORAGE_KEY_PROVIDER_CONFIGS,
  DEFAULT_RPM_TRANSLATION,
  DEFAULT_RPM_AI_VISION_OCR,
  DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE,
  DEFAULT_TRANSLATION_MAX_RETRIES,
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_MAX_RETRIES
} from '@/constants'
import { getProviderManifest, normalizeProviderId } from '@/config/aiProviders'
import { normalizeHybridOcrConfig } from '@/utils/hybridOcr'
import {
  normalizeOpenAiOptions
} from '@/utils/openaiOptions'

import type { ProviderConfigsCache } from './types'
import { createDefaultSettings } from './defaults'

// 导入各功能模块
import {
  useOcrSettings,
  useTranslationSettings,
  useDetectionSettings,
  useHqTranslationSettings,
  usePluginAgentSettings,
  useProofreadingSettings,
  usePromptsSettings,
  useMiscSettings
} from './modules'

// ============================================================
// Store 定义
// ============================================================

export const useSettingsStore = defineStore('settings', () => {
  function inferAiVisionPromptMode(prompt: unknown): 'normal' | 'json' | 'paddleocr_vl' {
    const promptText = typeof prompt === 'string' ? prompt.trim() : ''
    if (promptText.startsWith('对图中的') && promptText.endsWith('进行OCR:')) {
      return 'paddleocr_vl'
    }
    return 'normal'
  }

  // ============================================================
  // 核心状态定义
  // ============================================================

  /** 翻译设置 */
  const settings = ref<TranslationSettings>(createDefaultSettings())

  /** 主题状态 */
  const theme = ref<'light' | 'dark'>('light')

  /** 服务商配置分组存储（用于切换服务商时保存/恢复配置） */
  const providerConfigs = ref<ProviderConfigsCache>({
    translation: {},
    hqTranslation: {},
    pluginAgent: {},
    aiVisionOcr: {}
  })

  function normalizeTextDetector(detector: unknown): TextDetector {
    if (detector === 'ctd' || detector === 'yolo' || detector === 'default') {
      return detector
    }
    return 'default'
  }

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

  function savePluginAgentSettingsToStorage(): void {
    try {
      const defaults = createDefaultSettings()
      const stored = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)
      const baseSettings = stored
        ? deepMerge(defaults, JSON.parse(stored) as Partial<TranslationSettings>)
        : defaults

      baseSettings.pluginAgent = JSON.parse(JSON.stringify(settings.value.pluginAgent))
      localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, JSON.stringify(baseSettings))
    } catch (error) {
      console.error('保存插件 Agent 设置到 localStorage 失败:', error)
    }
  }

  function savePluginAgentProviderConfigsToStorage(): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_PROVIDER_CONFIGS)
      const parsed = stored
        ? JSON.parse(stored) as Partial<ProviderConfigsCache>
        : {}

      const nextProviderConfigs: ProviderConfigsCache = {
        translation: parsed.translation || {},
        hqTranslation: parsed.hqTranslation || {},
        pluginAgent: JSON.parse(JSON.stringify(providerConfigs.value.pluginAgent)),
        aiVisionOcr: parsed.aiVisionOcr || {}
      }

      localStorage.setItem(STORAGE_KEY_PROVIDER_CONFIGS, JSON.stringify(nextProviderConfigs))
    } catch (error) {
      console.error('保存插件 Agent 服务商配置缓存失败:', error)
    }
  }

  /**
   * 从 localStorage 加载设置
   * 从当前 settings schema 存储键读取设置。
   */
  function loadFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)

      if (data) {
        const parsed = JSON.parse(data) as Partial<TranslationSettings>
        if (parsed.settingsSchemaVersion !== 3) {
          console.warn('localStorage 设置 schemaVersion 无效，已忽略该设置对象')
          return
        }
        const defaults = createDefaultSettings()
        // 深度合并，确保新增的默认值不会丢失
        settings.value = deepMerge(defaults, parsed)
        normalizeProviderIds()
        settings.value.textDetector = normalizeTextDetector(settings.value.textDetector)
        // 确保数值类型正确
        ensureNumericTypes()

        // 确保 translatePrompt 与当前翻译模式和 JSON 模式同步（4个独立存储字段之一）
        const t = settings.value.translation
        if (t.translationMode === 'single') {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
        } else {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
        }
        console.log('已从 localStorage 加载设置')
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
        const parsed = JSON.parse(data) as Partial<ProviderConfigsCache>
        // 确保结构完整
        providerConfigs.value = {
          translation: parsed.translation || {},
          hqTranslation: parsed.hqTranslation || {},
          pluginAgent: parsed.pluginAgent || {},
          aiVisionOcr: parsed.aiVisionOcr || {}
        }
        normalizeProviderConfigIds()
        normalizeProviderConfigOpenAiOptions()
        console.log('已从 localStorage 加载服务商配置缓存')
      }
    } catch (error) {
      console.error('加载服务商配置缓存失败:', error)
    }
  }

  /**
   * 确保设置中的数值类型正确
   * textStyle 由编辑侧栏默认样式统一决定，不在数值归一化里处理。
   */
  function ensureNumericTypes(): void {
    const parseNumberOrFallback = (value: unknown, fallback: number): number => {
      if (value === undefined || value === null || value === '') return fallback
      const parsed = Number(value)
      return Number.isNaN(parsed) ? fallback : parsed
    }

    const be = settings.value.boxExpand
    be.ratio = Number(be.ratio) || 1.0
    be.top = Number(be.top) || 0
    be.bottom = Number(be.bottom) || 0
    be.left = Number(be.left) || 0
    be.right = Number(be.right) || 0

    const pm = settings.value.preciseMask
    pm.dilateSize = Number(pm.dilateSize) || 5
    pm.boxExpandRatio = Number(pm.boxExpandRatio) || 1.0

    if (
      settings.value.saberYoloRefineOverlapThreshold === undefined ||
      settings.value.saberYoloRefineOverlapThreshold === null ||
      isNaN(Number(settings.value.saberYoloRefineOverlapThreshold))
    ) {
      settings.value.saberYoloRefineOverlapThreshold = 50
    } else {
      settings.value.saberYoloRefineOverlapThreshold = Number(settings.value.saberYoloRefineOverlapThreshold)
    }

    settings.value.minTextBlockAreaPercent = Math.max(
      0,
      parseNumberOrFallback(settings.value.minTextBlockAreaPercent, 0.05)
    )

    settings.value.enableAuxYoloDetection = Boolean(settings.value.enableAuxYoloDetection)
    if (
      settings.value.auxYoloConfThreshold === undefined ||
      settings.value.auxYoloConfThreshold === null ||
      isNaN(Number(settings.value.auxYoloConfThreshold))
    ) {
      settings.value.auxYoloConfThreshold = 0.4
    } else {
      settings.value.auxYoloConfThreshold = Number(settings.value.auxYoloConfThreshold)
    }
    if (
      settings.value.auxYoloOverlapThreshold === undefined ||
      settings.value.auxYoloOverlapThreshold === null ||
      isNaN(Number(settings.value.auxYoloOverlapThreshold))
    ) {
      settings.value.auxYoloOverlapThreshold = 0.1
    } else {
      settings.value.auxYoloOverlapThreshold = Number(settings.value.auxYoloOverlapThreshold)
    }

    const tr = settings.value.translation as typeof settings.value.translation & Record<string, unknown>
    tr.openaiOptions = normalizeOpenAiOptions(
      tr.openaiOptions,
      {
        execution: {
          useStream: true,
          rpmLimit: DEFAULT_RPM_TRANSLATION,
          transportRetries: 1,
          businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
        }
      }
    )

    const hq = settings.value.hqTranslation as typeof settings.value.hqTranslation & Record<string, unknown>
    hq.batchSize = Number(hq.batchSize) || 10
    hq.openaiOptions = normalizeOpenAiOptions(
      hq.openaiOptions,
      {
        execution: {
          useStream: true,
          rpmLimit: 7,
          transportRetries: 3,
          businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
        }
      }
    )

    const pluginAgent = settings.value.pluginAgent as typeof settings.value.pluginAgent & Record<string, unknown>
    pluginAgent.openaiOptions = normalizeOpenAiOptions(
      pluginAgent.openaiOptions,
      {
        execution: {
          useStream: true,
          rpmLimit: 0,
          transportRetries: 10,
          businessRetries: 10
        }
      }
    )

    const av = settings.value.aiVisionOcr as typeof settings.value.aiVisionOcr & Record<string, unknown>
    av.openaiOptions = normalizeOpenAiOptions(
      av.openaiOptions,
      {
        execution: {
          useStream: false,
          rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
          transportRetries: 1,
          businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
        }
      }
    )
    av.promptMode = av.promptMode || inferAiVisionPromptMode(av.prompt)
    av.openaiOptions.request.forceJsonOutput = av.promptMode === 'json'
    // 对于 minImageSize，0 是合法值（表示禁用自动放大），所以不能用 || 操作符
    if (av.minImageSize === undefined || av.minImageSize === null || isNaN(Number(av.minImageSize))) {
      av.minImageSize = DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE
    } else {
      av.minImageSize = Number(av.minImageSize)
    }

    const hybrid = settings.value.hybridOcr as typeof settings.value.hybridOcr & Record<string, unknown>
    const normalizedHybrid = normalizeHybridOcrConfig(
      settings.value.ocrEngine,
      {
        ...hybrid,
        enabled: Boolean(hybrid.enabled)
      },
      {
        preferRecommendedOrder: Boolean(hybrid.enabled)
      }
    )
    settings.value.ocrEngine = normalizedHybrid.primaryEngine
    settings.value.hybridOcr = normalizedHybrid.hybrid

    const pr = settings.value.proofreading
    pr.maxRetries = parseNumberOrFallback(pr.maxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
    pr.rounds = pr.rounds.map(round => ({
      ...round,
      openaiOptions: normalizeOpenAiOptions(
        round.openaiOptions,
        {
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 1,
            businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
          }
        }
      )
    }))

    settings.value.settingsSchemaVersion = 3

  }

  function normalizeCurrentProviderId(provider: unknown, fallback: string): string {
    const normalized = normalizeProviderId(typeof provider === 'string' ? provider : '')
    return getProviderManifest(normalized) ? normalized : fallback
  }

  function normalizeProviderIds(): void {
    settings.value.translation.provider = normalizeCurrentProviderId(
      settings.value.translation.provider,
      'siliconflow'
    ) as TranslationProvider
    settings.value.hqTranslation.provider = normalizeCurrentProviderId(
      settings.value.hqTranslation.provider,
      'siliconflow'
    ) as HqTranslationProvider
    settings.value.pluginAgent.provider = normalizeCurrentProviderId(
      settings.value.pluginAgent.provider,
      'deepseek'
    ) as PluginAgentProvider
    settings.value.aiVisionOcr.provider = normalizeCurrentProviderId(settings.value.aiVisionOcr.provider, 'volcano')
    settings.value.proofreading.rounds = settings.value.proofreading.rounds.map(round => ({
      ...round,
      provider: normalizeCurrentProviderId(round.provider, 'siliconflow') as HqTranslationProvider
    }))
  }

  function normalizeProviderConfigIds(): void {
    const normalizeRecord = <T>(record: Record<string, T>): Record<string, T> => {
      const normalized: Record<string, T> = {}
      for (const [provider, config] of Object.entries(record)) {
        const providerId = normalizeProviderId(provider)
        if (!getProviderManifest(providerId)) continue
        normalized[providerId] = config
      }
      return normalized
    }

    providerConfigs.value.translation = normalizeRecord(providerConfigs.value.translation)
    providerConfigs.value.hqTranslation = normalizeRecord(providerConfigs.value.hqTranslation)
    providerConfigs.value.pluginAgent = normalizeRecord(providerConfigs.value.pluginAgent)
    providerConfigs.value.aiVisionOcr = normalizeRecord(providerConfigs.value.aiVisionOcr)
  }

  function normalizeProviderConfigOpenAiOptions(): void {
    for (const config of Object.values(providerConfigs.value.translation)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          execution: {
            useStream: true,
            rpmLimit: DEFAULT_RPM_TRANSLATION,
            transportRetries: 1,
            businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
          }
        }
      )
    }

    for (const config of Object.values(providerConfigs.value.hqTranslation)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 3,
            businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
          }
        }
      )
    }

    for (const config of Object.values(providerConfigs.value.pluginAgent)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          execution: {
            useStream: true,
            rpmLimit: 0,
            transportRetries: 10,
            businessRetries: 10
          }
        }
      )
    }

    for (const config of Object.values(providerConfigs.value.aiVisionOcr)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          execution: {
            useStream: false,
            rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
            transportRetries: 1,
            businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
          }
        }
      )
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
        if (!(key in target)) continue
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

  const pluginAgentModule = usePluginAgentSettings(
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

  // ============================================================
  // 初始化方法
  // ============================================================

  /**
   * 初始化设置（从 localStorage 加载）
   */
  function initSettings(): void {
    loadFromStorage()
    loadProviderConfigsFromStorage()
  }

  /**
   * 重置所有设置为默认值
   */
  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
    saveToStorage()
    console.log('设置已重置为默认值')
  }

  function setTheme(nextTheme: 'light' | 'dark'): void {
    theme.value = nextTheme
    try {
      localStorage.setItem(STORAGE_KEY_THEME, nextTheme)
      if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-theme', nextTheme)
        document.body.setAttribute('data-theme', nextTheme)
      }
    } catch (error) {
      console.warn('保存主题设置失败:', error)
    }
  }

  function toggleTheme(): void {
    setTheme(theme.value === 'light' ? 'dark' : 'light')
  }

  function loadThemeFromStorage(): void {
    try {
      const storedTheme = localStorage.getItem(STORAGE_KEY_THEME)
      if (storedTheme === 'light' || storedTheme === 'dark') {
        setTheme(storedTheme)
      }
    } catch (error) {
      console.warn('读取主题设置失败:', error)
    }
  }

  // ============================================================
  // 后端同步方法
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
        if (!applyBackendSettings(backendSettings)) {
          return false
        }
        ensureNumericTypes()

        // 编辑侧栏文字设置始终使用当前默认值，不从后端恢复。
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
  function applyBackendSettings(backendSettings: Record<string, unknown>): boolean {
    const schemaVersion = backendSettings.settingsSchemaVersion
    if (schemaVersion !== 3) {
      console.warn('[Settings] 后端设置 schemaVersion 无效，已忽略该设置对象')
      return false
    }

    const { providerConfigs: nestedProviderConfigs, ...settingsPayload } = backendSettings
    settings.value = deepMerge(
      createDefaultSettings(),
      settingsPayload as Partial<TranslationSettings>
    )

    if (nestedProviderConfigs && typeof nestedProviderConfigs === 'object') {
      providerConfigs.value = {
        translation: (nestedProviderConfigs as ProviderConfigsCache).translation || {},
        hqTranslation: (nestedProviderConfigs as ProviderConfigsCache).hqTranslation || {},
        pluginAgent: (nestedProviderConfigs as ProviderConfigsCache).pluginAgent || {},
        aiVisionOcr: (nestedProviderConfigs as ProviderConfigsCache).aiVisionOcr || {}
      }
    }

    normalizeProviderIds()
    normalizeProviderConfigIds()
    normalizeProviderConfigOpenAiOptions()
    console.log('[Settings] 后端设置已应用')
    return true
  }

  /**
   * 构建服务商分组配置用于保存到后端
   */
  function buildProviderSettingsForBackend(): ProviderConfigsCache {
    return JSON.parse(JSON.stringify(providerConfigs.value)) as ProviderConfigsCache
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
      pluginAgentModule.savePluginAgentProviderConfig(settings.value.pluginAgent.provider)
      ocrModule.saveAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

      const backendSettings: Record<string, unknown> = JSON.parse(JSON.stringify(settings.value))
      backendSettings.settingsSchemaVersion = 3
      backendSettings.providerConfigs = buildProviderSettingsForBackend()

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

  async function savePluginAgentSettings(): Promise<boolean> {
    try {
      const currentProvider = normalizeProviderId(settings.value.pluginAgent.provider)
      providerConfigs.value.pluginAgent[currentProvider] = {
        apiKey: settings.value.pluginAgent.apiKey,
        modelName: settings.value.pluginAgent.modelName,
        customBaseUrl: settings.value.pluginAgent.customBaseUrl,
        openaiOptions: JSON.parse(JSON.stringify(settings.value.pluginAgent.openaiOptions))
      }

      savePluginAgentSettingsToStorage()
      savePluginAgentProviderConfigsToStorage()

      const { getUserSettings, saveUserSettings } = await import('@/api/config')
      let backendSettings: Record<string, unknown> = {}

      try {
        const response = await getUserSettings()
        if (response.success && response.settings && typeof response.settings === 'object') {
          backendSettings = JSON.parse(JSON.stringify(response.settings))
        }
      } catch (error) {
        console.warn('[Settings] 读取后端设置失败，插件 Agent 将基于空设置保存:', error)
      }

      backendSettings.pluginAgent = JSON.parse(JSON.stringify(settings.value.pluginAgent))
      const backendProviderConfigs = (
        backendSettings.providerConfigs && typeof backendSettings.providerConfigs === 'object'
          ? JSON.parse(JSON.stringify(backendSettings.providerConfigs))
          : {}
      ) as Record<string, unknown>
      backendProviderConfigs.pluginAgent = JSON.parse(JSON.stringify(providerConfigs.value.pluginAgent))
      backendSettings.providerConfigs = backendProviderConfigs

      backendSettings.settingsSchemaVersion = 3

      const saveResponse = await saveUserSettings(backendSettings)
      if (saveResponse.success) {
        return true
      }

      console.error('[Settings] 保存插件 Agent 设置到后端失败:', saveResponse)
      return false
    } catch (error) {
      console.error('[Settings] 保存插件 Agent 设置时出错:', error)
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
    theme,

    // OCR 模块
    ocrEngine: ocrModule.ocrEngine,
    sourceLanguage: ocrModule.sourceLanguage,
    setOcrEngine: ocrModule.setOcrEngine,
    setSourceLanguage: ocrModule.setSourceLanguage,
    updateBaiduOcr: ocrModule.updateBaiduOcr,
    updatePaddleOcrVl: ocrModule.updatePaddleOcrVl,
    updateAiVisionOcr: ocrModule.updateAiVisionOcr,
    updateHybridOcr: ocrModule.updateHybridOcr,
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
    setMinTextBlockAreaPercent: detectionModule.setMinTextBlockAreaPercent,
    setEnableAuxYoloDetection: detectionModule.setEnableAuxYoloDetection,
    setAuxYoloConfThreshold: detectionModule.setAuxYoloConfThreshold,
    setAuxYoloOverlapThreshold: detectionModule.setAuxYoloOverlapThreshold,
    setEnableSaberYoloRefine: detectionModule.setEnableSaberYoloRefine,
    setSaberYoloRefineOverlapThreshold: detectionModule.setSaberYoloRefineOverlapThreshold,
    updateBoxExpand: detectionModule.updateBoxExpand,
    updatePreciseMask: detectionModule.updatePreciseMask,

    // 高质量翻译模块
    hqProvider: hqTranslationModule.hqProvider,
    setHqProvider: hqTranslationModule.setHqProvider,
    updateHqTranslation: hqTranslationModule.updateHqTranslation,
    setHqUseStream: hqTranslationModule.setHqUseStream,
    setHqForceJsonOutput: hqTranslationModule.setHqForceJsonOutput,
    saveHqProviderConfig: hqTranslationModule.saveHqProviderConfig,
    restoreHqProviderConfig: hqTranslationModule.restoreHqProviderConfig,

    // 插件 Agent 模块
    pluginAgentProvider: pluginAgentModule.pluginAgentProvider,
    setPluginAgentProvider: pluginAgentModule.setPluginAgentProvider,
    updatePluginAgent: pluginAgentModule.updatePluginAgent,
    savePluginAgentProviderConfig: pluginAgentModule.savePluginAgentProviderConfig,
    restorePluginAgentProviderConfig: pluginAgentModule.restorePluginAgentProviderConfig,

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
    setAutoSaveInBookshelfMode: miscModule.setAutoSaveInBookshelfMode,
    setRemoveTextWithOcr: miscModule.setRemoveTextWithOcr,
    setEnableVerboseLogs: miscModule.setEnableVerboseLogs,
    setLamaDisableResize: miscModule.setLamaDisableResize,

    // 持久化方法
    saveToStorage,
    loadFromStorage,
    setTheme,
    toggleTheme,
    loadThemeFromStorage,
    initSettings,
    resetToDefaults,

    // 后端同步方法
    loadFromBackend,
    saveToBackend,
    savePluginAgentSettings
  }
})

// 导出类型
export type { ProviderConfigsCache } from './types'
