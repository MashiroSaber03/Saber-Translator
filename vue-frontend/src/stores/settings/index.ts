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
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  TranslationSettings,
  OcrEngine,
  TextDetector,
  TranslationProvider,
  HqTranslationProvider,
  PluginAgentProvider,
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
import { normalizeProviderId } from '@/config/aiProviders'
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
  useTranslationConstraintsSettings,
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
  function inferAiVisionPromptMode(prompt: unknown, isJsonMode: unknown): 'normal' | 'json' | 'paddleocr_vl' {
    if (isJsonMode === true) {
      return 'json'
    }
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
      stripLegacyOpenAiMirrorFields()
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
      stripLegacyOpenAiMirrorFields()
      const data = JSON.stringify(providerConfigs.value)
      localStorage.setItem(STORAGE_KEY_PROVIDER_CONFIGS, data)
    } catch (error) {
      console.error('保存服务商配置缓存失败:', error)
    }
  }

  function savePluginAgentSettingsToStorage(): void {
    try {
      stripLegacyOpenAiMirrorFields()
      const defaults = createDefaultSettings()
      const stored = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)
      const baseSettings = stored
        ? deepMerge(defaults, stripDeprecatedSettingsFields(JSON.parse(stored)) as Partial<TranslationSettings>)
        : defaults

      baseSettings.pluginAgent = JSON.parse(JSON.stringify(settings.value.pluginAgent))
      localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, JSON.stringify(baseSettings))
    } catch (error) {
      console.error('保存插件 Agent 设置到 localStorage 失败:', error)
    }
  }

  function savePluginAgentProviderConfigsToStorage(): void {
    try {
      stripLegacyOpenAiMirrorFields()
      const stored = localStorage.getItem(STORAGE_KEY_PROVIDER_CONFIGS)
      const parsed = stored
        ? stripDeprecatedProviderConfigFields(JSON.parse(stored))
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

  /** 原版 localStorage 存储键（用于兼容迁移） */
  const LEGACY_STORAGE_KEY = 'saber_translator_settings'

  function stripDeprecatedSettingsFields(raw: unknown): Partial<TranslationSettings> {
    if (!raw || typeof raw !== 'object') return {}

    const cloned = JSON.parse(JSON.stringify(raw)) as Record<string, unknown>
    const hq = cloned.hqTranslation
    if (hq && typeof hq === 'object') {
      delete (hq as Record<string, unknown>).sessionReset
      delete (hq as Record<string, unknown>).lowReasoning
      delete (hq as Record<string, unknown>).noThinkingMethod
    }

    const proofreading = cloned.proofreading
    if (proofreading && typeof proofreading === 'object') {
      const rounds = (proofreading as Record<string, unknown>).rounds
      if (Array.isArray(rounds)) {
        for (const round of rounds) {
          if (round && typeof round === 'object') {
            delete (round as Record<string, unknown>).sessionReset
            delete (round as Record<string, unknown>).lowReasoning
            delete (round as Record<string, unknown>).noThinkingMethod
          }
        }
      }
    }

    return cloned as Partial<TranslationSettings>
  }

  function stripDeprecatedProviderConfigFields(raw: unknown): Partial<ProviderConfigsCache> {
    if (!raw || typeof raw !== 'object') return {}

    const cloned = JSON.parse(JSON.stringify(raw)) as Record<string, unknown>
    const hqConfigs = cloned.hqTranslation
    if (hqConfigs && typeof hqConfigs === 'object') {
      for (const config of Object.values(hqConfigs as Record<string, unknown>)) {
        if (config && typeof config === 'object') {
          delete (config as Record<string, unknown>).sessionReset
          delete (config as Record<string, unknown>).lowReasoning
          delete (config as Record<string, unknown>).noThinkingMethod
        }
      }
    }

    return cloned as Partial<ProviderConfigsCache>
  }

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
        const parsed = stripDeprecatedSettingsFields(JSON.parse(data))
        const defaults = createDefaultSettings()
        // 深度合并，确保新增的默认值不会丢失
        settings.value = deepMerge(defaults, parsed)
        normalizeProviderAliases()
        settings.value.textDetector = normalizeTextDetector(settings.value.textDetector)
        // 确保数值类型正确
        ensureNumericTypes()

        // 【复刻原版】左侧边栏文字设置始终使用默认值，不从 localStorage 恢复
        settings.value.textStyle = { ...defaults.textStyle }

        // 确保 translatePrompt 与当前翻译模式和 JSON 模式同步（4个独立存储字段之一）
        const t = settings.value.translation
        if (t.translationMode === 'single') {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
        } else {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
        }

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
        const parsed = stripDeprecatedProviderConfigFields(JSON.parse(data))
        // 确保结构完整
        providerConfigs.value = {
          translation: parsed.translation || {},
          hqTranslation: parsed.hqTranslation || {},
          pluginAgent: parsed.pluginAgent || {},
          aiVisionOcr: parsed.aiVisionOcr || {}
        }
        normalizeProviderConfigAliases()
        normalizeProviderConfigOpenAiOptions()
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
        rpmLimit: tr.rpmLimit,
        maxRetries: tr.maxRetries,
        isJsonMode: tr.isJsonMode,
        extraBody: tr.extraBody,
        useStream: tr.useStream
      },
      {
        execution: {
          useStream: false,
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
        rpmLimit: hq.rpmLimit,
        maxRetries: hq.maxRetries,
        forceJsonOutput: hq.forceJsonOutput,
        extraBody: hq.extraBody,
        useStream: hq.useStream
      },
      {
        execution: {
          useStream: true,
          rpmLimit: 7,
          transportRetries: 1,
          businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
        }
      }
    )

    const pluginAgent = settings.value.pluginAgent as typeof settings.value.pluginAgent & Record<string, unknown>
    pluginAgent.openaiOptions = normalizeOpenAiOptions(
      pluginAgent.openaiOptions,
      {
        rpmLimit: pluginAgent.rpmLimit,
        maxRetries: pluginAgent.maxRetries,
        forceJsonOutput: pluginAgent.forceJsonOutput,
        extraBody: pluginAgent.extraBody,
        useStream: pluginAgent.useStream
      },
      {
        execution: {
          useStream: true,
          rpmLimit: 7,
          transportRetries: 1,
          businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
        }
      }
    )

    const av = settings.value.aiVisionOcr as typeof settings.value.aiVisionOcr & Record<string, unknown>
    av.openaiOptions = normalizeOpenAiOptions(
      av.openaiOptions,
      {
        rpmLimit: av.rpmLimit,
        maxRetries: av.maxRetries,
        isJsonMode: av.isJsonMode,
        extraBody: av.extraBody,
        useStream: av.useStream
      },
      {
        execution: {
          useStream: false,
          rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
          transportRetries: 1,
          businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
        }
      }
    )
    av.promptMode = av.promptMode || inferAiVisionPromptMode(av.prompt, av.openaiOptions.request.forceJsonOutput)
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
    delete (settings.value.hybridOcr as Record<string, unknown>).threshold48px
    delete (settings.value.hybridOcr as Record<string, unknown>).thresholdMangaOcr
    delete (settings.value.hybridOcr as Record<string, unknown>).thresholdPaddleOcr

    const pr = settings.value.proofreading
    pr.maxRetries = parseNumberOrFallback(pr.maxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
    pr.rounds = pr.rounds.map(round => ({
      ...round,
      openaiOptions: normalizeOpenAiOptions(
        round.openaiOptions,
        {
          rpmLimit: (round as Record<string, unknown>).rpmLimit,
          maxRetries: (round as Record<string, unknown>).maxRetries,
          forceJsonOutput: (round as Record<string, unknown>).forceJsonOutput,
          extraBody: (round as Record<string, unknown>).extraBody,
          useStream: (round as Record<string, unknown>).useStream
        },
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

    settings.value.settingsSchemaVersion = 2

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

  function normalizeProviderAliases(): void {
    settings.value.translation.provider = normalizeProviderId(settings.value.translation.provider) as TranslationProvider
    settings.value.hqTranslation.provider = normalizeProviderId(settings.value.hqTranslation.provider) as HqTranslationProvider
    settings.value.pluginAgent.provider = normalizeProviderId(settings.value.pluginAgent.provider) as PluginAgentProvider
    settings.value.aiVisionOcr.provider = normalizeProviderId(settings.value.aiVisionOcr.provider)
    settings.value.proofreading.rounds = settings.value.proofreading.rounds.map(round => ({
      ...round,
      provider: normalizeProviderId(round.provider) as HqTranslationProvider
    }))
  }

  function normalizeProviderConfigAliases(): void {
    const normalizeRecord = <T>(record: Record<string, T>): Record<string, T> => {
      const normalized: Record<string, T> = {}
      for (const [provider, config] of Object.entries(record)) {
        normalized[normalizeProviderId(provider)] = config
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
          forceJsonOutput: (config as Record<string, unknown>).forceJsonOutput,
          isJsonMode: (config as Record<string, unknown>).isJsonMode,
          extraBody: (config as Record<string, unknown>).extraBody,
          useStream: (config as Record<string, unknown>).useStream,
          rpmLimit: (config as Record<string, unknown>).rpmLimit,
          maxRetries: (config as Record<string, unknown>).maxRetries
        },
        {
          execution: {
            useStream: false,
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
          forceJsonOutput: (config as Record<string, unknown>).forceJsonOutput,
          extraBody: (config as Record<string, unknown>).extraBody,
          useStream: (config as Record<string, unknown>).useStream,
          rpmLimit: (config as Record<string, unknown>).rpmLimit,
          maxRetries: (config as Record<string, unknown>).maxRetries
        },
        {
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 1,
            businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
          }
        }
      )
    }

    for (const config of Object.values(providerConfigs.value.pluginAgent)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          forceJsonOutput: (config as Record<string, unknown>).forceJsonOutput,
          extraBody: (config as Record<string, unknown>).extraBody,
          useStream: (config as Record<string, unknown>).useStream,
          rpmLimit: (config as Record<string, unknown>).rpmLimit,
          maxRetries: (config as Record<string, unknown>).maxRetries
        },
        {
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 1,
            businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES
          }
        }
      )
    }

    for (const config of Object.values(providerConfigs.value.aiVisionOcr)) {
      config.openaiOptions = normalizeOpenAiOptions(
        config.openaiOptions,
        {
          isJsonMode: (config as Record<string, unknown>).isJsonMode,
          extraBody: (config as Record<string, unknown>).extraBody,
          useStream: (config as Record<string, unknown>).useStream,
          rpmLimit: (config as Record<string, unknown>).rpmLimit,
          maxRetries: (config as Record<string, unknown>).maxRetries
        },
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

  function stripLegacyOpenAiMirrorFields(): void {
    const translation = settings.value.translation as Record<string, unknown>
    delete translation.rpmLimit
    delete translation.maxRetries
    delete translation.isJsonMode
    delete translation.useStream
    delete translation.extraBody

    const hq = settings.value.hqTranslation as Record<string, unknown>
    delete hq.rpmLimit
    delete hq.maxRetries
    delete hq.forceJsonOutput
    delete hq.useStream
    delete hq.extraBody

    const pluginAgent = settings.value.pluginAgent as Record<string, unknown>
    delete pluginAgent.rpmLimit
    delete pluginAgent.maxRetries
    delete pluginAgent.forceJsonOutput
    delete pluginAgent.useStream
    delete pluginAgent.extraBody

    const aiVision = settings.value.aiVisionOcr as Record<string, unknown>
    delete aiVision.rpmLimit
    delete aiVision.maxRetries
    delete aiVision.isJsonMode
    delete aiVision.forceJsonOutput
    delete aiVision.useStream
    delete aiVision.extraBody

    settings.value.proofreading.rounds.forEach((round) => {
      const target = round as Record<string, unknown>
      delete target.rpmLimit
      delete target.maxRetries
      delete target.forceJsonOutput
      delete target.useStream
      delete target.extraBody
      delete target.showApiKey
    })

    for (const config of Object.values(providerConfigs.value.translation)) {
      const target = config as Record<string, unknown>
      delete target.rpmLimit
      delete target.maxRetries
      delete target.isJsonMode
      delete target.forceJsonOutput
      delete target.useStream
      delete target.extraBody
    }

    for (const config of Object.values(providerConfigs.value.hqTranslation)) {
      const target = config as Record<string, unknown>
      delete target.rpmLimit
      delete target.maxRetries
      delete target.forceJsonOutput
      delete target.useStream
      delete target.extraBody
    }

    for (const config of Object.values(providerConfigs.value.pluginAgent)) {
      const target = config as Record<string, unknown>
      delete target.rpmLimit
      delete target.maxRetries
      delete target.forceJsonOutput
      delete target.useStream
      delete target.extraBody
    }

    for (const config of Object.values(providerConfigs.value.aiVisionOcr)) {
      const target = config as Record<string, unknown>
      delete target.rpmLimit
      delete target.maxRetries
      delete target.isJsonMode
      delete target.forceJsonOutput
      delete target.useStream
      delete target.extraBody
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

  const translationConstraintsModule = useTranslationConstraintsSettings(
    settings,
    saveToStorage
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
    } else if (category === 'pluginAgent') {
      pluginAgentModule.savePluginAgentProviderConfig(provider)
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
    } else if (category === 'pluginAgent') {
      pluginAgentModule.restorePluginAgentProviderConfig(provider)
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
        ensureNumericTypes()

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

    if ((backendSettings.settingsSchemaVersion as number | undefined) >= 2) {
      const defaults = createDefaultSettings()
      const mergedSettings = deepMerge(
        defaults,
        stripDeprecatedSettingsFields(backendSettings) as Partial<TranslationSettings>
      )
      settings.value = mergedSettings
      const nestedProviderConfigs = backendSettings.providerConfigs
      if (nestedProviderConfigs && typeof nestedProviderConfigs === 'object') {
        providerConfigs.value = {
          translation: (nestedProviderConfigs as ProviderConfigsCache).translation || {},
          hqTranslation: (nestedProviderConfigs as ProviderConfigsCache).hqTranslation || {},
          pluginAgent: (nestedProviderConfigs as ProviderConfigsCache).pluginAgent || {},
          aiVisionOcr: (nestedProviderConfigs as ProviderConfigsCache).aiVisionOcr || {}
        }
      }
      normalizeProviderAliases()
      normalizeProviderConfigAliases()
      normalizeProviderConfigOpenAiOptions()
      return
    }

    // OCR 设置
    if (backendSettings.ocrEngine) {
      settings.value.ocrEngine = backendSettings.ocrEngine as OcrEngine
    }
    if (backendSettings.sourceLanguage) {
      settings.value.sourceLanguage = backendSettings.sourceLanguage as string
    }
    if (backendSettings.textDetector) {
      settings.value.textDetector = normalizeTextDetector(backendSettings.textDetector)
    }
    if (backendSettings.enableAuxYoloDetection !== undefined) {
      settings.value.enableAuxYoloDetection = backendSettings.enableAuxYoloDetection as boolean
    }
    if (backendSettings.auxYoloConfThreshold !== undefined) {
      settings.value.auxYoloConfThreshold = Number(backendSettings.auxYoloConfThreshold)
    }
    if (backendSettings.auxYoloOverlapThreshold !== undefined) {
      settings.value.auxYoloOverlapThreshold = Number(backendSettings.auxYoloOverlapThreshold)
    }
    if (backendSettings.enableSaberYoloRefine !== undefined) {
      settings.value.enableSaberYoloRefine = backendSettings.enableSaberYoloRefine as boolean
    }
    if (backendSettings.saberYoloRefineOverlapThreshold !== undefined) {
      settings.value.saberYoloRefineOverlapThreshold = Number(backendSettings.saberYoloRefineOverlapThreshold)
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

    // PaddleOCR-VL 设置
    if (backendSettings.paddleOcrVlSourceLanguage) {
      settings.value.paddleOcrVl.sourceLanguage = backendSettings.paddleOcrVlSourceLanguage as string
    }

    // AI 视觉 OCR 设置
    if (backendSettings.aiVisionProvider) {
      settings.value.aiVisionOcr.provider = normalizeProviderId(backendSettings.aiVisionProvider as string)
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
      settings.value.aiVisionOcr.openaiOptions.execution.rpmLimit = parseNum(backendSettings.rpmAiVisionOcr, DEFAULT_RPM_AI_VISION_OCR)
    }
    if (backendSettings.aiVisionPromptModeSelect === 'json') {
      settings.value.aiVisionOcr.promptMode = 'json'
    } else if (backendSettings.aiVisionPromptModeSelect === 'paddleocr_vl') {
      settings.value.aiVisionOcr.promptMode = 'paddleocr_vl'
    } else if (backendSettings.aiVisionPromptModeSelect === 'normal') {
      settings.value.aiVisionOcr.promptMode = 'normal'
    } else {
      settings.value.aiVisionOcr.promptMode = inferAiVisionPromptMode(
        settings.value.aiVisionOcr.prompt,
        settings.value.aiVisionOcr.openaiOptions.request.forceJsonOutput
      )
    }
    settings.value.aiVisionOcr.openaiOptions.request.forceJsonOutput = settings.value.aiVisionOcr.promptMode === 'json'
    if (backendSettings.aiVisionMinImageSize !== undefined) {
      settings.value.aiVisionOcr.minImageSize = parseNum(backendSettings.aiVisionMinImageSize, DEFAULT_AI_VISION_OCR_MIN_IMAGE_SIZE)
    }
    if (backendSettings.enableHybridOcr !== undefined) {
      settings.value.hybridOcr.enabled = backendSettings.enableHybridOcr as boolean
    }
    if (backendSettings.secondaryOcrEngine) {
      settings.value.hybridOcr.secondaryEngine = backendSettings.secondaryOcrEngine as any
    }
    const hybridThresholdFromBackend =
      backendSettings.hybridOcrConfidenceThreshold
      ?? backendSettings.ocrConfidenceThreshold48px
      ?? backendSettings.ocrConfidenceThresholdMangaOcr
      ?? backendSettings.ocrConfidenceThresholdPaddleOcr
    if (hybridThresholdFromBackend !== undefined) {
      settings.value.hybridOcr.confidenceThreshold = parseNum(hybridThresholdFromBackend, 0.2)
    }
    const normalizedHybrid = normalizeHybridOcrConfig(
      settings.value.ocrEngine,
      settings.value.hybridOcr,
      {
        preferRecommendedOrder: settings.value.hybridOcr.enabled
      }
    )
    settings.value.ocrEngine = normalizedHybrid.primaryEngine
    settings.value.hybridOcr = normalizedHybrid.hybrid

    // 翻译服务设置
    if (backendSettings.modelProvider) {
      settings.value.translation.provider = normalizeProviderId(backendSettings.modelProvider as string) as TranslationProvider
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
      settings.value.translation.openaiOptions.execution.rpmLimit = parseNum(backendSettings.rpmTranslation, DEFAULT_RPM_TRANSLATION)
    }
    if (backendSettings.translationMaxRetries !== undefined) {
      settings.value.translation.openaiOptions.execution.businessRetries = parseNum(backendSettings.translationMaxRetries, DEFAULT_TRANSLATION_MAX_RETRIES)
    }
    if (backendSettings.translatePromptModeSelect === 'json') {
      settings.value.translation.openaiOptions.request.forceJsonOutput = true
    }
    if (backendSettings.translationMode) {
      settings.value.translation.translationMode = backendSettings.translationMode as 'batch' | 'single'
    }

    // 目标语言
    if (backendSettings.targetLanguage) {
      settings.value.targetLanguage = backendSettings.targetLanguage as string
    }

    // 4个独立的提示词字段
    if (backendSettings.batchNormalPrompt) {
      settings.value.translation.batchNormalPrompt = backendSettings.batchNormalPrompt as string
    }
    if (backendSettings.batchJsonPrompt) {
      settings.value.translation.batchJsonPrompt = backendSettings.batchJsonPrompt as string
    }
    if (backendSettings.singleNormalPrompt) {
      settings.value.translation.singleNormalPrompt = backendSettings.singleNormalPrompt as string
    }
    if (backendSettings.singleJsonPrompt) {
      settings.value.translation.singleJsonPrompt = backendSettings.singleJsonPrompt as string
    }
    // 确保 translatePrompt 与当前翻译模式和 JSON 模式同步（4个独立存储字段之一）
    const t = settings.value.translation
    if (t.translationMode === 'single') {
      settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
    } else {
      settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
    }
    if (backendSettings.enableTextboxPrompt !== undefined) {
      settings.value.useTextboxPrompt = backendSettings.enableTextboxPrompt as boolean
    }
    if (backendSettings.textboxPromptContent) {
      settings.value.textboxPrompt = backendSettings.textboxPromptContent as string
    }

    // 高质量翻译设置
    if (backendSettings.hqTranslateProvider) {
      settings.value.hqTranslation.provider = normalizeProviderId(backendSettings.hqTranslateProvider as string) as HqTranslationProvider
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
    if (backendSettings.hqRpmLimit !== undefined) {
      settings.value.hqTranslation.openaiOptions.execution.rpmLimit = parseNum(backendSettings.hqRpmLimit, 7)
    }
    if (backendSettings.hqMaxRetries !== undefined) {
      settings.value.hqTranslation.openaiOptions.execution.businessRetries = parseNum(backendSettings.hqMaxRetries, DEFAULT_HQ_TRANSLATION_MAX_RETRIES)
    }
    if (backendSettings.hqPrompt) {
      settings.value.hqTranslation.prompt = backendSettings.hqPrompt as string
    }
    if (backendSettings.hqForceJsonOutput !== undefined) {
      settings.value.hqTranslation.openaiOptions.request.forceJsonOutput = backendSettings.hqForceJsonOutput as boolean
    }
    if (backendSettings.hqUseStream !== undefined) {
      settings.value.hqTranslation.openaiOptions.execution.useStream = backendSettings.hqUseStream as boolean
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
            provider: normalizeProviderId((round.provider as string) || 'siliconflow') as HqTranslationProvider,
            apiKey: (round.apiKey as string) || '',
          modelName: (round.modelName as string) || '',
          customBaseUrl: (round.customBaseUrl as string) || '',
          prompt: (round.prompt as string) || '',
          batchSize: parseNum(round.batchSize, 3),
          openaiOptions: {
            request: {
              forceJsonOutput: (round.forceJsonOutput as boolean) || false
            },
            execution: {
              useStream: round.useStream !== undefined ? (round.useStream as boolean) : true,
              rpmLimit: parseNum(round.rpmLimit, 7),
              transportRetries: 1,
              businessRetries: parseNum(round.maxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
            }
          }
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

    // 精确掩膜设置（常驻启用，无开关）
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

    // 书架模式自动保存
    if (backendSettings.autoSaveInBookshelfMode !== undefined) {
      settings.value.autoSaveInBookshelfMode = backendSettings.autoSaveInBookshelfMode as boolean
    }

    // 消除文字模式OCR
    if (backendSettings.removeTextWithOcr !== undefined) {
      settings.value.removeTextWithOcr = backendSettings.removeTextWithOcr as boolean
    }

    // 详细日志
    if (backendSettings.enableVerboseLogs !== undefined) {
      settings.value.enableVerboseLogs = backendSettings.enableVerboseLogs as boolean
    }

    // LAMA修复禁用缩放设置
    if (backendSettings.lamaDisableResize !== undefined) {
      settings.value.lamaDisableResize = backendSettings.lamaDisableResize as boolean
    }

    // 服务商配置缓存
    if (backendSettings.providerSettings && typeof backendSettings.providerSettings === 'object') {
      const providerSettings = backendSettings.providerSettings as Record<string, Record<string, Record<string, unknown>>>

      if (providerSettings.modelProvider) {
        for (const [provider, config] of Object.entries(providerSettings.modelProvider)) {
          providerConfigs.value.translation[normalizeProviderId(provider)] = {
            apiKey: config.apiKey as string,
            modelName: config.modelName as string,
            customBaseUrl: config.customBaseUrl as string,
            openaiOptions: {
              request: {
                forceJsonOutput: config.translatePromptModeSelect === 'json'
              },
              execution: {
                useStream: false,
                rpmLimit: parseNum(config.rpmTranslation, DEFAULT_RPM_TRANSLATION),
                transportRetries: 1,
                businessRetries: parseNum(config.translationMaxRetries, DEFAULT_TRANSLATION_MAX_RETRIES)
              }
            }
          }
        }
      }

      if (providerSettings.hqTranslateProvider) {
        for (const [provider, config] of Object.entries(providerSettings.hqTranslateProvider)) {
          providerConfigs.value.hqTranslation[normalizeProviderId(provider)] = {
            apiKey: config.hqApiKey as string,
            modelName: config.hqModelName as string,
            customBaseUrl: config.hqCustomBaseUrl as string,
            batchSize: parseNum(config.hqBatchSize, 3),
            openaiOptions: {
              request: {
                forceJsonOutput: config.hqForceJsonOutput as boolean
              },
              execution: {
                useStream: config.hqUseStream as boolean,
                rpmLimit: parseNum(config.hqRpmLimit, 7),
                transportRetries: 1,
                businessRetries: parseNum(config.hqMaxRetries, DEFAULT_HQ_TRANSLATION_MAX_RETRIES)
              }
            },
            prompt: config.hqPrompt as string
          }
        }
      }

      if (providerSettings.aiVisionProvider) {
        for (const [provider, config] of Object.entries(providerSettings.aiVisionProvider)) {
          providerConfigs.value.aiVisionOcr[normalizeProviderId(provider)] = {
            apiKey: config.aiVisionApiKey as string,
            modelName: config.aiVisionModelName as string,
            customBaseUrl: config.customAiVisionBaseUrl as string,
            prompt: config.aiVisionOcrPrompt as string,
            promptMode: (
              config.aiVisionPromptModeSelect === 'paddleocr_vl'
                ? 'paddleocr_vl'
                : config.aiVisionPromptModeSelect === 'json'
                  ? 'json'
                  : 'normal'
            ),
            openaiOptions: {
              request: {
                forceJsonOutput: config.aiVisionPromptModeSelect === 'json'
              },
              execution: {
                useStream: false,
                rpmLimit: parseNum(config.rpmAiVisionOcr, DEFAULT_RPM_AI_VISION_OCR),
                transportRetries: 1,
                businessRetries: DEFAULT_TRANSLATION_MAX_RETRIES
              }
            },
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
      stripLegacyOpenAiMirrorFields()

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
      stripLegacyOpenAiMirrorFields()
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

    // 术语表 / 禁翻表模块
    glossary: translationConstraintsModule.glossary,
    nonTranslate: translationConstraintsModule.nonTranslate,
    setGlossaryEnabled: translationConstraintsModule.setGlossaryEnabled,
    setGlossaryEntries: translationConstraintsModule.setGlossaryEntries,
    updateGlossary: translationConstraintsModule.updateGlossary,
    setNonTranslateEnabled: translationConstraintsModule.setNonTranslateEnabled,
    setNonTranslateEntries: translationConstraintsModule.setNonTranslateEntries,
    updateNonTranslate: translationConstraintsModule.updateNonTranslate,

    // 检测设置模块
    setTextDetector: detectionModule.setTextDetector,
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
    saveToBackend,
    savePluginAgentSettings
  }
})

// 导出类型
export type { ProviderConfigsCache } from './types'
