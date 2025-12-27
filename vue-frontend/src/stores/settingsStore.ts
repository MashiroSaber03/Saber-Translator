/**
 * 设置状态管理 Store
 * 管理翻译设置、OCR设置、高质量翻译设置、AI校对设置等
 * 支持 localStorage 持久化和主题切换
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  TranslationSettings,
  TranslationSettingsUpdates,
  TextStyleSettings,
  BaiduOcrSettings,
  AiVisionOcrSettings,
  TranslationServiceSettings,
  HqTranslationSettings,
  ProofreadingSettings,
  ProofreadingRound,
  BoxExpandSettings,
  PreciseMaskSettings,
  OcrEngine,
  TextDetector,
  TranslationProvider,
  HqTranslationProvider,
  NoThinkingMethod,
  PdfProcessingMethod,
  Theme
} from '@/types/settings'
import {
  DEFAULT_FILL_COLOR,
  DEFAULT_STROKE_ENABLED,
  DEFAULT_STROKE_COLOR,
  DEFAULT_STROKE_WIDTH,
  DEFAULT_TRANSLATE_PROMPT,
  DEFAULT_TRANSLATE_JSON_PROMPT,
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT,
  DEFAULT_HQ_TRANSLATE_PROMPT,
  DEFAULT_RPM_TRANSLATION,
  DEFAULT_RPM_AI_VISION_OCR,
  DEFAULT_TRANSLATION_MAX_RETRIES,
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_MAX_RETRIES,
  STORAGE_KEY_TRANSLATION_SETTINGS,
  STORAGE_KEY_THEME,
  STORAGE_KEY_PROVIDER_CONFIGS,
  STORAGE_KEY_MODEL_HISTORY
} from '@/constants'

// ============================================================
// 服务商配置缓存类型定义
// ============================================================

/** 翻译服务配置缓存项 */
interface TranslationProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  rpmLimit?: number
  maxRetries?: number
  isJsonMode?: boolean
}

/** 高质量翻译服务配置缓存项 */
interface HqTranslationProviderConfig {
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
interface AiVisionOcrProviderConfig {
  apiKey?: string
  modelName?: string
  customBaseUrl?: string
  prompt?: string
  rpmLimit?: number
  isJsonMode?: boolean
}

/** 服务商配置缓存结构 */
interface ProviderConfigsCache {
  translation: Record<string, TranslationProviderConfig>
  hqTranslation: Record<string, HqTranslationProviderConfig>
  aiVisionOcr: Record<string, AiVisionOcrProviderConfig>
}

// ============================================================
// 默认值定义
// ============================================================

/** 默认文字样式设置 */
const DEFAULT_TEXT_STYLE: TextStyleSettings = {
  fontSize: 25,
  autoFontSize: true,
  fontFamily: 'fonts/STSONG.TTF',
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: DEFAULT_FILL_COLOR,
  strokeEnabled: DEFAULT_STROKE_ENABLED,
  strokeColor: DEFAULT_STROKE_COLOR,
  strokeWidth: DEFAULT_STROKE_WIDTH,
  inpaintMethod: 'solid'
}

/** 默认百度OCR设置 */
const DEFAULT_BAIDU_OCR: BaiduOcrSettings = {
  apiKey: '',
  secretKey: '',
  version: 'standard',
  sourceLanguage: 'JAP'
}

/** 默认AI视觉OCR设置 */
const DEFAULT_AI_VISION_OCR: AiVisionOcrSettings = {
  provider: 'gemini',
  apiKey: '',
  modelName: '',
  prompt: DEFAULT_AI_VISION_OCR_PROMPT,
  rpmLimit: DEFAULT_RPM_AI_VISION_OCR,
  customBaseUrl: '',
  isJsonMode: false
}


/** 默认翻译服务设置 */
const DEFAULT_TRANSLATION_SERVICE: TranslationServiceSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  rpmLimit: DEFAULT_RPM_TRANSLATION,
  maxRetries: DEFAULT_TRANSLATION_MAX_RETRIES,
  isJsonMode: false
}

/** 默认高质量翻译设置 */
const DEFAULT_HQ_TRANSLATION: HqTranslationSettings = {
  provider: 'siliconflow',
  apiKey: '',
  modelName: '',
  customBaseUrl: '',
  batchSize: 3,
  sessionReset: 20,
  rpmLimit: 7,
  maxRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  lowReasoning: false,
  noThinkingMethod: 'gemini',
  forceJsonOutput: true,
  useStream: false,
  prompt: DEFAULT_HQ_TRANSLATE_PROMPT
}

/** 默认AI校对设置 */
const DEFAULT_PROOFREADING: ProofreadingSettings = {
  enabled: false,
  rounds: [],
  maxRetries: DEFAULT_PROOFREADING_MAX_RETRIES
}

/** 默认文本框扩展参数 */
const DEFAULT_BOX_EXPAND: BoxExpandSettings = {
  ratio: 0,
  top: 0,
  bottom: 0,
  left: 0,
  right: 0
}

/** 默认精确文字掩膜设置 */
const DEFAULT_PRECISE_MASK: PreciseMaskSettings = {
  enabled: true,
  dilateSize: 10,
  boxExpandRatio: 20
}

/** 创建默认翻译设置 */
function createDefaultSettings(): TranslationSettings {
  return {
    textStyle: { ...DEFAULT_TEXT_STYLE },
    ocrEngine: 'manga_ocr',
    sourceLanguage: 'japanese',
    textDetector: 'default',
    baiduOcr: { ...DEFAULT_BAIDU_OCR },
    aiVisionOcr: { ...DEFAULT_AI_VISION_OCR },
    translation: { ...DEFAULT_TRANSLATION_SERVICE },
    targetLanguage: 'zh',
    translatePrompt: '',
    useTextboxPrompt: false,
    textboxPrompt: '',
    hqTranslation: { ...DEFAULT_HQ_TRANSLATION },
    proofreading: { ...DEFAULT_PROOFREADING },
    boxExpand: { ...DEFAULT_BOX_EXPAND },
    preciseMask: { ...DEFAULT_PRECISE_MASK },
    pdfProcessingMethod: 'backend',
    showDetectionDebug: false
  }
}

// ============================================================
// Store 定义
// ============================================================

export const useSettingsStore = defineStore('settings', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 翻译设置 */
  const settings = ref<TranslationSettings>(createDefaultSettings())

  /** 当前主题 */
  const theme = ref<Theme>('light')

  /** 服务商配置分组存储（用于切换服务商时保存/恢复配置） */
  const providerConfigs = ref<ProviderConfigsCache>({
    translation: {},
    hqTranslation: {},
    aiVisionOcr: {}
  })

  /** 模型使用历史记录（按服务商分组） */
  const modelHistory = ref<Record<string, string[]>>({})

  /** 自定义字号预设 */
  const customFontPresets = ref<number[]>([])

  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前文字样式设置 */
  const textStyle = computed(() => settings.value.textStyle)

  /** 当前OCR引擎 */
  const ocrEngine = computed(() => settings.value.ocrEngine)

  /** 当前源语言 */
  const sourceLanguage = computed(() => settings.value.sourceLanguage)

  /** 当前翻译服务商 */
  const translationProvider = computed(() => settings.value.translation.provider)

  /** 当前高质量翻译服务商 */
  const hqProvider = computed(() => settings.value.hqTranslation.provider)

  /** AI校对是否启用 */
  const isProofreadingEnabled = computed(() => settings.value.proofreading.enabled)

  /** 是否为深色主题 */
  const isDarkTheme = computed(() => theme.value === 'dark')

  // ============================================================
  // 设置更新方法
  // ============================================================

  /**
   * 更新翻译设置
   * @param updates - 要更新的设置
   */
  function updateSettings(updates: TranslationSettingsUpdates): void {
    Object.assign(settings.value, updates)
    saveToStorage()
  }

  /**
   * 更新文字样式设置
   * @param updates - 要更新的样式
   */
  function updateTextStyle(updates: Partial<TextStyleSettings>): void {
    Object.assign(settings.value.textStyle, updates)
    saveToStorage()
  }


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
   * 设置文本检测器
   * @param detector - 检测器类型
   */
  function setTextDetector(detector: TextDetector): void {
    settings.value.textDetector = detector
    saveToStorage()
    console.log(`文本检测器已设置为: ${detector}`)
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
   * 设置文本框提示词
   * @param prompt - 提示词内容
   */
  function setTextboxPrompt(prompt: string): void {
    settings.value.textboxPrompt = prompt
    saveToStorage()
  }

  /**
   * 切换文本框提示词启用状态
   * @param enabled - 是否启用
   */
  function setUseTextboxPrompt(enabled: boolean): void {
    settings.value.useTextboxPrompt = enabled
    saveToStorage()
  }

  /**
   * 设置翻译提示词模式
   * 切换时自动更新当前提示词内容为对应模式的默认提示词
   * @param isJsonMode - 是否为JSON格式模式
   */
  function setTranslatePromptMode(isJsonMode: boolean): void {
    // 更新模式状态
    settings.value.translation.isJsonMode = isJsonMode

    // 根据模式切换默认提示词
    const defaultPrompt = isJsonMode ? DEFAULT_TRANSLATE_JSON_PROMPT : DEFAULT_TRANSLATE_PROMPT
    settings.value.translatePrompt = defaultPrompt

    saveToStorage()
    console.log(`翻译提示词模式已切换为: ${isJsonMode ? 'JSON格式' : '普通模式'}`)
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
  // AI校对设置方法
  // ============================================================

  /**
   * 设置AI校对启用状态
   * @param enabled - 是否启用
   */
  function setProofreadingEnabled(enabled: boolean): void {
    settings.value.proofreading.enabled = enabled
    saveToStorage()
    console.log(`AI校对已${enabled ? '启用' : '禁用'}`)
  }

  /**
   * 添加校对轮次
   * @param round - 校对轮次配置
   */
  function addProofreadingRound(round: ProofreadingRound): void {
    settings.value.proofreading.rounds.push(round)
    saveToStorage()
    console.log(`已添加校对轮次: ${round.name}`)
  }

  /**
   * 更新校对轮次
   * @param index - 轮次索引
   * @param updates - 要更新的配置
   */
  function updateProofreadingRound(index: number, updates: Partial<ProofreadingRound>): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const round = settings.value.proofreading.rounds[index]
      if (round) {
        Object.assign(round, updates)
        saveToStorage()
      }
    }
  }

  /**
   * 删除校对轮次
   * @param index - 轮次索引
   */
  function removeProofreadingRound(index: number): void {
    if (index >= 0 && index < settings.value.proofreading.rounds.length) {
      const removed = settings.value.proofreading.rounds.splice(index, 1)
      saveToStorage()
      console.log(`已删除校对轮次: ${removed[0]?.name}`)
    }
  }

  /**
   * 设置校对重试次数
   * @param maxRetries - 最大重试次数
   */
  function setProofreadingMaxRetries(maxRetries: number): void {
    settings.value.proofreading.maxRetries = maxRetries
    saveToStorage()
  }

  // ============================================================
  // 文本框扩展和精确掩膜设置方法
  // ============================================================

  /**
   * 更新文本框扩展参数
   * @param updates - 要更新的参数
   */
  function updateBoxExpand(updates: Partial<BoxExpandSettings>): void {
    Object.assign(settings.value.boxExpand, updates)
    saveToStorage()
  }

  /**
   * 更新精确文字掩膜设置
   * @param updates - 要更新的设置
   */
  function updatePreciseMask(updates: Partial<PreciseMaskSettings>): void {
    Object.assign(settings.value.preciseMask, updates)
    saveToStorage()
  }

  // ============================================================
  // PDF处理和调试设置方法
  // ============================================================

  /**
   * 设置PDF处理方式
   * @param method - 处理方式
   */
  function setPdfProcessingMethod(method: PdfProcessingMethod): void {
    settings.value.pdfProcessingMethod = method
    saveToStorage()
    console.log(`PDF处理方式已设置为: ${method}`)
  }

  /**
   * 设置检测框调试开关
   * @param show - 是否显示
   */
  function setShowDetectionDebug(show: boolean): void {
    settings.value.showDetectionDebug = show
    saveToStorage()
  }

  // ============================================================
  // 主题切换方法
  // ============================================================

  /**
   * 切换主题
   */
  function toggleTheme(): void {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
    applyTheme()
    saveThemeToStorage()
    console.log(`主题已切换为: ${theme.value}`)
  }

  /**
   * 设置主题
   * @param newTheme - 新主题
   */
  function setTheme(newTheme: Theme): void {
    theme.value = newTheme
    applyTheme()
    saveThemeToStorage()
  }

  /**
   * 应用主题到DOM
   * 统一使用 data-theme 属性
   */
  function applyTheme(): void {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', theme.value)
    }
  }

  // ============================================================
  // 服务商配置分组存储方法
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
      isJsonMode: settings.value.translation.isJsonMode
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
      console.log(`[Settings] 恢复翻译服务商配置: ${provider}`, cached)
    } else {
      // 无缓存时清空配置（保留默认值）
      settings.value.translation.apiKey = ''
      settings.value.translation.modelName = ''
      settings.value.translation.customBaseUrl = ''
      console.log(`[Settings] ${provider} 无缓存配置，使用默认值`)
    }
  }

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
      isJsonMode: settings.value.aiVisionOcr.isJsonMode
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
      console.log(`[Settings] 恢复AI视觉OCR服务商配置: ${provider}`, cached)
    } else {
      // 无缓存时清空配置
      settings.value.aiVisionOcr.apiKey = ''
      settings.value.aiVisionOcr.modelName = ''
      settings.value.aiVisionOcr.customBaseUrl = ''
      console.log(`[Settings] ${provider} 无缓存配置，使用默认值`)
    }
  }

  /**
   * 保存服务商配置（兼容旧接口）
   * @param category - 配置类别（translation/hqTranslation/aiVisionOcr）
   * @param provider - 服务商名称
   */
  function saveProviderConfig(category: string, provider: string): void {
    if (category === 'translation') {
      saveTranslationProviderConfig(provider)
    } else if (category === 'hqTranslation') {
      saveHqProviderConfig(provider)
    } else if (category === 'aiVisionOcr') {
      saveAiVisionOcrProviderConfig(provider)
    }
  }

  /**
   * 恢复服务商配置（兼容旧接口）
   * @param category - 配置类别
   * @param provider - 服务商名称
   */
  function restoreProviderConfig(category: string, provider: string): void {
    if (category === 'translation') {
      restoreTranslationProviderConfig(provider)
    } else if (category === 'hqTranslation') {
      restoreHqProviderConfig(provider)
    } else if (category === 'aiVisionOcr') {
      restoreAiVisionOcrProviderConfig(provider)
    }
  }

  // ============================================================
  // 模型历史记录方法
  // ============================================================

  /**
   * 添加模型到历史记录
   * @param provider - 服务商名称
   * @param modelName - 模型名称
   */
  function addModelToHistory(provider: string, modelName: string): void {
    if (!provider || !modelName) return

    if (!modelHistory.value[provider]) {
      modelHistory.value[provider] = []
    }

    // 如果已存在，先移除
    const index = modelHistory.value[provider].indexOf(modelName)
    if (index !== -1) {
      modelHistory.value[provider].splice(index, 1)
    }

    // 添加到开头（最近使用的在前）
    modelHistory.value[provider].unshift(modelName)

    // 限制历史记录数量（最多保留20个）
    if (modelHistory.value[provider].length > 20) {
      modelHistory.value[provider] = modelHistory.value[provider].slice(0, 20)
    }

    saveModelHistoryToStorage()
    console.log(`[Settings] 添加模型到历史: ${provider} -> ${modelName}`)
  }

  /**
   * 获取服务商的模型历史记录
   * @param provider - 服务商名称
   * @returns 模型名称列表
   */
  function getModelHistory(provider: string): string[] {
    return modelHistory.value[provider] || []
  }

  /**
   * 清除服务商的模型历史记录
   * @param provider - 服务商名称
   */
  function clearModelHistory(provider: string): void {
    if (modelHistory.value[provider]) {
      delete modelHistory.value[provider]
      saveModelHistoryToStorage()
      console.log(`[Settings] 清除模型历史: ${provider}`)
    }
  }


  // ============================================================
  // 自定义字号预设方法
  // ============================================================

  /**
   * 添加自定义字号预设
   * @param fontSize - 字号
   */
  function addCustomFontPreset(fontSize: number): void {
    if (!customFontPresets.value.includes(fontSize)) {
      customFontPresets.value.push(fontSize)
      customFontPresets.value.sort((a, b) => a - b)
      saveCustomFontPresetsToStorage()
      console.log(`已添加自定义字号预设: ${fontSize}`)
    }
  }

  /**
   * 删除自定义字号预设
   * @param fontSize - 字号
   */
  function removeCustomFontPreset(fontSize: number): void {
    const index = customFontPresets.value.indexOf(fontSize)
    if (index !== -1) {
      customFontPresets.value.splice(index, 1)
      saveCustomFontPresetsToStorage()
      console.log(`已删除自定义字号预设: ${fontSize}`)
    }
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
   * 从 localStorage 加载设置
   */
  function loadFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)
      if (data) {
        const parsed = JSON.parse(data)
        // 深度合并，确保新增的默认值不会丢失
        settings.value = deepMerge(createDefaultSettings(), parsed)
        // 【根源修复】确保数值类型正确（防止 localStorage 中的旧数据类型错误）
        ensureNumericTypes()
        console.log('已从 localStorage 加载设置')
      }
    } catch (error) {
      console.error('从 localStorage 加载设置失败:', error)
    }
  }

  /**
   * 确保设置中的数值类型正确
   * 防止从 localStorage 加载的旧数据类型错误导致后端报错
   */
  function ensureNumericTypes(): void {
    const ts = settings.value.textStyle
    ts.fontSize = Number(ts.fontSize) || 25
    ts.strokeWidth = Number(ts.strokeWidth) || 3

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
   * 保存主题到 localStorage
   */
  function saveThemeToStorage(): void {
    try {
      localStorage.setItem(STORAGE_KEY_THEME, theme.value)
    } catch (error) {
      console.error('保存主题到 localStorage 失败:', error)
    }
  }

  /**
   * 从 localStorage 加载主题
   */
  function loadThemeFromStorage(): void {
    try {
      const savedTheme = localStorage.getItem(STORAGE_KEY_THEME)
      if (savedTheme === 'light' || savedTheme === 'dark') {
        theme.value = savedTheme
        applyTheme()
        console.log(`已从 localStorage 加载主题: ${theme.value}`)
      }
    } catch (error) {
      console.error('从 localStorage 加载主题失败:', error)
    }
  }

  /**
   * 保存自定义字号预设到 localStorage
   */
  function saveCustomFontPresetsToStorage(): void {
    try {
      const data = JSON.stringify(customFontPresets.value)
      localStorage.setItem('customFontSizePresets', data)
    } catch (error) {
      console.error('保存自定义字号预设失败:', error)
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
   * 保存模型历史记录到 localStorage
   */
  function saveModelHistoryToStorage(): void {
    try {
      const data = JSON.stringify(modelHistory.value)
      localStorage.setItem(STORAGE_KEY_MODEL_HISTORY, data)
    } catch (error) {
      console.error('保存模型历史记录失败:', error)
    }
  }

  /**
   * 从 localStorage 加载模型历史记录
   */
  function loadModelHistoryFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_MODEL_HISTORY)
      if (data) {
        modelHistory.value = JSON.parse(data)
        console.log('已从 localStorage 加载模型历史记录')
      }
    } catch (error) {
      console.error('加载模型历史记录失败:', error)
    }
  }

  /**
   * 从 localStorage 加载自定义字号预设
   */
  function loadCustomFontPresetsFromStorage(): void {
    try {
      const data = localStorage.getItem('customFontSizePresets')
      if (data) {
        customFontPresets.value = JSON.parse(data)
      }
    } catch (error) {
      console.error('加载自定义字号预设失败:', error)
    }
  }

  /**
   * 初始化设置（从 localStorage 加载）
   */
  function initSettings(): void {
    loadFromStorage()
    loadThemeFromStorage()
    loadCustomFontPresetsFromStorage()
    loadProviderConfigsFromStorage()
    loadModelHistoryFromStorage()
  }

  /**
   * 从后端加载用户设置
   * 后端设置保存在 config/user_settings.json 文件中
   * 这是原版前端使用的配置存储方式
   */
  async function loadFromBackend(): Promise<boolean> {
    try {
      console.log('[Settings] 开始从后端加载设置...')
      const { getUserSettings } = await import('@/api/config')
      const response = await getUserSettings()

      console.log('[Settings] API 响应:', response)

      if (response.success && response.settings) {
        const backendSettings = response.settings
        console.log('[Settings] 从后端加载设置:', backendSettings)
        console.log('[Settings] 设置字段数量:', Object.keys(backendSettings).length)

        // 将后端设置映射到 Vue 版本的设置结构
        applyBackendSettings(backendSettings)

        // 保存到 localStorage 以便下次快速加载
        saveToStorage()
        saveProviderConfigsToStorage()

        console.log('[Settings] 后端设置已应用，当前翻译服务商:', settings.value.translation.provider)
        console.log('[Settings] 当前翻译 API Key:', settings.value.translation.apiKey ? '已设置' : '未设置')
        return true
      } else {
        console.warn('[Settings] 后端无设置数据，使用 localStorage 或默认值', response)
        return false
      }
    } catch (error) {
      console.error('[Settings] 从后端加载设置失败:', error)
      return false
    }
  }

  /**
   * 将后端设置应用到当前设置
   * 处理原版前端和 Vue 版本之间的字段映射
   * 字段名参考 config/user_settings.json 的实际格式
   */
  function applyBackendSettings(backendSettings: Record<string, unknown>): void {
    // 辅助函数：解析数字（后端可能存储为字符串）
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

    // 百度 OCR 设置（原版字段名）
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

    // AI 视觉 OCR 设置（原版字段名）
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

    // 翻译服务设置（原版字段名）
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

    // 翻译提示词（原版字段名）
    if (backendSettings.promptContent) {
      settings.value.translatePrompt = backendSettings.promptContent as string
    }
    if (backendSettings.enableTextboxPrompt !== undefined) {
      settings.value.useTextboxPrompt = backendSettings.enableTextboxPrompt as boolean
    }
    if (backendSettings.textboxPromptContent) {
      settings.value.textboxPrompt = backendSettings.textboxPromptContent as string
    }

    // 高质量翻译设置（原版字段名）
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

    // AI 校对设置（原版字段名）
    if (backendSettings.proofreadingEnabled !== undefined) {
      settings.value.proofreading.enabled = backendSettings.proofreadingEnabled as boolean
    }
    if (backendSettings.proofreadingMaxRetries !== undefined) {
      settings.value.proofreading.maxRetries = parseNum(backendSettings.proofreadingMaxRetries, DEFAULT_PROOFREADING_MAX_RETRIES)
    }
    // 校对轮次配置
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

    // 文本框扩展设置（原版字段名）
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

    // 精确掩膜设置（原版字段名）
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
      settings.value.pdfProcessingMethod = backendSettings.pdfProcessingMethod as PdfProcessingMethod
    }

    // 调试设置
    if (backendSettings.showDetectionDebug !== undefined) {
      settings.value.showDetectionDebug = backendSettings.showDetectionDebug as boolean
    }

    // 服务商配置缓存（providerSettings）
    if (backendSettings.providerSettings && typeof backendSettings.providerSettings === 'object') {
      const providerSettings = backendSettings.providerSettings as Record<string, Record<string, Record<string, unknown>>>

      // 翻译服务商配置
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

      // 高质量翻译服务商配置
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

      // AI 视觉 OCR 服务商配置
      if (providerSettings.aiVisionProvider) {
        for (const [provider, config] of Object.entries(providerSettings.aiVisionProvider)) {
          providerConfigs.value.aiVisionOcr[provider] = {
            apiKey: config.aiVisionApiKey as string,
            modelName: config.aiVisionModelName as string,
            customBaseUrl: config.customAiVisionBaseUrl as string,
            prompt: config.aiVisionOcrPrompt as string,
            rpmLimit: parseNum(config.rpmAiVisionOcr, DEFAULT_RPM_AI_VISION_OCR),
            isJsonMode: config.aiVisionPromptModeSelect === 'json'
          }
        }
      }
    }

    console.log('[Settings] 后端设置映射完成')
  }

  /**
   * 构建服务商分组配置用于保存到后端
   * 复刻原版 providerSettingsCache 的结构
   * 使用原版字段名以保持兼容性
   */
  function buildProviderSettingsForBackend(): Record<string, Record<string, Record<string, unknown>>> {
    // 初始化结果对象
    const modelProviderConfigs: Record<string, Record<string, unknown>> = {}
    const hqTranslateProviderConfigs: Record<string, Record<string, unknown>> = {}
    const aiVisionProviderConfigs: Record<string, Record<string, unknown>> = {}

    // 翻译服务商配置（modelProvider）
    for (const [provider, config] of Object.entries(providerConfigs.value.translation)) {
      modelProviderConfigs[provider] = {
        apiKey: config.apiKey || '',
        modelName: config.modelName || '',
        customBaseUrl: config.customBaseUrl || '',
        rpmTranslation: String(config.rpmLimit || 0),
        translationMaxRetries: String(config.maxRetries || 3)
      }
    }

    // 高质量翻译服务商配置（hqTranslateProvider）
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

    // AI 视觉 OCR 服务商配置（aiVisionProvider）
    for (const [provider, config] of Object.entries(providerConfigs.value.aiVisionOcr)) {
      aiVisionProviderConfigs[provider] = {
        aiVisionApiKey: config.apiKey || '',
        aiVisionModelName: config.modelName || '',
        customAiVisionBaseUrl: config.customBaseUrl || '',
        aiVisionOcrPrompt: config.prompt || '',
        rpmAiVisionOcr: String(config.rpmLimit || 0),
        aiVisionPromptModeSelect: config.isJsonMode ? 'json' : 'normal'
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
   * 将当前设置转换为后端格式并保存到 config/user_settings.json
   * 使用与原版前端相同的字段名
   */
  async function saveToBackend(): Promise<boolean> {
    try {
      const { saveUserSettings } = await import('@/api/config')

      // 复刻原版 saveAllCurrentProviderSettings：保存当前所有服务商的配置到缓存
      // 确保当前选中的服务商配置也被保存
      saveTranslationProviderConfig(settings.value.translation.provider)
      saveHqProviderConfig(settings.value.hqTranslation.provider)
      saveAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

      // 将 Vue 版本的设置转换为后端格式（使用原版字段名）
      const backendSettings: Record<string, unknown> = {
        // OCR 设置
        ocrEngine: settings.value.ocrEngine,
        sourceLanguage: settings.value.sourceLanguage,
        textDetector: settings.value.textDetector,

        // 百度 OCR（原版字段名）
        baiduApiKey: settings.value.baiduOcr.apiKey,
        baiduSecretKey: settings.value.baiduOcr.secretKey,
        baiduVersion: settings.value.baiduOcr.version,
        baiduSourceLanguage: settings.value.baiduOcr.sourceLanguage,

        // AI 视觉 OCR（原版字段名）
        aiVisionProvider: settings.value.aiVisionOcr.provider,
        aiVisionApiKey: settings.value.aiVisionOcr.apiKey,
        aiVisionModelName: settings.value.aiVisionOcr.modelName,
        aiVisionOcrPrompt: settings.value.aiVisionOcr.prompt,
        customAiVisionBaseUrl: settings.value.aiVisionOcr.customBaseUrl,
        rpmAiVisionOcr: String(settings.value.aiVisionOcr.rpmLimit),
        aiVisionPromptModeSelect: settings.value.aiVisionOcr.isJsonMode ? 'json' : 'normal',

        // 翻译服务（原版字段名）
        modelProvider: settings.value.translation.provider,
        apiKey: settings.value.translation.apiKey,
        modelName: settings.value.translation.modelName,
        customBaseUrl: settings.value.translation.customBaseUrl,
        rpmTranslation: String(settings.value.translation.rpmLimit),
        translationMaxRetries: String(settings.value.translation.maxRetries),
        translatePromptModeSelect: settings.value.translation.isJsonMode ? 'json' : 'normal',

        // 翻译提示词（原版字段名）
        promptContent: settings.value.translatePrompt,
        enableTextboxPrompt: settings.value.useTextboxPrompt,
        textboxPromptContent: settings.value.textboxPrompt,

        // 高质量翻译（原版字段名）
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

        // AI 校对（原版字段名）
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
            lowReasoning: round.lowReasoning,
            forceJsonOutput: round.forceJsonOutput
          }))
        },

        // 文本框扩展（原版字段名）
        boxExpandRatio: String(settings.value.boxExpand.ratio),
        boxExpandTop: String(settings.value.boxExpand.top),
        boxExpandBottom: String(settings.value.boxExpand.bottom),
        boxExpandLeft: String(settings.value.boxExpand.left),
        boxExpandRight: String(settings.value.boxExpand.right),

        // 精确掩膜（原版字段名）
        usePreciseMask: settings.value.preciseMask.enabled,
        maskDilateSize: String(settings.value.preciseMask.dilateSize),
        maskBoxExpandRatio: String(settings.value.preciseMask.boxExpandRatio || 0),

        // PDF 处理方式
        pdfProcessingMethod: settings.value.pdfProcessingMethod,

        // 调试
        showDetectionDebug: settings.value.showDetectionDebug,

        // ===== 服务商分组配置缓存（复刻原版 providerSettingsCache）=====
        // 保存所有服务商的配置，实现切换服务商时的配置记忆
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

  /**
   * 重置所有设置为默认值
   */
  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
    saveToStorage()
    console.log('设置已重置为默认值')
  }

  // ============================================================
  // 工具函数
  // ============================================================

  /**
   * 深度合并对象
   * @param target - 目标对象
   * @param source - 源对象
   * @returns 合并后的对象
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
          // 对于嵌套对象，进行浅合并
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
  // 返回 Store
  // ============================================================

  return {
    // 状态
    settings,
    theme,
    providerConfigs,
    customFontPresets,

    // 计算属性
    textStyle,
    ocrEngine,
    sourceLanguage,
    translationProvider,
    hqProvider,
    isProofreadingEnabled,
    isDarkTheme,

    // 设置更新方法
    updateSettings,
    updateTextStyle,
    setOcrEngine,
    setTextDetector,
    setSourceLanguage,
    updateBaiduOcr,
    updateAiVisionOcr,

    // 翻译服务设置方法
    setTranslationProvider,
    updateTranslationService,
    setTranslatePrompt,
    setTextboxPrompt,
    setUseTextboxPrompt,
    setTranslatePromptMode,
    setAiVisionOcrPromptMode,

    // 高质量翻译设置方法
    setHqProvider,
    updateHqTranslation,
    setHqUseStream,
    setHqNoThinkingMethod,
    setHqForceJsonOutput,

    // AI校对设置方法
    setProofreadingEnabled,
    addProofreadingRound,
    updateProofreadingRound,
    removeProofreadingRound,
    setProofreadingMaxRetries,

    // 文本框扩展和精确掩膜设置方法
    updateBoxExpand,
    updatePreciseMask,

    // PDF处理和调试设置方法
    setPdfProcessingMethod,
    setShowDetectionDebug,

    // 主题切换方法
    toggleTheme,
    setTheme,
    applyTheme,

    // 服务商配置分组存储方法
    saveProviderConfig,
    restoreProviderConfig,
    saveTranslationProviderConfig,
    restoreTranslationProviderConfig,
    saveHqProviderConfig,
    restoreHqProviderConfig,
    saveAiVisionOcrProviderConfig,
    restoreAiVisionOcrProviderConfig,
    setAiVisionOcrProvider,

    // 模型历史记录方法
    modelHistory,
    addModelToHistory,
    getModelHistory,
    clearModelHistory,

    // 自定义字号预设方法
    addCustomFontPreset,
    removeCustomFontPreset,

    // localStorage 持久化方法
    saveToStorage,
    loadFromStorage,
    loadThemeFromStorage,
    initSettings,
    resetToDefaults,

    // 后端设置同步方法
    loadFromBackend,
    saveToBackend
  }
})
