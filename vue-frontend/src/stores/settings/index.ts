import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  TranslationSettings,
  TextDetector,
  HqTranslationProvider,
  TranslationMode,
  OpenAICompatibleOptions,
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
import { getUserSettings, saveUserSettings } from '@/api/config'
import { normalizeHybridOcrConfig } from '@/utils/hybridOcr'
import {
  cloneOpenAiOptions,
  normalizeOpenAiOptions
} from '@/utils/openaiOptions'
import { deepClone } from '@/utils/deepClone'

import type { ProviderConfigsCache } from './types'
import { createDefaultSettings } from './defaults'
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

type PlainRecord = Record<string, unknown>
type AiVisionPromptMode = TranslationSettings['aiVisionOcr']['promptMode']
type ThemePreference = 'light' | 'dark' | 'system'
type EffectiveTheme = 'light' | 'dark'

function isPlainRecord(value: unknown): value is PlainRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function parseString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function isTranslationMode(value: unknown): value is TranslationMode {
  return value === 'batch' || value === 'single'
}

function isAiVisionPromptMode(value: unknown): value is AiVisionPromptMode {
  return value === 'normal' || value === 'json' || value === 'paddleocr_vl'
}

function parseCurrentOpenAiOptions(
  value: unknown,
  defaults: OpenAICompatibleOptions
): OpenAICompatibleOptions | null {
  if (!isPlainRecord(value) || !isPlainRecord(value.request) || !isPlainRecord(value.execution)) {
    return null
  }

  const forceJsonOutput = parseBoolean(value.request.forceJsonOutput)
  const useStream = parseBoolean(value.execution.useStream)
  const rpmLimit = parseNumber(value.execution.rpmLimit)
  const transportRetries = parseNumber(value.execution.transportRetries)
  const businessRetries = parseNumber(value.execution.businessRetries)
  if (
    forceJsonOutput === null ||
    useStream === null ||
    rpmLimit === null ||
    transportRetries === null ||
    businessRetries === null
  ) {
    return null
  }

  const request: OpenAICompatibleOptions['request'] = { forceJsonOutput }
  if (value.request.temperature !== undefined) {
    const temperature = parseNumber(value.request.temperature)
    if (temperature === null) return null
    request.temperature = temperature
  } else if (defaults.request.temperature !== undefined) {
    request.temperature = defaults.request.temperature
  }
  if (value.request.extraBody !== undefined) {
    if (!isPlainRecord(value.request.extraBody)) return null
    request.extraBody = deepClone(value.request.extraBody)
  }

  return {
    request,
    execution: {
      useStream,
      rpmLimit,
      transportRetries,
      businessRetries,
    },
  }
}

function sanitizeByTemplate(value: unknown, template: unknown, path = ''): unknown | null {
  if (path.endsWith('.openaiOptions')) {
    return parseCurrentOpenAiOptions(value, template as OpenAICompatibleOptions)
  }

  if (Array.isArray(template)) {
    return Array.isArray(value) ? deepClone(value) : null
  }

  if (isPlainRecord(template)) {
    if (!isPlainRecord(value)) return null
    const result: PlainRecord = {}
    for (const key of Object.keys(template)) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) return null
      const sanitized = sanitizeByTemplate(value[key], template[key], `${path}.${key}`)
      if (sanitized === null) return null
      result[key] = sanitized
    }
    return result
  }

  if (typeof template === 'number') return parseNumber(value)
  if (typeof template === 'boolean') return parseBoolean(value)
  if (typeof template === 'string') return parseString(value)
  return value === template ? value : null
}

function isCurrentProviderId(provider: unknown): provider is string {
  return typeof provider === 'string'
    && provider === normalizeProviderId(provider)
    && Boolean(getProviderManifest(provider))
}

function sanitizeProofreadingRounds(value: unknown): TranslationSettings['proofreading']['rounds'] | null {
  if (!Array.isArray(value)) return null
  const rounds: TranslationSettings['proofreading']['rounds'] = []
  for (const round of value) {
    if (!isPlainRecord(round)) return null
    if (!isCurrentProviderId(round.provider)) return null
    const name = parseString(round.name)
    const apiKey = parseString(round.apiKey)
    const modelName = parseString(round.modelName)
    const customBaseUrl = parseString(round.customBaseUrl)
    const prompt = parseString(round.prompt)
    const batchSize = parseNumber(round.batchSize)
    const openaiOptions = parseCurrentOpenAiOptions(round.openaiOptions, createDefaultSettings().hqTranslation.openaiOptions)
    if (
      name === null ||
      apiKey === null ||
      modelName === null ||
      customBaseUrl === null ||
      prompt === null ||
      batchSize === null ||
      openaiOptions === null
    ) {
      return null
    }
    rounds.push({
      name,
      provider: round.provider as HqTranslationProvider,
      apiKey,
      modelName,
      customBaseUrl,
      openaiOptions,
      batchSize,
      prompt,
    })
  }
  return rounds
}

function parseCurrentSettings(value: unknown): TranslationSettings | null {
  if (!isPlainRecord(value) || value.settingsSchemaVersion !== 3) {
    return null
  }

  const defaults = createDefaultSettings()
  const sanitized = sanitizeByTemplate(value, defaults) as TranslationSettings | null
  if (!sanitized) return null

  if (!isCurrentProviderId(sanitized.translation.provider)) return null
  if (!isCurrentProviderId(sanitized.hqTranslation.provider)) return null
  if (!isCurrentProviderId(sanitized.pluginAgent.provider)) return null
  if (!isCurrentProviderId(sanitized.aiVisionOcr.provider)) return null
  if (!isTranslationMode(sanitized.translation.translationMode)) return null
  if (!isAiVisionPromptMode(sanitized.aiVisionOcr.promptMode)) return null
  const rounds = sanitizeProofreadingRounds((value.proofreading as PlainRecord).rounds)
  if (!rounds) return null
  sanitized.proofreading.rounds = rounds

  return sanitized
}

function sanitizeProviderConfig(
  value: unknown,
  openaiDefaults: OpenAICompatibleOptions
): PlainRecord | null {
  if (!isPlainRecord(value)) return null
  const result: PlainRecord = {}
  for (const key of ['apiKey', 'modelName', 'customBaseUrl', 'prompt', 'translationMode', 'promptMode']) {
    if (value[key] === undefined) continue
    const parsed = parseString(value[key])
    if (parsed === null) return null
    if (key === 'translationMode' && !isTranslationMode(parsed)) return null
    if (key === 'promptMode' && !isAiVisionPromptMode(parsed)) return null
    result[key] = parsed
  }
  for (const key of ['batchSize', 'minImageSize']) {
    if (value[key] === undefined) continue
    const parsed = parseNumber(value[key])
    if (parsed === null) return null
    result[key] = parsed
  }
  if (value.openaiOptions !== undefined) {
    const openaiOptions = parseCurrentOpenAiOptions(value.openaiOptions, openaiDefaults)
    if (!openaiOptions) return null
    result.openaiOptions = openaiOptions
  }
  return result
}

function parseCurrentProviderConfigs(value: unknown): ProviderConfigsCache | null {
  if (!isPlainRecord(value)) return null
  const defaults = createDefaultSettings()
  const requiredGroups = ['translation', 'hqTranslation', 'pluginAgent', 'aiVisionOcr'] as const
  for (const group of requiredGroups) {
    if (!isPlainRecord(value[group])) return null
  }

  const parseGroup = (
    group: (typeof requiredGroups)[number],
    openaiDefaults: OpenAICompatibleOptions
  ) => {
    const records: Record<string, PlainRecord> = {}
    for (const [provider, config] of Object.entries(value[group] as PlainRecord)) {
      if (!isCurrentProviderId(provider)) continue
      const parsed = sanitizeProviderConfig(config, openaiDefaults)
      if (!parsed) continue
      records[provider] = parsed
    }
    return records
  }

  return {
    translation: parseGroup('translation', defaults.translation.openaiOptions),
    hqTranslation: parseGroup('hqTranslation', defaults.hqTranslation.openaiOptions),
    pluginAgent: parseGroup('pluginAgent', defaults.pluginAgent.openaiOptions),
    aiVisionOcr: parseGroup('aiVisionOcr', defaults.aiVisionOcr.openaiOptions),
  } as ProviderConfigsCache
}

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<TranslationSettings>(createDefaultSettings())
  const theme = ref<ThemePreference>('light')
  const effectiveTheme = ref<EffectiveTheme>('light')
  const providerConfigs = ref<ProviderConfigsCache>({
    translation: {},
    hqTranslation: {},
    pluginAgent: {},
    aiVisionOcr: {}
  })

  let systemThemeMedia: MediaQueryList | null = null

  function resolveSystemTheme(): EffectiveTheme {
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return 'light'
  }

  function applyEffectiveTheme(nextTheme: EffectiveTheme): void {
    effectiveTheme.value = nextTheme
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', nextTheme)
      document.body.setAttribute('data-theme', nextTheme)
    }
  }

  function handleSystemThemeChange(event: MediaQueryListEvent): void {
    if (theme.value === 'system') {
      applyEffectiveTheme(event.matches ? 'dark' : 'light')
    }
  }

  function detachSystemThemeListener(): void {
    if (!systemThemeMedia) return
    if (typeof systemThemeMedia.removeEventListener === 'function') {
      systemThemeMedia.removeEventListener('change', handleSystemThemeChange)
    } else {
      systemThemeMedia.removeListener(handleSystemThemeChange)
    }
    systemThemeMedia = null
  }

  function attachSystemThemeListener(): void {
    detachSystemThemeListener()
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    systemThemeMedia = window.matchMedia('(prefers-color-scheme: dark)')
    if (typeof systemThemeMedia.addEventListener === 'function') {
      systemThemeMedia.addEventListener('change', handleSystemThemeChange)
    } else {
      systemThemeMedia.addListener(handleSystemThemeChange)
    }
  }

  function applyThemePreference(): void {
    if (theme.value === 'system') {
      attachSystemThemeListener()
      applyEffectiveTheme(resolveSystemTheme())
      return
    }
    detachSystemThemeListener()
    applyEffectiveTheme(theme.value)
  }

  function normalizeTextDetector(detector: unknown): TextDetector {
    if (detector === 'ctd' || detector === 'yolo' || detector === 'default') {
      return detector
    }
    return 'default'
  }

  function saveToStorage(): void {
    try {
      const data = JSON.stringify(settings.value)
      localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, data)
    } catch {
      return
    }
  }

  function saveProviderConfigsToStorage(): void {
    try {
      const data = JSON.stringify(providerConfigs.value)
      localStorage.setItem(STORAGE_KEY_PROVIDER_CONFIGS, data)
    } catch {
      return
    }
  }

  function savePluginAgentSettingsToStorage(): void {
    try {
      const defaults = createDefaultSettings()
      const stored = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)
      const parsedSettings = stored ? parseCurrentSettings(JSON.parse(stored)) : null
      const baseSettings = parsedSettings ? deepClone(parsedSettings) : defaults

      baseSettings.pluginAgent = deepClone(settings.value.pluginAgent)
      localStorage.setItem(STORAGE_KEY_TRANSLATION_SETTINGS, JSON.stringify(baseSettings))
    } catch {
      return
    }
  }

  function savePluginAgentProviderConfigsToStorage(): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_PROVIDER_CONFIGS)
      const parsed = stored
        ? parseCurrentProviderConfigs(JSON.parse(stored))
        : null

      const nextProviderConfigs: ProviderConfigsCache = {
        translation: parsed?.translation || {},
        hqTranslation: parsed?.hqTranslation || {},
        pluginAgent: deepClone(providerConfigs.value.pluginAgent),
        aiVisionOcr: parsed?.aiVisionOcr || {}
      }

      localStorage.setItem(STORAGE_KEY_PROVIDER_CONFIGS, JSON.stringify(nextProviderConfigs))
    } catch {
      return
    }
  }

  function loadFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_TRANSLATION_SETTINGS)

      if (data) {
        const parsed = parseCurrentSettings(JSON.parse(data))
        if (!parsed) {
          return
        }
        settings.value = parsed
        settings.value.textDetector = normalizeTextDetector(settings.value.textDetector)
        ensureNumericTypes()

        // 确保 translatePrompt 与当前翻译模式和 JSON 模式同步（4个独立存储字段之一）
        const t = settings.value.translation
        if (t.translationMode === 'single') {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.singleJsonPrompt : t.singleNormalPrompt
        } else {
          settings.value.translatePrompt = t.openaiOptions.request.forceJsonOutput ? t.batchJsonPrompt : t.batchNormalPrompt
        }
      }
    } catch {
      return
    }
  }

  function loadProviderConfigsFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_PROVIDER_CONFIGS)
      if (data) {
        const parsed = parseCurrentProviderConfigs(JSON.parse(data))
        if (!parsed) {
          return
        }
        providerConfigs.value = parsed
      }
    } catch {
      return
    }
  }

  function ensureNumericTypes(): void {
    const defaults = createDefaultSettings()
    const parseNumberOrFallback = (value: unknown, fallback: number): number => {
      if (value === undefined || value === null || value === '') return fallback
      const parsed = Number(value)
      return Number.isNaN(parsed) ? fallback : parsed
    }

    const be = settings.value.boxExpand
    be.ratio = parseNumberOrFallback(be.ratio, defaults.boxExpand.ratio)
    be.top = parseNumberOrFallback(be.top, defaults.boxExpand.top)
    be.bottom = parseNumberOrFallback(be.bottom, defaults.boxExpand.bottom)
    be.left = parseNumberOrFallback(be.left, defaults.boxExpand.left)
    be.right = parseNumberOrFallback(be.right, defaults.boxExpand.right)

    const pm = settings.value.preciseMask
    pm.dilateSize = parseNumberOrFallback(pm.dilateSize, defaults.preciseMask.dilateSize)
    pm.boxExpandRatio = parseNumberOrFallback(pm.boxExpandRatio, defaults.preciseMask.boxExpandRatio)

    if (
      settings.value.saberYoloRefineOverlapThreshold === undefined ||
      settings.value.saberYoloRefineOverlapThreshold === null ||
      isNaN(Number(settings.value.saberYoloRefineOverlapThreshold))
    ) {
      settings.value.saberYoloRefineOverlapThreshold = defaults.saberYoloRefineOverlapThreshold
    } else {
      settings.value.saberYoloRefineOverlapThreshold = Number(settings.value.saberYoloRefineOverlapThreshold)
    }

    settings.value.minTextBlockAreaPercent = Math.max(
      0,
      parseNumberOrFallback(settings.value.minTextBlockAreaPercent, defaults.minTextBlockAreaPercent)
    )

    settings.value.enableAuxYoloDetection = Boolean(settings.value.enableAuxYoloDetection)
    if (
      settings.value.auxYoloConfThreshold === undefined ||
      settings.value.auxYoloConfThreshold === null ||
      isNaN(Number(settings.value.auxYoloConfThreshold))
    ) {
      settings.value.auxYoloConfThreshold = defaults.auxYoloConfThreshold
    } else {
      settings.value.auxYoloConfThreshold = Number(settings.value.auxYoloConfThreshold)
    }
    if (
      settings.value.auxYoloOverlapThreshold === undefined ||
      settings.value.auxYoloOverlapThreshold === null ||
      isNaN(Number(settings.value.auxYoloOverlapThreshold))
    ) {
      settings.value.auxYoloOverlapThreshold = defaults.auxYoloOverlapThreshold
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
    hq.batchSize = parseNumberOrFallback(hq.batchSize, defaults.hqTranslation.batchSize)
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

  const ocrModule = useOcrSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage
  )

  const translationModule = useTranslationSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage
  )

  const detectionModule = useDetectionSettings(settings, saveToStorage)

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

  const proofreadingModule = useProofreadingSettings(settings, saveToStorage)

  const promptsModule = usePromptsSettings(settings, saveToStorage)

  const miscModule = useMiscSettings(settings, saveToStorage)

  function initSettings(): void {
    loadFromStorage()
    loadProviderConfigsFromStorage()
    loadThemeFromStorage()
  }

  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
    saveToStorage()
  }

  function setTheme(nextTheme: ThemePreference): void {
    theme.value = nextTheme
    try {
      localStorage.setItem(STORAGE_KEY_THEME, nextTheme)
    } catch {
      // Theme persistence is best-effort; the active product theme should still update.
    }
    applyThemePreference()
  }

  function toggleTheme(): void {
    if (theme.value === 'light') {
      setTheme('dark')
    } else if (theme.value === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  function loadThemeFromStorage(): void {
    try {
      const storedTheme = localStorage.getItem(STORAGE_KEY_THEME)
      if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') {
        setTheme(storedTheme)
      }
    } catch {
      return
    }
  }

  async function loadFromBackend(): Promise<boolean> {
    try {
      const response = await getUserSettings()

      if (response.success && response.settings) {
        const backendSettings = response.settings
        if (!applyBackendSettings(backendSettings)) {
          return false
        }
        ensureNumericTypes()

        // 编辑侧栏文字设置始终使用当前默认值，不从后端恢复。
        const defaults = createDefaultSettings()
        settings.value.textStyle = { ...defaults.textStyle }

        saveToStorage()
        saveProviderConfigsToStorage()
        return true
      }
      return false
    } catch {
      return false
    }
  }

  function applyBackendSettings(backendSettings: Record<string, unknown>): boolean {
    const schemaVersion = backendSettings.settingsSchemaVersion
    if (schemaVersion !== 3) {
      return false
    }

    const { providerConfigs: nestedProviderConfigs, ...settingsPayload } = backendSettings
    const parsedSettings = parseCurrentSettings(settingsPayload)
    if (!parsedSettings) {
      return false
    }

    let parsedProviderConfigs: ProviderConfigsCache | null = null

    if (nestedProviderConfigs && typeof nestedProviderConfigs === 'object') {
      parsedProviderConfigs = parseCurrentProviderConfigs(nestedProviderConfigs)
      if (!parsedProviderConfigs) {
        return false
      }
    }

    settings.value = parsedSettings

    if (parsedProviderConfigs) {
      providerConfigs.value = parsedProviderConfigs
    }

    return true
  }

  function buildProviderSettingsForBackend(): ProviderConfigsCache {
    return deepClone(providerConfigs.value)
  }

  async function saveToBackend(): Promise<boolean> {
    try {
      translationModule.saveTranslationProviderConfig(settings.value.translation.provider)
      hqTranslationModule.saveHqProviderConfig(settings.value.hqTranslation.provider)
      pluginAgentModule.savePluginAgentProviderConfig(settings.value.pluginAgent.provider)
      ocrModule.saveAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

      const backendSettings: Record<string, unknown> = deepClone(settings.value)
      backendSettings.settingsSchemaVersion = 3
      backendSettings.providerConfigs = buildProviderSettingsForBackend()

      const response = await saveUserSettings(backendSettings)

      if (response.success) {
        return true
      }
      return false
    } catch {
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
        openaiOptions: cloneOpenAiOptions(settings.value.pluginAgent.openaiOptions)
      }

      savePluginAgentSettingsToStorage()
      savePluginAgentProviderConfigsToStorage()

      let backendSettings: Record<string, unknown> = {}

      try {
        const response = await getUserSettings()
        if (response.success && response.settings && typeof response.settings === 'object') {
          backendSettings = deepClone(response.settings)
        }
      } catch {
        // Backend settings are best-effort; saving proceeds with the current plugin Agent state.
      }

      backendSettings.pluginAgent = deepClone(settings.value.pluginAgent)
      const backendProviderConfigs = (
        backendSettings.providerConfigs && typeof backendSettings.providerConfigs === 'object'
          ? deepClone(backendSettings.providerConfigs)
          : {}
      ) as Record<string, unknown>
      backendProviderConfigs.pluginAgent = deepClone(providerConfigs.value.pluginAgent)
      backendSettings.providerConfigs = backendProviderConfigs

      backendSettings.settingsSchemaVersion = 3

      const saveResponse = await saveUserSettings(backendSettings)
      if (saveResponse.success) {
        return true
      }

      return false
    } catch {
      return false
    }
  }

  return {
    settings,
    providerConfigs,
    theme,
    effectiveTheme,

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

    translationProvider: translationModule.translationProvider,
    setTranslationProvider: translationModule.setTranslationProvider,
    updateTranslationService: translationModule.updateTranslationService,
    setTranslatePrompt: translationModule.setTranslatePrompt,
    setTranslatePromptMode: translationModule.setTranslatePromptMode,
    saveTranslationProviderConfig: translationModule.saveTranslationProviderConfig,
    restoreTranslationProviderConfig: translationModule.restoreTranslationProviderConfig,

    setTextDetector: detectionModule.setTextDetector,
    setMinTextBlockAreaPercent: detectionModule.setMinTextBlockAreaPercent,
    setEnableAuxYoloDetection: detectionModule.setEnableAuxYoloDetection,
    setAuxYoloConfThreshold: detectionModule.setAuxYoloConfThreshold,
    setAuxYoloOverlapThreshold: detectionModule.setAuxYoloOverlapThreshold,
    setEnableSaberYoloRefine: detectionModule.setEnableSaberYoloRefine,
    setSaberYoloRefineOverlapThreshold: detectionModule.setSaberYoloRefineOverlapThreshold,
    updateBoxExpand: detectionModule.updateBoxExpand,
    updatePreciseMask: detectionModule.updatePreciseMask,

    hqProvider: hqTranslationModule.hqProvider,
    setHqProvider: hqTranslationModule.setHqProvider,
    updateHqTranslation: hqTranslationModule.updateHqTranslation,
    setHqUseStream: hqTranslationModule.setHqUseStream,
    setHqForceJsonOutput: hqTranslationModule.setHqForceJsonOutput,
    saveHqProviderConfig: hqTranslationModule.saveHqProviderConfig,
    restoreHqProviderConfig: hqTranslationModule.restoreHqProviderConfig,

    pluginAgentProvider: pluginAgentModule.pluginAgentProvider,
    setPluginAgentProvider: pluginAgentModule.setPluginAgentProvider,
    updatePluginAgent: pluginAgentModule.updatePluginAgent,
    savePluginAgentProviderConfig: pluginAgentModule.savePluginAgentProviderConfig,
    restorePluginAgentProviderConfig: pluginAgentModule.restorePluginAgentProviderConfig,

    isProofreadingEnabled: proofreadingModule.isProofreadingEnabled,
    setProofreadingEnabled: proofreadingModule.setProofreadingEnabled,
    addProofreadingRound: proofreadingModule.addProofreadingRound,
    updateProofreadingRound: proofreadingModule.updateProofreadingRound,
    removeProofreadingRound: proofreadingModule.removeProofreadingRound,
    setProofreadingMaxRetries: proofreadingModule.setProofreadingMaxRetries,

    setTextboxPrompt: promptsModule.setTextboxPrompt,
    setUseTextboxPrompt: promptsModule.setUseTextboxPrompt,

    textStyle: miscModule.textStyle,
    updateSettings: miscModule.updateSettings,
    updateTextStyle: miscModule.updateTextStyle,
    setPdfProcessingMethod: miscModule.setPdfProcessingMethod,
    setShowDetectionDebug: miscModule.setShowDetectionDebug,
    setAutoSaveInBookshelfMode: miscModule.setAutoSaveInBookshelfMode,
    setRemoveTextWithOcr: miscModule.setRemoveTextWithOcr,
    setEnableVerboseLogs: miscModule.setEnableVerboseLogs,
    setLamaDisableResize: miscModule.setLamaDisableResize,

    saveToStorage,
    loadFromStorage,
    setTheme,
    toggleTheme,
    loadThemeFromStorage,
    initSettings,
    resetToDefaults,

    loadFromBackend,
    saveToBackend,
    savePluginAgentSettings
  }
})

export type { ProviderConfigsCache } from './types'
