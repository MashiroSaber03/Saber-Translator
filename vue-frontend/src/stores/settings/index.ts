import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'
import {
  STORAGE_KEY_TRANSLATION_SETTINGS,
  STORAGE_KEY_THEME,
  STORAGE_KEY_PROVIDER_CONFIGS,
} from '@/constants'
import { normalizeProviderId } from '@/config/aiProviders'
import { getUserSettings, saveUserSettings } from '@/api/config'
import { cloneOpenAiOptions } from '@/utils/openaiOptions'
import { deepClone } from '@/utils/deepClone'

import type { ProviderConfigsCache } from './types'
import { createDefaultSettings } from './defaults'
import { normalizeSettings } from './normalizeSettings'
import { parseCurrentProviderConfigs, parseCurrentSettings } from './schema'
import { useThemePreference } from './useThemePreference'
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

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<TranslationSettings>(createDefaultSettings())
  const themePreference = useThemePreference(STORAGE_KEY_THEME)
  const { theme, effectiveTheme, setTheme, toggleTheme, loadThemeFromStorage } = themePreference
  const providerConfigs = ref<ProviderConfigsCache>({
    translation: {},
    hqTranslation: {},
    pluginAgent: {},
    aiVisionOcr: {}
  })

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
        normalizeSettings(settings.value)

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

  async function loadFromBackend(): Promise<boolean> {
    try {
      const response = await getUserSettings()

      if (response.success && response.settings) {
        const backendSettings = response.settings
        if (!applyBackendSettings(backendSettings)) {
          return false
        }
        normalizeSettings(settings.value)

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
