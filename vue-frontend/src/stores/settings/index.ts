import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'
import {
  STORAGE_KEY_PROVIDER_CONFIGS,
  STORAGE_KEY_THEME,
  STORAGE_KEY_TRANSLATION_SETTINGS,
} from '@/constants'
import {
  getV2Settings,
  saveV2SettingsTransaction,
  type V2CredentialEdit,
  type V2CredentialSummary,
  type V2ProviderSettingMutation,
  type V2SettingsDocument,
} from '@/api/v2/settings'
import { deepClone } from '@/utils/deepClone'

import type { ProviderConfigsCache } from './types'
import { createDefaultSettings } from './defaults'
import { normalizeSettings } from './normalizeSettings'
import { parseCurrentSettings } from './schema'
import { useThemePreference } from './useThemePreference'
import {
  useOcrSettings,
  useTranslationSettings,
  useDetectionSettings,
  useHqTranslationSettings,
  usePluginAgentSettings,
  useProofreadingSettings,
  usePromptsSettings,
  useMiscSettings,
} from './modules'

type ProviderCacheDomain = keyof ProviderConfigsCache

const PROVIDER_DOMAIN_BY_CACHE: Record<ProviderCacheDomain, string> = {
  translation: 'translation',
  hqTranslation: 'hq',
  pluginAgent: 'plugin_agent',
  aiVisionOcr: 'ai_vision_ocr',
}

const CACHE_BY_PROVIDER_DOMAIN = Object.fromEntries(
  Object.entries(PROVIDER_DOMAIN_BY_CACHE).map(([cache, domain]) => [domain, cache]),
) as Record<string, ProviderCacheDomain>

function emptyProviderConfigs(): ProviderConfigsCache {
  return {
    translation: {},
    hqTranslation: {},
    pluginAgent: {},
    aiVisionOcr: {},
  }
}

function withoutApiKey<T extends Record<string, unknown>>(value: T): Omit<T, 'apiKey'> {
  const { apiKey: _apiKey, ...payload } = value
  return payload
}

function sanitizedSettingsPayload(value: TranslationSettings): Record<string, unknown> {
  const payload = deepClone(value) as TranslationSettings
  payload.translation.apiKey = ''
  payload.hqTranslation.apiKey = ''
  payload.pluginAgent.apiKey = ''
  payload.aiVisionOcr.apiKey = ''
  payload.baiduOcr.apiKey = ''
  payload.baiduOcr.secretKey = ''
  payload.proofreading.rounds.forEach((round) => {
    round.apiKey = ''
  })
  payload.settingsSchemaVersion = 3
  return payload as unknown as Record<string, unknown>
}

function credentialIdentity(domain: string, provider: string): string {
  return `${domain}\u0000${provider}`
}

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<TranslationSettings>(createDefaultSettings())
  const themePreference = useThemePreference(STORAGE_KEY_THEME)
  const { theme, effectiveTheme, setTheme, toggleTheme, loadThemeFromStorage } = themePreference
  const providerConfigs = ref<ProviderConfigsCache>(emptyProviderConfigs())
  const credentialSummaries = ref<V2CredentialSummary[]>([])
  const isBackendReady = ref(false)
  const backendError = ref<string | null>(null)

  let settingsRevision = 0
  let providerRevisions = new Map<string, number>()
  let loadPromise: Promise<boolean> | null = null

  // Business settings and credentials are backend facts. These compatibility
  // functions intentionally do not persist browser copies.
  function saveToStorage(): void {}
  function saveProviderConfigsToStorage(): void {}

  function clearRetiredBusinessStorage(): void {
    try {
      localStorage.removeItem(STORAGE_KEY_TRANSLATION_SETTINGS)
      localStorage.removeItem(STORAGE_KEY_PROVIDER_CONFIGS)
    } catch {
      // Storage may be unavailable; backend loading remains authoritative.
    }
  }

  function loadFromStorage(): void {
    clearRetiredBusinessStorage()
  }

  const ocrModule = useOcrSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage,
  )

  const translationModule = useTranslationSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage,
  )

  const detectionModule = useDetectionSettings(settings, saveToStorage)

  const hqTranslationModule = useHqTranslationSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage,
  )

  const pluginAgentModule = usePluginAgentSettings(
    settings,
    providerConfigs,
    saveToStorage,
    saveProviderConfigsToStorage,
  )

  const proofreadingModule = useProofreadingSettings(settings, saveToStorage)
  const promptsModule = usePromptsSettings(settings, saveToStorage)
  const miscModule = useMiscSettings(settings, saveToStorage)

  function initSettings(): void {
    clearRetiredBusinessStorage()
    loadThemeFromStorage()
    void loadFromBackend()
  }

  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
  }

  function applyBackendDocument(document: V2SettingsDocument): void {
    const translationEntry = document.settings.find(row => row.domain === 'translation')
    settingsRevision = translationEntry?.revision ?? 0
    const parsed = translationEntry
      ? parseCurrentSettings(translationEntry.payload)
      : null
    settings.value = parsed ?? createDefaultSettings()
    normalizeSettings(settings.value)

    providerConfigs.value = emptyProviderConfigs()
    providerRevisions = new Map()
    for (const row of document.providerSettings) {
      const cacheDomain = CACHE_BY_PROVIDER_DOMAIN[row.domain]
      if (cacheDomain) {
        providerConfigs.value[cacheDomain][row.provider] = {
          ...deepClone(row.payload),
          apiKey: '',
        }
      }
      providerRevisions.set(credentialIdentity(row.domain, row.provider), row.revision)
    }
    credentialSummaries.value = document.credentials

    translationModule.restoreTranslationProviderConfig(settings.value.translation.provider)
    hqTranslationModule.restoreHqProviderConfig(settings.value.hqTranslation.provider)
    pluginAgentModule.restorePluginAgentProviderConfig(settings.value.pluginAgent.provider)
    ocrModule.restoreAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

    // A credential summary deliberately never hydrates a secret into the form.
    settings.value.translation.apiKey = ''
    settings.value.hqTranslation.apiKey = ''
    settings.value.pluginAgent.apiKey = ''
    settings.value.aiVisionOcr.apiKey = ''
    settings.value.baiduOcr.apiKey = ''
    settings.value.baiduOcr.secretKey = ''
    settings.value.proofreading.rounds.forEach((round) => {
      round.apiKey = ''
    })
  }

  async function loadFromBackend(): Promise<boolean> {
    if (loadPromise) return loadPromise
    loadPromise = (async () => {
      try {
        applyBackendDocument(await getV2Settings())
        backendError.value = null
        isBackendReady.value = true
        return true
      } catch (error) {
        isBackendReady.value = false
        backendError.value = error instanceof Error ? error.message : '设置加载失败'
        return false
      }
    })()
    try {
      return await loadPromise
    } finally {
      loadPromise = null
    }
  }

  function currentCredential(
    domain: string,
    provider: string,
  ): V2CredentialSummary | undefined {
    return credentialSummaries.value.find(
      row => row.domain === domain && row.provider === provider,
    )
  }

  function addProviderMutation(
    providerSettings: V2ProviderSettingMutation[],
    credentialEdits: V2CredentialEdit[],
    {
      domain,
      provider,
      rawPayload,
      secret,
    }: {
      domain: string
      provider: string
      rawPayload: Record<string, unknown>
      secret: Record<string, unknown>
    },
  ): void {
    const nonEmptySecret = Object.fromEntries(
      Object.entries(secret).filter(([, value]) => value !== '' && value != null),
    )
    const existingCredential = currentCredential(domain, provider)
    const clientRef = `credential:${domain}:${provider}`
    const mutation: V2ProviderSettingMutation = {
      domain,
      provider,
      payload: withoutApiKey(rawPayload),
      baseRevision: providerRevisions.get(credentialIdentity(domain, provider)) ?? 0,
      schemaVersion: 1,
    }
    if (Object.keys(nonEmptySecret).length > 0) {
      credentialEdits.push({
        domain,
        provider,
        secret: nonEmptySecret,
        baseRevision: existingCredential?.revision ?? 0,
        credentialId: existingCredential?.credentialId,
        clientRef,
      })
      mutation.credentialEditRef = clientRef
    } else if (existingCredential) {
      mutation.credentialVersionId = existingCredential.credentialVersionId
    }
    providerSettings.push(mutation)
  }

  function buildSettingsTransaction() {
    translationModule.saveTranslationProviderConfig(settings.value.translation.provider)
    hqTranslationModule.saveHqProviderConfig(settings.value.hqTranslation.provider)
    pluginAgentModule.savePluginAgentProviderConfig(settings.value.pluginAgent.provider)
    ocrModule.saveAiVisionOcrProviderConfig(settings.value.aiVisionOcr.provider)

    const providerSettings: V2ProviderSettingMutation[] = []
    const credentialEdits: V2CredentialEdit[] = []
    for (const [cacheDomain, domain] of Object.entries(PROVIDER_DOMAIN_BY_CACHE) as Array<
      [ProviderCacheDomain, string]
    >) {
      for (const [provider, rawConfig] of Object.entries(providerConfigs.value[cacheDomain])) {
        const config = deepClone(rawConfig) as Record<string, unknown>
        addProviderMutation(providerSettings, credentialEdits, {
          domain,
          provider,
          rawPayload: config,
          secret: {
            [domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key']:
              config.apiKey,
          },
        })
      }
    }

    addProviderMutation(providerSettings, credentialEdits, {
      domain: 'ocr',
      provider: 'baidu',
      rawPayload: {
        version: settings.value.baiduOcr.version,
        sourceLanguage: settings.value.baiduOcr.sourceLanguage,
      },
      secret: {
        baidu_api_key: settings.value.baiduOcr.apiKey,
        baidu_secret_key: settings.value.baiduOcr.secretKey,
      },
    })

    settings.value.proofreading.rounds.forEach((round, index) => {
      const config = deepClone(round) as unknown as Record<string, unknown>
      addProviderMutation(providerSettings, credentialEdits, {
        domain: `proofreading_${index}`,
        provider: round.provider,
        rawPayload: config,
        secret: { api_key: round.apiKey },
      })
    })

    return {
      settings: [{
        domain: 'translation',
        payload: sanitizedSettingsPayload(settings.value),
        baseRevision: settingsRevision,
        schemaVersion: 3,
      }],
      providerSettings,
      credentialEdits,
    }
  }

  async function saveToBackend(): Promise<boolean> {
    if (!isBackendReady.value) {
      backendError.value = '后端设置尚未加载，已阻止覆盖保存'
      return false
    }
    try {
      await saveV2SettingsTransaction(buildSettingsTransaction())
      return await loadFromBackend()
    } catch (error) {
      backendError.value = error instanceof Error ? error.message : '设置保存失败'
      return false
    }
  }

  async function savePluginAgentSettings(): Promise<boolean> {
    return saveToBackend()
  }

  return {
    settings,
    providerConfigs,
    credentialSummaries,
    isBackendReady,
    backendError,
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
    savePluginAgentSettings,
  }
})

export type { ProviderConfigsCache } from './types'
