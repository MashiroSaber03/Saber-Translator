import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TranslationSettings } from '@/types/settings'
import { STORAGE_KEY_THEME } from '@/constants'
import {
  getV2Settings,
  saveV2SettingsTransaction,
  type V2CredentialEdit,
  type V2CredentialSummary,
  type V2Font,
  type V2Prompt,
  type V2ProviderSettingMutation,
  type V2SettingsDocument,
  type V2WorkflowPreferences,
  updateV2WorkflowPreferences,
} from '@/api/v2/settings'
import { deepClone } from '@/utils/deepClone'
import { setBackendAccessRestricted } from '@/services/backendAccessGate'
import { normalizeTextStyleSettings } from '@/defaults/textStyleDefaults'

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

const CHAPTER_WORK_STATE_KEYS = [
  'ocrEngine',
  'sourceLanguage',
  'textDetector',
  'minTextBlockAreaPercent',
  'enableAuxYoloDetection',
  'auxYoloConfThreshold',
  'auxYoloOverlapThreshold',
  'enableSaberYoloRefine',
  'saberYoloRefineOverlapThreshold',
  'baiduOcr',
  'paddleOcrVl',
  'aiVisionOcr',
  'hybridOcr',
  'translation',
  'targetLanguage',
  'translatePrompt',
  'useTextboxPrompt',
  'textboxPrompt',
  'hqTranslation',
  'proofreading',
  'boxExpand',
  'preciseMask',
  'showDetectionDebug',
  'parallel',
  'removeTextWithOcr',
  'lamaDisableResize',
] as const satisfies readonly (keyof TranslationSettings)[]
const CHAPTER_WORK_STATE_KEY_SET = new Set<string>(CHAPTER_WORK_STATE_KEYS)

const CHAPTER_SECRET_KEYS = new Set([
  'apikey',
  'secretkey',
  'secret',
  'token',
  'password',
  'credentialversionid',
])

function scrubChapterWorkState(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(scrubChapterWorkState)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !CHAPTER_SECRET_KEYS.has(
        key.toLowerCase().replaceAll(/[^a-z0-9]/g, ''),
      ))
      .map(([key, child]) => [key, scrubChapterWorkState(child)]),
  )
}

function mergeObjects(
  base: Record<string, unknown>,
  override: Record<string, unknown>,
): Record<string, unknown> {
  const result = deepClone(base)
  for (const [key, value] of Object.entries(override)) {
    const current = result[key]
    if (Array.isArray(current) && Array.isArray(value)) {
      result[key] = value.map((item, index) => {
        const baseItem = current[index]
        return (
          baseItem
          && item
          && typeof baseItem === 'object'
          && typeof item === 'object'
          && !Array.isArray(baseItem)
          && !Array.isArray(item)
        )
          ? mergeObjects(
              baseItem as Record<string, unknown>,
              item as Record<string, unknown>,
            )
          : deepClone(item)
      })
      continue
    }
    result[key] = (
      current
      && value
      && typeof current === 'object'
      && typeof value === 'object'
      && !Array.isArray(current)
      && !Array.isArray(value)
    )
      ? mergeObjects(
          current as Record<string, unknown>,
          value as Record<string, unknown>,
        )
      : deepClone(value)
  }
  return result
}

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
  const fontCatalog = ref<V2Font[]>([])
  const promptCatalog = ref<V2Prompt[]>([])
  const workflowPreferences = ref<V2WorkflowPreferences>({
    rememberWorkflowModeEnabled: false,
    lastWorkflowMode: 'translate-current',
  })
  const isBackendReady = ref(false)
  const backendError = ref<string | null>(null)

  let settingsRevision = 0
  let textStyleDefaultsRevision = 0
  let workflowPreferencesRevision = 0
  let providerRevisions = new Map<string, number>()
  let loadPromise: Promise<boolean> | null = null
  let activeChapterWorkState: {
    chapterId: string
    payload: Record<string, unknown>
  } | null = null

  const ocrModule = useOcrSettings(
    settings,
    providerConfigs,
  )

  const translationModule = useTranslationSettings(
    settings,
    providerConfigs,
  )

  const detectionModule = useDetectionSettings(settings)

  const hqTranslationModule = useHqTranslationSettings(
    settings,
    providerConfigs,
  )

  const pluginAgentModule = usePluginAgentSettings(
    settings,
    providerConfigs,
  )

  const proofreadingModule = useProofreadingSettings(settings)
  const promptsModule = usePromptsSettings(settings)
  const miscModule = useMiscSettings(settings)

  function initSettings(): void {
    loadThemeFromStorage()
    setBackendAccessRestricted(
      true,
      '正在读取后端设置',
    )
  }

  function resetToDefaults(): void {
    settings.value = createDefaultSettings()
  }

  function applyBackendDocument(document: V2SettingsDocument): void {
    // Reloading the global document must not roll an active chapter back to the
    // snapshot captured when the translation page first opened. Settings edited
    // in the modal are already the current chapter work state, while the
    // debounced chapter-memory write may still be in flight.
    const currentChapterWorkState = activeChapterWorkState
      ? chapterWorkStatePayload()
      : null
    const translationEntry = document.settings.find(row => row.domain === 'translation')
    const textStyleDefaultsEntry = document.settings.find(
      row => row.domain === 'text_style_defaults',
    )
    const workflowPreferencesEntry = document.settings.find(
      row => row.domain === 'workflow_preferences',
    )
    settingsRevision = translationEntry?.revision ?? 0
    textStyleDefaultsRevision = textStyleDefaultsEntry?.revision ?? 0
    workflowPreferencesRevision = workflowPreferencesEntry?.revision ?? 0
    const parsed = translationEntry
      ? parseCurrentSettings(translationEntry.payload)
      : null
    if (translationEntry && !parsed) {
      throw new Error('后端翻译设置格式无效')
    }
    settings.value = parsed ?? createDefaultSettings()
    if (textStyleDefaultsEntry) {
      settings.value.textStyle = normalizeTextStyleSettings(
        textStyleDefaultsEntry.payload,
      )
    }
    workflowPreferences.value = {
      rememberWorkflowModeEnabled: Boolean(
        workflowPreferencesEntry?.payload.rememberWorkflowModeEnabled,
      ),
      lastWorkflowMode: String(
        workflowPreferencesEntry?.payload.lastWorkflowMode ?? 'translate-current',
      ),
    }
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
    if (activeChapterWorkState && currentChapterWorkState) {
      activeChapterWorkState.payload = deepClone(currentChapterWorkState)
      applyChapterWorkState(currentChapterWorkState)
    }
  }

  function chapterWorkStatePayload(): Record<string, unknown> {
    const source = settings.value as unknown as Record<string, unknown>
    return Object.fromEntries(
      CHAPTER_WORK_STATE_KEYS.map(key => [
        key,
        scrubChapterWorkState(source[key]),
      ]),
    )
  }

  function applyChapterWorkState(payload: Record<string, unknown>): boolean {
    const unknown = Object.keys(payload).filter(
      key => !CHAPTER_WORK_STATE_KEY_SET.has(key),
    )
    if (unknown.length > 0) return false
    const current = settings.value as unknown as Record<string, unknown>
    const candidate = mergeObjects(current, scrubChapterWorkState(payload) as Record<string, unknown>)
    candidate.settingsSchemaVersion = 3
    candidate.textStyle = deepClone(settings.value.textStyle)
    candidate.pluginAgent = deepClone(settings.value.pluginAgent)
    candidate.enableVerboseLogs = settings.value.enableVerboseLogs
    const parsed = parseCurrentSettings(candidate)
    if (!parsed) return false
    settings.value = parsed
    return true
  }

  function hydrateChapterWorkState(
    chapterId: string,
    payload: Record<string, unknown>,
  ): boolean {
    const cloned = deepClone(payload)
    activeChapterWorkState = { chapterId, payload: cloned }
    if (applyChapterWorkState(cloned)) return true
    activeChapterWorkState = null
    return false
  }

  function clearChapterWorkState(chapterId?: string): void {
    if (
      chapterId
      && activeChapterWorkState
      && activeChapterWorkState.chapterId !== chapterId
    ) {
      return
    }
    activeChapterWorkState = null
  }

  function hydrateFromBackendDocument(document: V2SettingsDocument): boolean {
    try {
      applyBackendDocument(document)
      backendError.value = null
      isBackendReady.value = true
      setBackendAccessRestricted(false)
      return true
    } catch (error) {
      isBackendReady.value = false
      backendError.value = error instanceof Error ? error.message : '设置加载失败'
      setBackendAccessRestricted(true, backendError.value)
      return false
    }
  }

  function hydrateResourceCatalogs(fonts: V2Font[], prompts: V2Prompt[]): void {
    fontCatalog.value = deepClone(fonts)
    promptCatalog.value = deepClone(prompts)
  }

  async function loadFromBackend(): Promise<boolean> {
    if (loadPromise) return loadPromise
    loadPromise = (async () => {
      try {
        return hydrateFromBackendDocument(await getV2Settings())
      } catch (error) {
        isBackendReady.value = false
        backendError.value = error instanceof Error ? error.message : '设置加载失败'
        setBackendAccessRestricted(true, backendError.value)
        return false
      }
    })()
    try {
      return await loadPromise
    } finally {
      loadPromise = null
    }
  }

  async function saveWorkflowPreferences(
    preferences: V2WorkflowPreferences,
  ): Promise<boolean> {
    try {
      const updated = await updateV2WorkflowPreferences(
        preferences,
        workflowPreferencesRevision,
      )
      workflowPreferencesRevision = updated.revision
      workflowPreferences.value = deepClone(preferences)
      return true
    } catch (error) {
      backendError.value = error instanceof Error
        ? error.message
        : '工作流偏好保存失败'
      return false
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

  function hasCredential(domain: string, provider: string): boolean {
    return Boolean(currentCredential(domain, provider)?.hasKey)
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
    source: {
      credentials?: V2CredentialSummary[]
      revisions?: Map<string, number>
    } = {},
  ): void {
    const nonEmptySecret = Object.fromEntries(
      Object.entries(secret).filter(([, value]) => value !== '' && value != null),
    )
    const credentials = source.credentials ?? credentialSummaries.value
    const revisions = source.revisions ?? providerRevisions
    const existingCredential = credentials.find(
      row => row.domain === domain && row.provider === provider,
    )
    const clientRef = `credential:${domain}:${provider}`
    const mutation: V2ProviderSettingMutation = {
      domain,
      provider,
      payload: withoutApiKey(rawPayload),
      baseRevision: revisions.get(credentialIdentity(domain, provider)) ?? 0,
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

    const baiduApiKey = settings.value.baiduOcr.apiKey.trim()
    const baiduSecretKey = settings.value.baiduOcr.secretKey.trim()
    if (Boolean(baiduApiKey) !== Boolean(baiduSecretKey)) {
      throw new Error('更换百度 OCR 凭据时必须同时填写 API Key 和 Secret Key')
    }
    addProviderMutation(providerSettings, credentialEdits, {
      domain: 'ocr',
      provider: 'baidu',
      rawPayload: {
        version: settings.value.baiduOcr.version,
        sourceLanguage: settings.value.baiduOcr.sourceLanguage,
      },
      secret: {
        baidu_api_key: baiduApiKey,
        baidu_secret_key: baiduSecretKey,
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
      settings: [
        {
          domain: 'translation',
          payload: sanitizedSettingsPayload(settings.value),
          baseRevision: settingsRevision,
          schemaVersion: 3,
        },
        {
          domain: 'text_style_defaults',
          payload: deepClone(settings.value.textStyle) as unknown as Record<string, unknown>,
          baseRevision: textStyleDefaultsRevision,
          schemaVersion: 1,
        },
      ],
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
    const localDraft = deepClone(settings.value)
    const localProviderDraft = deepClone(providerConfigs.value)
    try {
      const authoritative = await getV2Settings(['translation', 'plugin_agent'])
      const translationEntry = authoritative.settings.find(
        row => row.domain === 'translation',
      )
      const authoritativeSettings = translationEntry
        ? parseCurrentSettings(translationEntry.payload)
        : null
      if (!translationEntry || !authoritativeSettings) {
        throw new Error('后端翻译设置缺失或格式无效')
      }

      pluginAgentModule.savePluginAgentProviderConfig(
        settings.value.pluginAgent.provider,
      )
      authoritativeSettings.pluginAgent = deepClone(settings.value.pluginAgent)
      authoritativeSettings.pluginAgent.apiKey = ''

      const freshRevisions = new Map(
        authoritative.providerSettings.map(row => [
          credentialIdentity(row.domain, row.provider),
          row.revision,
        ]),
      )
      const providerSettings: V2ProviderSettingMutation[] = []
      const credentialEdits: V2CredentialEdit[] = []
      for (const [provider, rawConfig] of Object.entries(
        providerConfigs.value.pluginAgent,
      )) {
        const config = deepClone(rawConfig) as Record<string, unknown>
        addProviderMutation(
          providerSettings,
          credentialEdits,
          {
            domain: 'plugin_agent',
            provider,
            rawPayload: config,
            secret: { api_key: config.apiKey },
          },
          {
            credentials: authoritative.credentials,
            revisions: freshRevisions,
          },
        )
      }

      await saveV2SettingsTransaction({
        settings: [{
          domain: 'translation',
          payload: sanitizedSettingsPayload(authoritativeSettings),
          baseRevision: translationEntry.revision,
          schemaVersion: 3,
        }],
        providerSettings,
        credentialEdits,
      })

      const reloaded = await loadFromBackend()
      if (!reloaded) return false

      const persistedPluginAgent = deepClone(settings.value.pluginAgent)
      const persistedPluginProviders = deepClone(
        providerConfigs.value.pluginAgent,
      )
      settings.value = localDraft
      settings.value.pluginAgent = persistedPluginAgent
      providerConfigs.value = localProviderDraft
      providerConfigs.value.pluginAgent = persistedPluginProviders
      backendError.value = null
      return true
    } catch (error) {
      backendError.value = error instanceof Error
        ? error.message
        : '插件 Agent 设置保存失败'
      return false
    }
  }

  return {
    settings,
    providerConfigs,
    credentialSummaries,
    fontCatalog,
    promptCatalog,
    workflowPreferences,
    hasCredential,
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
    setShowDetectionDebug: miscModule.setShowDetectionDebug,
    setRemoveTextWithOcr: miscModule.setRemoveTextWithOcr,
    setEnableVerboseLogs: miscModule.setEnableVerboseLogs,
    setLamaDisableResize: miscModule.setLamaDisableResize,

    setTheme,
    toggleTheme,
    loadThemeFromStorage,
    initSettings,
    resetToDefaults,
    hydrateFromBackendDocument,
    hydrateResourceCatalogs,
    hydrateChapterWorkState,
    clearChapterWorkState,
    chapterWorkStatePayload,

    loadFromBackend,
    saveToBackend,
    savePluginAgentSettings,
    saveWorkflowPreferences,
  }
})

export type { ProviderConfigsCache } from './types'
