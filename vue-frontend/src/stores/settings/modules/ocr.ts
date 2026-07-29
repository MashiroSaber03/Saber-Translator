import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  TranslationSettings,
  BaiduOcrSettings,
  PaddleOcrVlSettings,
  AiVisionOcrSettings,
  HybridOcrSettings,
  OcrEngine
} from '@/types/settings'
import type { ProviderConfigsCache, AiVisionOcrProviderConfig } from '../types'
import {
  DEFAULT_AI_VISION_OCR_PROMPT,
  DEFAULT_AI_VISION_OCR_JSON_PROMPT
} from '@/constants'
import { normalizeHybridOcrConfig } from '@/utils/hybridOcr'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  normalizeOpenAiOptions,
  omitOpenAiOptionsPatchFields,
  type OpenAiOptionsPatch,
} from '@/utils/openaiOptions'
import {
  applyProviderCredentials,
  clearProviderCredentials,
  restoreProviderCacheEntry,
  saveProviderCacheEntry,
  snapshotProviderCredentials,
} from '../providerConfigCache'

export function useOcrSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
) {
  type AiVisionOcrUiUpdates = Partial<AiVisionOcrSettings> & OpenAiOptionsPatch
  const ocrEngine = computed(() => settings.value.ocrEngine)
  const sourceLanguage = computed(() => settings.value.sourceLanguage)

  function setOcrEngine(engine: OcrEngine): void {
    const normalized = normalizeHybridOcrConfig(engine, settings.value.hybridOcr)
    settings.value.ocrEngine = normalized.primaryEngine
    settings.value.hybridOcr = normalized.hybrid
  }

  function setSourceLanguage(language: string): void {
    settings.value.sourceLanguage = language
  }

  function updateBaiduOcr(updates: Partial<BaiduOcrSettings>): void {
    Object.assign(settings.value.baiduOcr, updates)
  }

  function updatePaddleOcrVl(updates: Partial<PaddleOcrVlSettings>): void {
    Object.assign(settings.value.paddleOcrVl, updates)
  }

  function updateAiVisionOcr(updates: AiVisionOcrUiUpdates): void {
    Object.assign(settings.value.aiVisionOcr, omitOpenAiOptionsPatchFields(updates))
    applyOpenAiOptionsPatch(settings.value.aiVisionOcr.openaiOptions, updates)
    if (updates.forceJsonOutput !== undefined) {
      settings.value.aiVisionOcr.promptMode = updates.forceJsonOutput ? 'json' : 'normal'
    }
  }

  function updateHybridOcr(updates: Partial<HybridOcrSettings>): void {
    const enablingHybrid = Boolean(updates.enabled) && !settings.value.hybridOcr.enabled
    const normalized = normalizeHybridOcrConfig(
      settings.value.ocrEngine,
      {
        ...settings.value.hybridOcr,
        ...updates
      },
      {
        preferRecommendedOrder: enablingHybrid
      }
    )
    settings.value.ocrEngine = normalized.primaryEngine
    settings.value.hybridOcr = normalized.hybrid
  }

  function setAiVisionOcrProvider(provider: string): void {
    provider = normalizeProviderId(provider)
    const previousProvider = settings.value.aiVisionOcr.provider
    if (previousProvider === provider) return

    saveAiVisionOcrProviderConfig(previousProvider)

    settings.value.aiVisionOcr.provider = provider

    restoreAiVisionOcrProviderConfig(provider)

  }

  function setAiVisionOcrPromptMode(mode: boolean | 'normal' | 'json' | 'paddleocr_vl'): void {
    const normalizedMode = typeof mode === 'boolean' ? (mode ? 'json' : 'normal') : mode

    settings.value.aiVisionOcr.promptMode = normalizedMode
    settings.value.aiVisionOcr.openaiOptions.request.forceJsonOutput = normalizedMode === 'json'

    if (normalizedMode === 'json') {
      settings.value.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_JSON_PROMPT
    } else if (normalizedMode === 'normal') {
      settings.value.aiVisionOcr.prompt = DEFAULT_AI_VISION_OCR_PROMPT
    }

  }

  function saveAiVisionOcrProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value.aiVisionOcr,
      buildConfig: (): AiVisionOcrProviderConfig => ({
        ...snapshotProviderCredentials(settings.value.aiVisionOcr),
        prompt: settings.value.aiVisionOcr.prompt,
        promptMode: settings.value.aiVisionOcr.promptMode,
        openaiOptions: cloneOpenAiOptions(settings.value.aiVisionOcr.openaiOptions),
        minImageSize: settings.value.aiVisionOcr.minImageSize
      }),
    })
  }

  function restoreAiVisionOcrProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value.aiVisionOcr,
      applyCached: (cached) => {
        applyProviderCredentials(settings.value.aiVisionOcr, cached)
        if (cached.prompt !== undefined) settings.value.aiVisionOcr.prompt = cached.prompt
        if (cached.promptMode !== undefined) settings.value.aiVisionOcr.promptMode = cached.promptMode
        if (cached.openaiOptions !== undefined) {
          settings.value.aiVisionOcr.openaiOptions = normalizeOpenAiOptions(
            cached.openaiOptions,
            settings.value.aiVisionOcr.openaiOptions,
          )
        }
        if (cached.minImageSize !== undefined) settings.value.aiVisionOcr.minImageSize = cached.minImageSize
      },
      applyMissing: () => {
        clearProviderCredentials(settings.value.aiVisionOcr)
      },
    })
  }

  return {
    ocrEngine,
    sourceLanguage,
    setOcrEngine,
    setSourceLanguage,
    updateBaiduOcr,
    updatePaddleOcrVl,
    updateAiVisionOcr,
    updateHybridOcr,
    setAiVisionOcrProvider,
    setAiVisionOcrPromptMode,
    saveAiVisionOcrProviderConfig,
    restoreAiVisionOcrProviderConfig
  }
}
