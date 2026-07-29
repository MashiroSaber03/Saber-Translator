import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  TranslationSettings,
  TranslationServiceSettings,
  TranslationProvider
} from '@/types/settings'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  normalizeOpenAiOptions,
  omitOpenAiOptionsPatchFields,
  type OpenAiOptionsPatch,
} from '@/utils/openaiOptions'
import type { ProviderConfigsCache, TranslationProviderConfig } from '../types'
import {
  applyProviderCredentials,
  clearProviderCredentials,
  restoreProviderCacheEntry,
  saveProviderCacheEntry,
  snapshotProviderCredentials,
} from '../providerConfigCache'

export function useTranslationSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
) {
  type TranslationServiceUiUpdates = Partial<TranslationServiceSettings> & OpenAiOptionsPatch
  const translationProvider = computed(() => settings.value.translation.provider)

  function setTranslationProvider(provider: TranslationProvider): void {
    provider = normalizeProviderId(provider) as TranslationProvider
    const previousProvider = settings.value.translation.provider
    if (previousProvider === provider) return

    saveTranslationProviderConfig(previousProvider)

    settings.value.translation.provider = provider

    restoreTranslationProviderConfig(provider)

  }

  function updateTranslationService(updates: TranslationServiceUiUpdates): void {
    Object.assign(settings.value.translation, omitOpenAiOptionsPatchFields(updates))
    applyOpenAiOptionsPatch(settings.value.translation.openaiOptions, updates)
  }

  function setTranslatePrompt(prompt: string): void {
    const translation = settings.value.translation
    const forceJsonOutput = translation.openaiOptions.request.forceJsonOutput
    if (translation.translationMode === 'single') {
      if (forceJsonOutput) {
        translation.singleJsonPrompt = prompt
      } else {
        translation.singleNormalPrompt = prompt
      }
    } else if (forceJsonOutput) {
      translation.batchJsonPrompt = prompt
    } else {
      translation.batchNormalPrompt = prompt
    }
    settings.value.translatePrompt = prompt
  }

  function setTranslatePromptMode(forceJsonOutput: boolean): void {
    settings.value.translation.openaiOptions.request.forceJsonOutput = forceJsonOutput

    // Prompt variants stay separate so toggling modes restores the last edited value.
    const isSingleMode = settings.value.translation.translationMode === 'single'
    let prompt: string
    if (isSingleMode) {
      prompt = forceJsonOutput
        ? settings.value.translation.singleJsonPrompt
        : settings.value.translation.singleNormalPrompt
    } else {
      prompt = forceJsonOutput
        ? settings.value.translation.batchJsonPrompt
        : settings.value.translation.batchNormalPrompt
    }
    settings.value.translatePrompt = prompt

  }

  function saveTranslationProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value.translation,
      buildConfig: (): TranslationProviderConfig => ({
        ...snapshotProviderCredentials(settings.value.translation),
        openaiOptions: cloneOpenAiOptions(settings.value.translation.openaiOptions),
        translationMode: settings.value.translation.translationMode
      }),
    })
  }

  function restoreTranslationProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value.translation,
      applyCached: (cached) => {
        applyProviderCredentials(settings.value.translation, cached)
        if (cached.openaiOptions !== undefined) {
          settings.value.translation.openaiOptions = normalizeOpenAiOptions(
            cached.openaiOptions,
            settings.value.translation.openaiOptions,
          )
        }
        if (cached.translationMode !== undefined) settings.value.translation.translationMode = cached.translationMode
      },
      applyMissing: () => {
        clearProviderCredentials(settings.value.translation)
      },
    })
  }

  return {
    translationProvider,
    setTranslationProvider,
    updateTranslationService,
    setTranslatePrompt,
    setTranslatePromptMode,
    saveTranslationProviderConfig,
    restoreTranslationProviderConfig
  }
}
