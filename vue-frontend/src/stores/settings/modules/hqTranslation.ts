import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  TranslationSettings,
  HqTranslationSettings,
  HqTranslationProvider
} from '@/types/settings'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  omitOpenAiOptionsPatchFields,
  type OpenAiOptionsPatch,
} from '@/utils/openaiOptions'
import type { ProviderConfigsCache, HqTranslationProviderConfig } from '../types'
import {
  applyProviderCredentials,
  clearProviderCredentials,
  restoreProviderCacheEntry,
  saveProviderCacheEntry,
  snapshotProviderCredentials,
} from '../providerConfigCache'

export function useHqTranslationSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void
) {
  type HqTranslationUiUpdates = Partial<HqTranslationSettings> & OpenAiOptionsPatch
  const hqProvider = computed(() => settings.value.hqTranslation.provider)

  function setHqProvider(provider: HqTranslationProvider): void {
    provider = normalizeProviderId(provider) as HqTranslationProvider
    const previousProvider = settings.value.hqTranslation.provider
    if (previousProvider === provider) return

    saveHqProviderConfig(previousProvider)

    settings.value.hqTranslation.provider = provider

    restoreHqProviderConfig(provider)

    saveToStorage()
  }

  function updateHqTranslation(updates: HqTranslationUiUpdates): void {
    Object.assign(settings.value.hqTranslation, omitOpenAiOptionsPatchFields(updates))
    applyOpenAiOptionsPatch(settings.value.hqTranslation.openaiOptions, updates)
    saveToStorage()
  }

  function setHqUseStream(useStream: boolean): void {
    settings.value.hqTranslation.openaiOptions.execution.useStream = useStream
    saveToStorage()
  }

  function setHqForceJsonOutput(forceJsonOutput: boolean): void {
    settings.value.hqTranslation.openaiOptions.request.forceJsonOutput = forceJsonOutput
    saveToStorage()
  }

  function saveHqProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value.hqTranslation,
      buildConfig: (): HqTranslationProviderConfig => ({
        ...snapshotProviderCredentials(settings.value.hqTranslation),
        batchSize: settings.value.hqTranslation.batchSize,
        openaiOptions: cloneOpenAiOptions(settings.value.hqTranslation.openaiOptions),
        prompt: settings.value.hqTranslation.prompt
      }),
      persist: saveProviderConfigsToStorage,
    })
  }

  function restoreHqProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value.hqTranslation,
      applyCached: (cached) => {
        applyProviderCredentials(settings.value.hqTranslation, cached)
        if (cached.batchSize !== undefined) settings.value.hqTranslation.batchSize = cached.batchSize
        if (cached.openaiOptions !== undefined) settings.value.hqTranslation.openaiOptions = cloneOpenAiOptions(cached.openaiOptions)
        if (cached.prompt !== undefined) settings.value.hqTranslation.prompt = cached.prompt
      },
      applyMissing: () => {
        clearProviderCredentials(settings.value.hqTranslation)
      },
    })
  }

  return {
    hqProvider,
    setHqProvider,
    updateHqTranslation,
    setHqUseStream,
    setHqForceJsonOutput,
    saveHqProviderConfig,
    restoreHqProviderConfig
  }
}
