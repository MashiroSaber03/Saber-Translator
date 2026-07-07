import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import { createDefaultSettings } from '../defaults'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  type OpenAiOptionsPatch,
} from '@/utils/openaiOptions'
import type {
  PluginAgentProvider,
  PluginAgentSettings,
  TranslationSettings,
} from '@/types/settings'
import type { PluginAgentProviderConfig, ProviderConfigsCache } from '../types'
import {
  applyProviderCredentials,
  clearProviderCredentials,
  restoreProviderCacheEntry,
  saveProviderCacheEntry,
  snapshotProviderCredentials,
} from '../providerConfigCache'

export function usePluginAgentSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void,
) {
  type PluginAgentUiUpdates = Partial<PluginAgentSettings> & OpenAiOptionsPatch

  const pluginAgentProvider = computed(() => settings.value.pluginAgent.provider)
  const getDefaultOpenAiOptions = () => cloneOpenAiOptions(createDefaultSettings().pluginAgent.openaiOptions)
  const getUncachedProviderOpenAiOptions = () => {
    const options = getDefaultOpenAiOptions()
    options.execution.rpmLimit = 7
    return options
  }

  function setPluginAgentProvider(provider: PluginAgentProvider): void {
    provider = normalizeProviderId(provider) as PluginAgentProvider
    const previousProvider = settings.value.pluginAgent.provider
    if (previousProvider === provider) return

    savePluginAgentProviderConfig(previousProvider)
    settings.value.pluginAgent.provider = provider
    restorePluginAgentProviderConfig(provider)
    saveToStorage()
  }

  function updatePluginAgent(updates: PluginAgentUiUpdates): void {
    if (updates.provider !== undefined) settings.value.pluginAgent.provider = updates.provider
    if (updates.apiKey !== undefined) settings.value.pluginAgent.apiKey = updates.apiKey
    if (updates.modelName !== undefined) settings.value.pluginAgent.modelName = updates.modelName
    if (updates.customBaseUrl !== undefined) settings.value.pluginAgent.customBaseUrl = updates.customBaseUrl
    if (updates.openaiOptions !== undefined) {
      settings.value.pluginAgent.openaiOptions = cloneOpenAiOptions(updates.openaiOptions)
    }
    applyOpenAiOptionsPatch(settings.value.pluginAgent.openaiOptions, updates)
    saveToStorage()
  }

  function savePluginAgentProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value.pluginAgent,
      buildConfig: (): PluginAgentProviderConfig => ({
        ...snapshotProviderCredentials(settings.value.pluginAgent),
        openaiOptions: cloneOpenAiOptions(settings.value.pluginAgent.openaiOptions),
      }),
      persist: saveProviderConfigsToStorage,
    })
  }

  function restorePluginAgentProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value.pluginAgent,
      applyCached: (cached) => {
        applyProviderCredentials(settings.value.pluginAgent, cached)
        settings.value.pluginAgent.openaiOptions = cached.openaiOptions !== undefined
          ? cloneOpenAiOptions(cached.openaiOptions)
          : getDefaultOpenAiOptions()
      },
      applyMissing: () => {
        clearProviderCredentials(settings.value.pluginAgent)
        settings.value.pluginAgent.openaiOptions = getUncachedProviderOpenAiOptions()
      },
    })
  }

  return {
    pluginAgentProvider,
    setPluginAgentProvider,
    updatePluginAgent,
    savePluginAgentProviderConfig,
    restorePluginAgentProviderConfig,
  }
}
