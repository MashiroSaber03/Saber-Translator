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

function useAgentProviderSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  section: 'pluginAgent' | 'browserDomAgent',
) {
  type AgentUiUpdates = Partial<Omit<PluginAgentSettings, 'provider'>>
    & OpenAiOptionsPatch

  const provider = computed(() => settings.value[section].provider)
  const getDefaultOpenAiOptions = () => cloneOpenAiOptions(
    createDefaultSettings()[section].openaiOptions,
  )

  function setProvider(provider: PluginAgentProvider): void {
    provider = normalizeProviderId(provider) as PluginAgentProvider
    const previousProvider = settings.value[section].provider
    if (previousProvider === provider) return

    saveProviderConfig(previousProvider)
    settings.value[section].provider = provider
    restoreProviderConfig(provider)
  }

  function updateAgent(updates: AgentUiUpdates): void {
    if (updates.apiKey !== undefined) settings.value[section].apiKey = updates.apiKey
    if (updates.modelName !== undefined) settings.value[section].modelName = updates.modelName
    if (updates.customBaseUrl !== undefined) settings.value[section].customBaseUrl = updates.customBaseUrl
    if (updates.openaiOptions !== undefined) {
      settings.value[section].openaiOptions = cloneOpenAiOptions(updates.openaiOptions)
    }
    applyOpenAiOptionsPatch(settings.value[section].openaiOptions, updates)
  }

  function saveProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value[section],
      buildConfig: (): PluginAgentProviderConfig => ({
        ...snapshotProviderCredentials(settings.value[section]),
        openaiOptions: cloneOpenAiOptions(settings.value[section].openaiOptions),
      }),
    })
  }

  function restoreProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value[section],
      applyCached: (cached) => {
        applyProviderCredentials(settings.value[section], cached)
        settings.value[section].openaiOptions = cached.openaiOptions !== undefined
          ? cloneOpenAiOptions(cached.openaiOptions)
          : getDefaultOpenAiOptions()
      },
      applyMissing: () => {
        clearProviderCredentials(settings.value[section])
        settings.value[section].openaiOptions = getDefaultOpenAiOptions()
      },
    })
  }

  return {
    provider,
    setProvider,
    updateAgent,
    saveProviderConfig,
    restoreProviderConfig,
  }
}

export function usePluginAgentSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
) {
  const agent = useAgentProviderSettings(
    settings,
    providerConfigs,
    'pluginAgent',
  )
  return {
    pluginAgentProvider: agent.provider,
    setPluginAgentProvider: agent.setProvider,
    updatePluginAgent: agent.updateAgent,
    savePluginAgentProviderConfig: agent.saveProviderConfig,
    restorePluginAgentProviderConfig: agent.restoreProviderConfig,
  }
}

export function useBrowserDomAgentSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
) {
  const agent = useAgentProviderSettings(
    settings,
    providerConfigs,
    'browserDomAgent',
  )
  return {
    browserDomAgentProvider: agent.provider,
    setBrowserDomAgentProvider: agent.setProvider,
    updateBrowserDomAgent: agent.updateAgent,
    saveBrowserDomAgentProviderConfig: agent.saveProviderConfig,
    restoreBrowserDomAgentProviderConfig: agent.restoreProviderConfig,
  }
}
