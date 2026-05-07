/**
 * 插件 Agent 设置模块
 */

import { computed, type Ref } from 'vue'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  PluginAgentProvider,
  PluginAgentSettings,
  TranslationSettings,
} from '@/types/settings'
import type { PluginAgentProviderConfig, ProviderConfigsCache } from '../types'

export function usePluginAgentSettings(
  settings: Ref<TranslationSettings>,
  providerConfigs: Ref<ProviderConfigsCache>,
  saveToStorage: () => void,
  saveProviderConfigsToStorage: () => void,
) {
  type PluginAgentUiUpdates = Partial<PluginAgentSettings> & {
    rpmLimit?: number
    transportRetries?: number
    businessRetries?: number
    forceJsonOutput?: boolean
    useStream?: boolean
    extraBody?: Record<string, unknown>
  }

  const pluginAgentProvider = computed(() => settings.value.pluginAgent.provider)

  function setPluginAgentProvider(provider: PluginAgentProvider): void {
    provider = normalizeProviderId(provider) as PluginAgentProvider
    const oldProvider = settings.value.pluginAgent.provider
    if (oldProvider === provider) return

    savePluginAgentProviderConfig(oldProvider)
    settings.value.pluginAgent.provider = provider
    restorePluginAgentProviderConfig(provider)
    saveToStorage()
  }

  function updatePluginAgent(updates: PluginAgentUiUpdates): void {
    Object.assign(settings.value.pluginAgent, updates)
    if (updates.rpmLimit !== undefined) settings.value.pluginAgent.openaiOptions.execution.rpmLimit = updates.rpmLimit
    if (updates.transportRetries !== undefined) settings.value.pluginAgent.openaiOptions.execution.transportRetries = updates.transportRetries
    if (updates.businessRetries !== undefined) settings.value.pluginAgent.openaiOptions.execution.businessRetries = updates.businessRetries
    if (updates.forceJsonOutput !== undefined) settings.value.pluginAgent.openaiOptions.request.forceJsonOutput = updates.forceJsonOutput
    if (updates.useStream !== undefined) settings.value.pluginAgent.openaiOptions.execution.useStream = updates.useStream
    if (Object.prototype.hasOwnProperty.call(updates, 'extraBody')) {
      settings.value.pluginAgent.openaiOptions.request.extraBody = updates.extraBody
    }
    saveToStorage()
  }

  function savePluginAgentProviderConfig(provider: string): void {
    if (!provider) return
    provider = normalizeProviderId(provider)

    const config: PluginAgentProviderConfig = {
      apiKey: settings.value.pluginAgent.apiKey,
      modelName: settings.value.pluginAgent.modelName,
      customBaseUrl: settings.value.pluginAgent.customBaseUrl,
      openaiOptions: JSON.parse(JSON.stringify(settings.value.pluginAgent.openaiOptions)),
    }

    providerConfigs.value.pluginAgent[provider] = config
    saveProviderConfigsToStorage()
  }

  function restorePluginAgentProviderConfig(provider: string): void {
    if (!provider) return
    provider = normalizeProviderId(provider)

    const cached = providerConfigs.value.pluginAgent[provider]
    if (cached) {
      if (cached.apiKey !== undefined) settings.value.pluginAgent.apiKey = cached.apiKey
      if (cached.modelName !== undefined) settings.value.pluginAgent.modelName = cached.modelName
      if (cached.customBaseUrl !== undefined) settings.value.pluginAgent.customBaseUrl = cached.customBaseUrl
      if (cached.openaiOptions !== undefined) settings.value.pluginAgent.openaiOptions = JSON.parse(JSON.stringify(cached.openaiOptions))
    } else {
      settings.value.pluginAgent.apiKey = ''
      settings.value.pluginAgent.modelName = ''
      settings.value.pluginAgent.customBaseUrl = ''
    }
  }

  return {
    pluginAgentProvider,
    setPluginAgentProvider,
    updatePluginAgent,
    savePluginAgentProviderConfig,
    restorePluginAgentProviderConfig,
  }
}
