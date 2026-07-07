import type { Ref } from 'vue'
import type {
  WebImportProviderConfigs,
  WebImportSettings,
} from '@/types/webImport'
import { DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT } from '@/constants'
import { normalizeProviderId, providerSupportsCapability } from '@/config/aiProviders'
import {
  applyProviderCredentials,
  clearProviderCredentials,
  restoreProviderCacheEntry,
  saveProviderCacheEntry,
  snapshotProviderCredentials,
} from '../providerConfigCache'

export function createDefaultWebImportSettings(): WebImportSettings {
  return {
    firecrawl: {
      apiKey: ''
    },
    agent: {
      provider: 'openai',
      apiKey: '',
      customBaseUrl: '',
      modelName: 'gpt-4o-mini',
      useStream: false,
      forceJsonOutput: true,
      maxRetries: 3,
      timeout: 120
    },
    extraction: {
      prompt: DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
      maxIterations: 10
    },
    download: {
      concurrency: 3,
      timeout: 30,
      retries: 3,
      delay: 100,
      useReferer: true
    },
    imagePreprocess: {
      enabled: false,
      autoRotate: true,
      compression: {
        enabled: false,
        quality: 85,
        maxWidth: 0,
        maxHeight: 0
      },
      formatConvert: {
        enabled: false,
        targetFormat: 'original'
      }
    },
    advanced: {
      customCookie: '',
      customHeaders: '',
      bypassProxy: false
    },
    ui: {
      showAgentLogs: true,
      autoImport: false
    }
  }
}

export function createDefaultWebImportProviderConfigs(): WebImportProviderConfigs {
  return {
    agent: {}
  }
}

export function isWebImportAgentProvider(provider: unknown): provider is WebImportSettings['agent']['provider'] {
  return typeof provider === 'string'
    && provider === normalizeProviderId(provider)
    && providerSupportsCapability(provider, 'webImportAgent')
}

function toWebImportAgentProvider(provider: string): WebImportSettings['agent']['provider'] | null {
  const canonicalProvider = normalizeProviderId(provider)
  return isWebImportAgentProvider(canonicalProvider) ? canonicalProvider : null
}

export function useWebImportSettings(
  webImportSettings: Ref<WebImportSettings>,
  providerConfigs: Ref<WebImportProviderConfigs>
) {
  function setFirecrawlApiKey(apiKey: string): void {
    webImportSettings.value.firecrawl.apiKey = apiKey
  }

  function setAgentProvider(provider: string): void {
    const canonicalProvider = toWebImportAgentProvider(provider)
    if (!canonicalProvider) return
    const previousProvider = webImportSettings.value.agent.provider
    if (previousProvider === canonicalProvider) return

    saveAgentProviderConfig(previousProvider)
    webImportSettings.value.agent.provider = canonicalProvider
    restoreAgentProviderConfig(canonicalProvider)
  }

  function setAgentApiKey(apiKey: string): void {
    webImportSettings.value.agent.apiKey = apiKey
  }

  function setAgentBaseUrl(baseUrl: string): void {
    webImportSettings.value.agent.customBaseUrl = baseUrl
  }

  function setAgentModelName(modelName: string): void {
    webImportSettings.value.agent.modelName = modelName
  }

  function setAgentUseStream(useStream: boolean): void {
    webImportSettings.value.agent.useStream = useStream
  }

  function setAgentForceJsonOutput(forceJsonOutput: boolean): void {
    webImportSettings.value.agent.forceJsonOutput = forceJsonOutput
  }

  function setAgentTimeout(timeout: number): void {
    webImportSettings.value.agent.timeout = timeout
  }

  function setExtractionPrompt(prompt: string): void {
    webImportSettings.value.extraction.prompt = prompt
  }

  function setExtractionMaxIterations(maxIterations: number): void {
    webImportSettings.value.extraction.maxIterations = maxIterations
  }

  function resetExtractionPrompt(): void {
    webImportSettings.value.extraction.prompt = DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT
  }

  function setDownloadConcurrency(concurrency: number): void {
    webImportSettings.value.download.concurrency = concurrency
  }

  function setDownloadTimeout(timeout: number): void {
    webImportSettings.value.download.timeout = timeout
  }

  function setDownloadRetries(retries: number): void {
    webImportSettings.value.download.retries = retries
  }

  function setDownloadDelay(delay: number): void {
    webImportSettings.value.download.delay = delay
  }

  function setDownloadUseReferer(useReferer: boolean): void {
    webImportSettings.value.download.useReferer = useReferer
  }

  function setImagePreprocessEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.enabled = enabled
  }

  function setImageAutoRotate(autoRotate: boolean): void {
    webImportSettings.value.imagePreprocess.autoRotate = autoRotate
  }

  function setImageCompressionEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.compression.enabled = enabled
  }

  function setImageCompressionQuality(quality: number): void {
    webImportSettings.value.imagePreprocess.compression.quality = quality
  }

  function setImageMaxWidth(maxWidth: number): void {
    webImportSettings.value.imagePreprocess.compression.maxWidth = maxWidth
  }

  function setImageMaxHeight(maxHeight: number): void {
    webImportSettings.value.imagePreprocess.compression.maxHeight = maxHeight
  }

  function setImageFormatConvertEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.formatConvert.enabled = enabled
  }

  function setImageTargetFormat(format: 'jpeg' | 'png' | 'webp' | 'original'): void {
    webImportSettings.value.imagePreprocess.formatConvert.targetFormat = format
  }

  function setCustomCookie(cookie: string): void {
    webImportSettings.value.advanced.customCookie = cookie
  }

  function setCustomHeaders(headers: string): void {
    webImportSettings.value.advanced.customHeaders = headers
  }

  function setBypassProxy(bypass: boolean): void {
    webImportSettings.value.advanced.bypassProxy = bypass
  }

  function setShowAgentLogs(show: boolean): void {
    webImportSettings.value.ui.showAgentLogs = show
  }

  function setAutoImport(autoImport: boolean): void {
    webImportSettings.value.ui.autoImport = autoImport
  }

  function saveAgentProviderConfig(provider: string): void {
    saveProviderCacheEntry({
      provider,
      cache: providerConfigs.value.agent,
      buildConfig: () => snapshotProviderCredentials(webImportSettings.value.agent),
      normalizeProvider: toWebImportAgentProvider,
    })
  }

  function restoreAgentProviderConfig(provider: string): void {
    restoreProviderCacheEntry({
      provider,
      cache: providerConfigs.value.agent,
      applyCached: cached => {
        applyProviderCredentials(webImportSettings.value.agent, cached)
      },
      applyMissing: () => {
        clearProviderCredentials(webImportSettings.value.agent)
      },
      normalizeProvider: toWebImportAgentProvider,
    })
  }

  return {
    setFirecrawlApiKey,
    setAgentProvider,
    setAgentApiKey,
    setAgentBaseUrl,
    setAgentModelName,
    setAgentUseStream,
    setAgentForceJsonOutput,
    setAgentTimeout,
    saveAgentProviderConfig,
    restoreAgentProviderConfig,
    setExtractionPrompt,
    setExtractionMaxIterations,
    resetExtractionPrompt,
    setDownloadConcurrency,
    setDownloadTimeout,
    setDownloadRetries,
    setDownloadDelay,
    setDownloadUseReferer,
    setImagePreprocessEnabled,
    setImageAutoRotate,
    setImageCompressionEnabled,
    setImageCompressionQuality,
    setImageMaxWidth,
    setImageMaxHeight,
    setImageFormatConvertEnabled,
    setImageTargetFormat,
    setCustomCookie,
    setCustomHeaders,
    setBypassProxy,
    setShowAgentLogs,
    setAutoImport,
  }
}
