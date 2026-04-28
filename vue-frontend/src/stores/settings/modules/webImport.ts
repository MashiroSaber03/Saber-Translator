/**
 * 网页导入设置模块
 */

import type { Ref } from 'vue'
import type {
  WebImportAgentProviderConfig,
  WebImportProviderConfigs,
  WebImportSettings,
} from '@/types/webImport'
import { DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT } from '@/constants'
import { normalizeProviderId } from '@/config/aiProviders'

// ============================================================
// 默认值
// ============================================================

/** 创建默认网页导入设置 */
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

function createEmptyAgentProviderConfig(): WebImportAgentProviderConfig {
  return {
    apiKey: '',
    modelName: '',
    customBaseUrl: ''
  }
}

// ============================================================
// Composable
// ============================================================

export function useWebImportSettings(
  webImportSettings: Ref<WebImportSettings>,
  providerConfigs: Ref<WebImportProviderConfigs>
) {
  // ============================================================
  // Firecrawl 设置
  // ============================================================

  function setFirecrawlApiKey(apiKey: string): void {
    webImportSettings.value.firecrawl.apiKey = apiKey
  }

  // ============================================================
  // Agent 设置
  // ============================================================

  function setAgentProvider(provider: string): void {
    const canonicalProvider = normalizeProviderId(provider) as WebImportSettings['agent']['provider']
    const oldProvider = normalizeProviderId(webImportSettings.value.agent.provider) as WebImportSettings['agent']['provider']
    if (oldProvider === canonicalProvider) return

    saveAgentProviderConfig(oldProvider)
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

  function setAgentForceJson(forceJson: boolean): void {
    webImportSettings.value.agent.forceJsonOutput = forceJson
  }

  function setAgentTimeout(timeout: number): void {
    webImportSettings.value.agent.timeout = timeout
  }

  // ============================================================
  // 提取设置
  // ============================================================

  function setExtractionPrompt(prompt: string): void {
    webImportSettings.value.extraction.prompt = prompt
  }

  function setExtractionMaxIterations(maxIterations: number): void {
    webImportSettings.value.extraction.maxIterations = maxIterations
  }

  function resetExtractionPrompt(): void {
    webImportSettings.value.extraction.prompt = DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT
  }

  // ============================================================
  // 下载设置
  // ============================================================

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

  // ============================================================
  // 图片预处理设置
  // ============================================================

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

  // ============================================================
  // 高级设置
  // ============================================================

  function setCustomCookie(cookie: string): void {
    webImportSettings.value.advanced.customCookie = cookie
  }

  function setCustomHeaders(headers: string): void {
    webImportSettings.value.advanced.customHeaders = headers
  }

  function setBypassProxy(bypass: boolean): void {
    webImportSettings.value.advanced.bypassProxy = bypass
  }

  // ============================================================
  // UI 设置
  // ============================================================

  function setShowAgentLogs(show: boolean): void {
    webImportSettings.value.ui.showAgentLogs = show
  }

  function setAutoImport(autoImport: boolean): void {
    webImportSettings.value.ui.autoImport = autoImport
  }

  function saveAgentProviderConfig(provider: string): void {
    const canonicalProvider = normalizeProviderId(provider)
    if (!canonicalProvider) return

    providerConfigs.value.agent[canonicalProvider] = {
      apiKey: webImportSettings.value.agent.apiKey,
      modelName: webImportSettings.value.agent.modelName,
      customBaseUrl: webImportSettings.value.agent.customBaseUrl
    }
  }

  function restoreAgentProviderConfig(provider: string): void {
    const canonicalProvider = normalizeProviderId(provider)
    const cached = canonicalProvider ? providerConfigs.value.agent[canonicalProvider] : undefined

    if (cached) {
      webImportSettings.value.agent.apiKey = cached.apiKey ?? ''
      webImportSettings.value.agent.modelName = cached.modelName ?? ''
      webImportSettings.value.agent.customBaseUrl = cached.customBaseUrl ?? ''
      return
    }

    const emptyConfig = createEmptyAgentProviderConfig()
    webImportSettings.value.agent.apiKey = emptyConfig.apiKey
    webImportSettings.value.agent.modelName = emptyConfig.modelName
    webImportSettings.value.agent.customBaseUrl = emptyConfig.customBaseUrl
  }

  return {
    // Firecrawl
    setFirecrawlApiKey,
    // Agent
    setAgentProvider,
    setAgentApiKey,
    setAgentBaseUrl,
    setAgentModelName,
    setAgentUseStream,
    setAgentForceJson,
    setAgentTimeout,
    saveAgentProviderConfig,
    restoreAgentProviderConfig,
    // 提取
    setExtractionPrompt,
    setExtractionMaxIterations,
    resetExtractionPrompt,
    // 下载
    setDownloadConcurrency,
    setDownloadTimeout,
    setDownloadRetries,
    setDownloadDelay,
    setDownloadUseReferer,
    // 图片预处理
    setImagePreprocessEnabled,
    setImageAutoRotate,
    setImageCompressionEnabled,
    setImageCompressionQuality,
    setImageMaxWidth,
    setImageMaxHeight,
    setImageFormatConvertEnabled,
    setImageTargetFormat,
    // 高级
    setCustomCookie,
    setCustomHeaders,
    setBypassProxy,
    // UI
    setShowAgentLogs,
    setAutoImport,
  }
}
