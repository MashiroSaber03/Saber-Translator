/**
 * 网页导入设置模块
 */

import type { Ref } from 'vue'
import type { WebImportSettings } from '@/types/webImport'
import { DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT } from '@/constants'

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

// ============================================================
// Composable
// ============================================================

export function useWebImportSettings(
  webImportSettings: Ref<WebImportSettings>,
  saveToStorage: () => void
) {
  // ============================================================
  // Firecrawl 设置
  // ============================================================

  function setFirecrawlApiKey(apiKey: string): void {
    webImportSettings.value.firecrawl.apiKey = apiKey
    saveToStorage()
  }

  // ============================================================
  // Agent 设置
  // ============================================================

  function setAgentProvider(provider: string): void {
    webImportSettings.value.agent.provider = provider as WebImportSettings['agent']['provider']
    saveToStorage()
  }

  function setAgentApiKey(apiKey: string): void {
    webImportSettings.value.agent.apiKey = apiKey
    saveToStorage()
  }

  function setAgentBaseUrl(baseUrl: string): void {
    webImportSettings.value.agent.customBaseUrl = baseUrl
    saveToStorage()
  }

  function setAgentModelName(modelName: string): void {
    webImportSettings.value.agent.modelName = modelName
    saveToStorage()
  }

  function setAgentUseStream(useStream: boolean): void {
    webImportSettings.value.agent.useStream = useStream
    saveToStorage()
  }

  function setAgentForceJson(forceJson: boolean): void {
    webImportSettings.value.agent.forceJsonOutput = forceJson
    saveToStorage()
  }

  function setAgentTimeout(timeout: number): void {
    webImportSettings.value.agent.timeout = timeout
    saveToStorage()
  }

  // ============================================================
  // 提取设置
  // ============================================================

  function setExtractionPrompt(prompt: string): void {
    webImportSettings.value.extraction.prompt = prompt
    saveToStorage()
  }

  function setExtractionMaxIterations(maxIterations: number): void {
    webImportSettings.value.extraction.maxIterations = maxIterations
    saveToStorage()
  }

  function resetExtractionPrompt(): void {
    webImportSettings.value.extraction.prompt = DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT
    saveToStorage()
  }

  // ============================================================
  // 下载设置
  // ============================================================

  function setDownloadConcurrency(concurrency: number): void {
    webImportSettings.value.download.concurrency = concurrency
    saveToStorage()
  }

  function setDownloadTimeout(timeout: number): void {
    webImportSettings.value.download.timeout = timeout
    saveToStorage()
  }

  function setDownloadRetries(retries: number): void {
    webImportSettings.value.download.retries = retries
    saveToStorage()
  }

  function setDownloadDelay(delay: number): void {
    webImportSettings.value.download.delay = delay
    saveToStorage()
  }

  function setDownloadUseReferer(useReferer: boolean): void {
    webImportSettings.value.download.useReferer = useReferer
    saveToStorage()
  }

  // ============================================================
  // 图片预处理设置
  // ============================================================

  function setImagePreprocessEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.enabled = enabled
    saveToStorage()
  }

  function setImageAutoRotate(autoRotate: boolean): void {
    webImportSettings.value.imagePreprocess.autoRotate = autoRotate
    saveToStorage()
  }

  function setImageCompressionEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.compression.enabled = enabled
    saveToStorage()
  }

  function setImageCompressionQuality(quality: number): void {
    webImportSettings.value.imagePreprocess.compression.quality = quality
    saveToStorage()
  }

  function setImageMaxWidth(maxWidth: number): void {
    webImportSettings.value.imagePreprocess.compression.maxWidth = maxWidth
    saveToStorage()
  }

  function setImageMaxHeight(maxHeight: number): void {
    webImportSettings.value.imagePreprocess.compression.maxHeight = maxHeight
    saveToStorage()
  }

  function setImageFormatConvertEnabled(enabled: boolean): void {
    webImportSettings.value.imagePreprocess.formatConvert.enabled = enabled
    saveToStorage()
  }

  function setImageTargetFormat(format: 'jpeg' | 'png' | 'webp' | 'original'): void {
    webImportSettings.value.imagePreprocess.formatConvert.targetFormat = format
    saveToStorage()
  }

  // ============================================================
  // 高级设置
  // ============================================================

  function setCustomCookie(cookie: string): void {
    webImportSettings.value.advanced.customCookie = cookie
    saveToStorage()
  }

  function setCustomHeaders(headers: string): void {
    webImportSettings.value.advanced.customHeaders = headers
    saveToStorage()
  }

  function setBypassProxy(bypass: boolean): void {
    webImportSettings.value.advanced.bypassProxy = bypass
    saveToStorage()
  }

  // ============================================================
  // UI 设置
  // ============================================================

  function setShowAgentLogs(show: boolean): void {
    webImportSettings.value.ui.showAgentLogs = show
    saveToStorage()
  }

  function setAutoImport(autoImport: boolean): void {
    webImportSettings.value.ui.autoImport = autoImport
    saveToStorage()
  }

  // ============================================================
  // 批量更新
  // ============================================================

  function updateWebImportSettings(updates: Partial<WebImportSettings>): void {
    webImportSettings.value = {
      ...webImportSettings.value,
      ...updates
    }
    saveToStorage()
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
    // 批量更新
    updateWebImportSettings
  }
}
