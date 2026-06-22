/**
 * 网页导入状态管理 Store
 * 管理网页导入设置、设置草稿和运行时状态
 */

import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import type {
  AgentLog,
  DownloadedImage,
  ExtractResult,
  WebImportAgentProviderConfig,
  WebImportProviderConfigs,
  WebImportSettings,
  WebImportSettingsPayload,
  WebImportState,
} from '@/types/webImport'
import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import {
  getWebImportSettings,
  saveWebImportSettings,
} from '@/api/webImport'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
  isWebImportAgentProvider,
  useWebImportSettings,
} from './settings/modules/webImport'

const STORAGE_KEY_DISCLAIMER_ACCEPTED = 'webImportDisclaimerAccepted'
export const WEB_IMPORT_SETTINGS_SCHEMA_VERSION = 1

type PlainRecord = Record<string, unknown>

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function serializeValue(value: unknown): string {
  return JSON.stringify(value)
}

function isPlainRecord(value: unknown): value is PlainRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasExactKeys(value: PlainRecord, keys: readonly string[]): boolean {
  const actualKeys = Object.keys(value)
  return actualKeys.length === keys.length
    && keys.every(key => Object.prototype.hasOwnProperty.call(value, key))
}

function parseString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function parseNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function parseImageFormat(value: unknown): WebImportSettings['imagePreprocess']['formatConvert']['targetFormat'] | null {
  return value === 'jpeg' || value === 'png' || value === 'webp' || value === 'original'
    ? value
    : null
}

function parseCurrentWebImportSettings(value: unknown): WebImportSettings | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, [
    'firecrawl',
    'agent',
    'extraction',
    'download',
    'imagePreprocess',
    'advanced',
    'ui',
  ])) {
    return null
  }

  const {
    firecrawl,
    agent,
    extraction,
    download,
    imagePreprocess,
    advanced,
    ui,
  } = value

  if (
    !isPlainRecord(firecrawl) ||
    !isPlainRecord(agent) ||
    !isPlainRecord(extraction) ||
    !isPlainRecord(download) ||
    !isPlainRecord(imagePreprocess) ||
    !isPlainRecord(advanced) ||
    !isPlainRecord(ui)
  ) {
    return null
  }

  if (
    !hasExactKeys(firecrawl, ['apiKey']) ||
    !hasExactKeys(agent, ['provider', 'apiKey', 'customBaseUrl', 'modelName', 'useStream', 'forceJsonOutput', 'maxRetries', 'timeout']) ||
    !hasExactKeys(extraction, ['prompt', 'maxIterations']) ||
    !hasExactKeys(download, ['concurrency', 'timeout', 'retries', 'delay', 'useReferer']) ||
    !hasExactKeys(imagePreprocess, ['enabled', 'autoRotate', 'compression', 'formatConvert']) ||
    !hasExactKeys(advanced, ['customCookie', 'customHeaders', 'bypassProxy']) ||
    !hasExactKeys(ui, ['showAgentLogs', 'autoImport'])
  ) {
    return null
  }

  if (!isPlainRecord(imagePreprocess.compression) || !isPlainRecord(imagePreprocess.formatConvert)) {
    return null
  }
  const compression = imagePreprocess.compression
  const formatConvert = imagePreprocess.formatConvert
  if (
    !hasExactKeys(compression, ['enabled', 'quality', 'maxWidth', 'maxHeight']) ||
    !hasExactKeys(formatConvert, ['enabled', 'targetFormat'])
  ) {
    return null
  }

  const provider = parseString(agent.provider)
  const targetFormat = parseImageFormat(formatConvert.targetFormat)
  if (!isWebImportAgentProvider(provider) || !targetFormat) {
    return null
  }

  const parsed = {
    firecrawl: {
      apiKey: parseString(firecrawl.apiKey),
    },
    agent: {
      provider,
      apiKey: parseString(agent.apiKey),
      customBaseUrl: parseString(agent.customBaseUrl),
      modelName: parseString(agent.modelName),
      useStream: parseBoolean(agent.useStream),
      forceJsonOutput: parseBoolean(agent.forceJsonOutput),
      maxRetries: parseNumber(agent.maxRetries),
      timeout: parseNumber(agent.timeout),
    },
    extraction: {
      prompt: parseString(extraction.prompt),
      maxIterations: parseNumber(extraction.maxIterations),
    },
    download: {
      concurrency: parseNumber(download.concurrency),
      timeout: parseNumber(download.timeout),
      retries: parseNumber(download.retries),
      delay: parseNumber(download.delay),
      useReferer: parseBoolean(download.useReferer),
    },
    imagePreprocess: {
      enabled: parseBoolean(imagePreprocess.enabled),
      autoRotate: parseBoolean(imagePreprocess.autoRotate),
      compression: {
        enabled: parseBoolean(compression.enabled),
        quality: parseNumber(compression.quality),
        maxWidth: parseNumber(compression.maxWidth),
        maxHeight: parseNumber(compression.maxHeight),
      },
      formatConvert: {
        enabled: parseBoolean(formatConvert.enabled),
        targetFormat,
      },
    },
    advanced: {
      customCookie: parseString(advanced.customCookie),
      customHeaders: parseString(advanced.customHeaders),
      bypassProxy: parseBoolean(advanced.bypassProxy),
    },
    ui: {
      showAgentLogs: parseBoolean(ui.showAgentLogs),
      autoImport: parseBoolean(ui.autoImport),
    },
  }

  if (
    parsed.firecrawl.apiKey === null ||
    parsed.agent.apiKey === null ||
    parsed.agent.customBaseUrl === null ||
    parsed.agent.modelName === null ||
    parsed.agent.useStream === null ||
    parsed.agent.forceJsonOutput === null ||
    parsed.agent.maxRetries === null ||
    parsed.agent.timeout === null ||
    parsed.extraction.prompt === null ||
    parsed.extraction.maxIterations === null ||
    parsed.download.concurrency === null ||
    parsed.download.timeout === null ||
    parsed.download.retries === null ||
    parsed.download.delay === null ||
    parsed.download.useReferer === null ||
    parsed.imagePreprocess.enabled === null ||
    parsed.imagePreprocess.autoRotate === null ||
    parsed.imagePreprocess.compression.enabled === null ||
    parsed.imagePreprocess.compression.quality === null ||
    parsed.imagePreprocess.compression.maxWidth === null ||
    parsed.imagePreprocess.compression.maxHeight === null ||
    parsed.imagePreprocess.formatConvert.enabled === null ||
    parsed.advanced.customCookie === null ||
    parsed.advanced.customHeaders === null ||
    parsed.advanced.bypassProxy === null ||
    parsed.ui.showAgentLogs === null ||
    parsed.ui.autoImport === null
  ) {
    return null
  }

  return parsed as WebImportSettings
}

function parseCurrentAgentProviderConfig(value: unknown): WebImportAgentProviderConfig | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['apiKey', 'modelName', 'customBaseUrl'])) {
    return null
  }
  const apiKey = parseString(value.apiKey)
  const modelName = parseString(value.modelName)
  const customBaseUrl = parseString(value.customBaseUrl)
  if (apiKey === null || modelName === null || customBaseUrl === null) {
    return null
  }
  return { apiKey, modelName, customBaseUrl }
}

function parseCurrentProviderConfigs(value: unknown): WebImportProviderConfigs | null {
  if (!isPlainRecord(value) || !hasExactKeys(value, ['agent']) || !isPlainRecord(value.agent)) {
    return null
  }

  const agent: WebImportProviderConfigs['agent'] = {}
  for (const [provider, config] of Object.entries(value.agent)) {
    if (!isWebImportAgentProvider(provider)) {
      continue
    }
    const parsed = parseCurrentAgentProviderConfig(config)
    if (parsed) {
      agent[provider] = parsed
    }
  }
  return { agent }
}

function parseCurrentWebImportPayload(value: unknown): WebImportSettingsPayload | null {
  if (!isPlainRecord(value)) return null
  const settings = parseCurrentWebImportSettings(value.settings)
  const providerConfigs = parseCurrentProviderConfigs(value.providerConfigs)
  if (!settings || !providerConfigs) return null
  return {
    webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
    settings,
    providerConfigs,
  }
}

function parseCurrentLocalPayload(value: unknown): WebImportSettingsPayload | null {
  if (!isPlainRecord(value) || value.webImportSettingsSchemaVersion !== WEB_IMPORT_SETTINGS_SCHEMA_VERSION) {
    return null
  }
  return parseCurrentWebImportPayload(value)
}

export const useWebImportStore = defineStore('webImport', () => {
  // ============================================================
  // 已提交设置
  // ============================================================

  const settings = ref<WebImportSettings>(createDefaultWebImportSettings())
  const providerConfigs = ref<WebImportProviderConfigs>(createDefaultWebImportProviderConfigs())

  // ============================================================
  // 草稿设置
  // ============================================================

  const draftSettings = ref<WebImportSettings>(cloneValue(settings.value))
  const draftProviderConfigs = ref<WebImportProviderConfigs>(cloneValue(providerConfigs.value))
  const isSavingSettings = ref(false)
  const isInitializingSettings = ref(false)
  const hasLoadedBackendSettings = ref(false)
  let initPromise: Promise<void> | null = null

  // ============================================================
  // 运行时状态
  // ============================================================

  const status = ref<WebImportState['status']>('idle')
  const url = ref('')
  const logs = ref<AgentLog[]>([])
  const extractResult = ref<ExtractResult | null>(null)
  const selectedPages = ref<Set<number>>(new Set())
  const downloadProgress = ref({ current: 0, total: 0 })
  const downloadedImages = ref<DownloadedImage[]>([])
  const error = ref<string | null>(null)
  const modalVisible = ref(false)
  const disclaimerAccepted = ref(false)
  const disclaimerVisible = ref(false)

  // ============================================================
  // 计算属性
  // ============================================================

  const isExtracting = computed(() => status.value === 'extracting')
  const isDownloading = computed(() => status.value === 'downloading')
  const isProcessing = computed(() => isExtracting.value || isDownloading.value)
  const selectedCount = computed(() => selectedPages.value.size)
  const downloadProgressPercent = computed(() => {
    if (downloadProgress.value.total === 0) return 0
    return Math.round((downloadProgress.value.current / downloadProgress.value.total) * 100)
  })
  const hasUnsavedSettings = computed(() => {
    return (
      serializeValue(settings.value) !== serializeValue(draftSettings.value) ||
      serializeValue(providerConfigs.value) !== serializeValue(draftProviderConfigs.value)
    )
  })

  function syncDraftFromCommitted(): void {
    draftSettings.value = cloneValue(settings.value)
    draftProviderConfigs.value = cloneValue(providerConfigs.value)
  }

  function toStoragePayload(): WebImportSettingsPayload {
    return {
      webImportSettingsSchemaVersion: WEB_IMPORT_SETTINGS_SCHEMA_VERSION,
      settings: cloneValue(settings.value),
      providerConfigs: cloneValue(providerConfigs.value)
    }
  }

  function applyLoadedPayload(payload: unknown): boolean {
    const parsed = parseCurrentWebImportPayload(payload)
    if (!parsed) {
      return false
    }
    settings.value = parsed.settings
    providerConfigs.value = parsed.providerConfigs
    syncDraftFromCommitted()
    return true
  }

  function hasMeaningfulSettingsPayload(payload: unknown): boolean {
    const parsed = parseCurrentWebImportPayload(payload)
    if (!parsed) {
      return false
    }

    return (
      serializeValue(parsed.settings) !== serializeValue(createDefaultWebImportSettings()) ||
      serializeValue(parsed.providerConfigs) !== serializeValue(createDefaultWebImportProviderConfigs())
    )
  }

  // ============================================================
  // localStorage 持久化
  // ============================================================

  function saveToStorage(): void {
    try {
      localStorage.setItem(STORAGE_KEY_WEB_IMPORT_SETTINGS, JSON.stringify(toStoragePayload()))
    } catch (e) {
      console.error('保存网页导入设置失败:', e)
    }
  }

  function loadFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_WEB_IMPORT_SETTINGS)
      if (!data) return

      const parsed = JSON.parse(data)
      const payload = parseCurrentLocalPayload(parsed)
      if (!payload) {
        console.warn('网页导入本地设置不符合当前 schema，已忽略该设置对象')
        syncDraftFromCommitted()
        return
      }
      applyLoadedPayload(payload)
    } catch (e) {
      console.error('加载网页导入设置失败:', e)
      syncDraftFromCommitted()
    }
  }

  async function loadFromBackend(): Promise<boolean> {
    try {
      const response = await getWebImportSettings()
      if (!response.success) return false

      const responsePayload = {
        settings: response.settings,
        providerConfigs: response.providerConfigs
      }
      const hasStoredSettings = response.hasStoredSettings === true || hasMeaningfulSettingsPayload(responsePayload)
      if (!hasStoredSettings) return false

      if (!applyLoadedPayload(responsePayload)) {
        console.warn('网页导入后端设置不符合当前 schema，已忽略该设置对象')
        return false
      }
      saveToStorage()
      hasLoadedBackendSettings.value = true
      return true
    } catch (e) {
      console.error('从后端加载网页导入设置失败:', e)
      return false
    }
  }

  async function saveToBackend(): Promise<boolean> {
    try {
      const response = await saveWebImportSettings(toStoragePayload())
      return Boolean(response.success)
    } catch (e) {
      console.error('保存网页导入设置到后端失败:', e)
      return false
    }
  }

  async function initSettings(force = false): Promise<void> {
    if (hasLoadedBackendSettings.value && !force) return
    if (initPromise && !force) {
      await initPromise
      return
    }

    if (force) {
      hasLoadedBackendSettings.value = false
    }

    initPromise = (async () => {
      isInitializingSettings.value = true
      try {
        loadFromStorage()
        await loadFromBackend()
      } finally {
        isInitializingSettings.value = false
      }
    })()

    try {
      await initPromise
    } finally {
      initPromise = null
    }
  }

  // ============================================================
  // 设置草稿操作
  // ============================================================

  function beginSettingsEdit(): void {
    syncDraftFromCommitted()
  }

  function discardSettingsChanges(): void {
    syncDraftFromCommitted()
  }

  async function saveSettings(): Promise<boolean> {
    if (isSavingSettings.value) return false

    settingsMethods.saveAgentProviderConfig(draftSettings.value.agent.provider)

    const previousSettings = cloneValue(settings.value)
    const previousProviderConfigs = cloneValue(providerConfigs.value)

    const parsedDraft = parseCurrentWebImportPayload({
      settings: draftSettings.value,
      providerConfigs: draftProviderConfigs.value,
    })
    if (!parsedDraft) {
      return false
    }
    settings.value = parsedDraft.settings
    providerConfigs.value = parsedDraft.providerConfigs
    saveToStorage()

    isSavingSettings.value = true
    try {
      const success = await saveToBackend()
      if (!success) {
        settings.value = previousSettings
        providerConfigs.value = previousProviderConfigs
        saveToStorage()
        return false
      }

      syncDraftFromCommitted()
      return true
    } finally {
      isSavingSettings.value = false
    }
  }

  // ============================================================
  // 运行时状态操作
  // ============================================================

  async function openModal(): Promise<void> {
    if (!disclaimerAccepted.value) {
      disclaimerVisible.value = true
      return
    }

    await initSettings()
    beginSettingsEdit()
    modalVisible.value = true
  }

  async function acceptDisclaimer(): Promise<void> {
    disclaimerAccepted.value = true
    disclaimerVisible.value = false

    try {
      localStorage.setItem(STORAGE_KEY_DISCLAIMER_ACCEPTED, 'true')
    } catch (e) {
      console.error('保存免责声明状态失败:', e)
    }

    await initSettings()
    beginSettingsEdit()
    modalVisible.value = true
  }

  function rejectDisclaimer(): void {
    disclaimerVisible.value = false
  }

  function loadDisclaimerState(): void {
    try {
      const accepted = localStorage.getItem(STORAGE_KEY_DISCLAIMER_ACCEPTED)
      disclaimerAccepted.value = accepted === 'true'
    } catch (e) {
      console.error('加载免责声明状态失败:', e)
    }
  }

  function closeModal(): void {
    modalVisible.value = false
  }

  function resetState(): void {
    status.value = 'idle'
    url.value = ''
    logs.value = []
    extractResult.value = null
    selectedPages.value = new Set()
    downloadProgress.value = { current: 0, total: 0 }
    downloadedImages.value = []
    error.value = null
  }

  function setUrl(newUrl: string): void {
    url.value = newUrl
  }

  function addLog(log: AgentLog): void {
    logs.value.push(log)
  }

  function clearLogs(): void {
    logs.value = []
  }

  function setExtractResult(result: ExtractResult): void {
    if (extractResult.value && extractResult.value.pages.length > 0) {
      extractResult.value.comicTitle = result.comicTitle
      extractResult.value.chapterTitle = result.chapterTitle
      extractResult.value.totalPages = result.totalPages
      extractResult.value.sourceUrl = result.sourceUrl
      extractResult.value.referer = result.referer
      extractResult.value.engine = result.engine
      extractResult.value.success = result.success
      extractResult.value.error = result.error
    } else {
      extractResult.value = result
      if (result.success && result.pages) {
        selectedPages.value = new Set(result.pages.map((p) => p.pageNumber))
      }
    }
  }

  function togglePageSelection(pageNumber: number): void {
    if (selectedPages.value.has(pageNumber)) {
      selectedPages.value.delete(pageNumber)
    } else {
      selectedPages.value.add(pageNumber)
    }
    selectedPages.value = new Set(selectedPages.value)
  }

  function toggleSelectAll(): void {
    if (!extractResult.value?.pages) return

    if (selectedPages.value.size === extractResult.value.pages.length) {
      selectedPages.value = new Set()
    } else {
      selectedPages.value = new Set(extractResult.value.pages.map((p) => p.pageNumber))
    }
  }

  function setStatus(newStatus: WebImportState['status']): void {
    status.value = newStatus
  }

  function setError(errorMsg: string | null): void {
    error.value = errorMsg
    if (errorMsg) {
      status.value = 'error'
    }
  }

  function updateDownloadProgress(current: number, total: number): void {
    downloadProgress.value = { current, total }
  }

  function setDownloadedImages(images: DownloadedImage[]): void {
    downloadedImages.value = images
  }

  function addPageIncremental(page: { pageNumber: number; imageUrl: string; localPath?: string }): void {
    if (!extractResult.value) {
      extractResult.value = {
        success: true,
        comicTitle: '',
        chapterTitle: '',
        pages: [],
        totalPages: 0,
        sourceUrl: url.value,
        referer: '',
        engine: 'gallery-dl'
      }
    }

    extractResult.value.pages.push(page)
    extractResult.value.totalPages = extractResult.value.pages.length

    selectedPages.value.add(page.pageNumber)
    selectedPages.value = new Set(selectedPages.value)
  }

  const settingsMethods = useWebImportSettings(draftSettings, draftProviderConfigs)

  loadFromStorage()
  loadDisclaimerState()
  syncDraftFromCommitted()

  return {
    settings,
    providerConfigs,
    draftSettings,
    draftProviderConfigs,
    status,
    url,
    logs,
    extractResult,
    selectedPages,
    downloadProgress,
    downloadedImages,
    error,
    modalVisible,
    disclaimerAccepted,
    disclaimerVisible,
    isExtracting,
    isDownloading,
    isProcessing,
    selectedCount,
    downloadProgressPercent,
    hasUnsavedSettings,
    isSavingSettings,
    isInitializingSettings,
    saveToStorage,
    loadFromStorage,
    loadFromBackend,
    saveToBackend,
    initSettings,
    openModal,
    closeModal,
    resetState,
    setUrl,
    addLog,
    clearLogs,
    setExtractResult,
    togglePageSelection,
    toggleSelectAll,
    setStatus,
    setError,
    updateDownloadProgress,
    setDownloadedImages,
    addPageIncremental,
    acceptDisclaimer,
    rejectDisclaimer,
    beginSettingsEdit,
    discardSettingsChanges,
    saveSettings,
    ...settingsMethods
  }
})
