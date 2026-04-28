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
  WebImportProviderConfigs,
  WebImportSettings,
  WebImportSettingsPayload,
  WebImportState,
} from '@/types/webImport'
import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import { normalizeProviderId } from '@/config/aiProviders'
import {
  getWebImportSettings,
  saveWebImportSettings,
} from '@/api/webImport'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
  useWebImportSettings,
} from './settings/modules/webImport'

const STORAGE_KEY_DISCLAIMER_ACCEPTED = 'webImportDisclaimerAccepted'

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function serializeValue(value: unknown): string {
  return JSON.stringify(value)
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

  // ============================================================
  // 深度合并
  // ============================================================

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function deepMerge(target: any, source: any): any {
    const result = { ...target }
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        const sourceValue = source[key]
        const targetValue = target[key]
        if (
          sourceValue !== null &&
          typeof sourceValue === 'object' &&
          !Array.isArray(sourceValue) &&
          targetValue !== null &&
          typeof targetValue === 'object' &&
          !Array.isArray(targetValue)
        ) {
          result[key] = deepMerge(targetValue, sourceValue)
        } else if (sourceValue !== undefined) {
          result[key] = sourceValue
        }
      }
    }
    return result
  }

  function normalizeProviderConfigsValue(
    value: Partial<WebImportProviderConfigs> | null | undefined
  ): WebImportProviderConfigs {
    const merged = deepMerge(createDefaultWebImportProviderConfigs(), value || {})
    const normalizedAgentConfigs: WebImportProviderConfigs['agent'] = {}

    for (const [provider, config] of Object.entries(merged.agent || {})) {
      const normalizedConfig = (config || {}) as Partial<WebImportProviderConfigs['agent'][string]>
      normalizedAgentConfigs[normalizeProviderId(provider)] = {
        apiKey: normalizedConfig.apiKey || '',
        modelName: normalizedConfig.modelName || '',
        customBaseUrl: normalizedConfig.customBaseUrl || ''
      }
    }

    return {
      agent: normalizedAgentConfigs
    }
  }

  function normalizeSettingsValue(value: Partial<WebImportSettings> | null | undefined): WebImportSettings {
    const merged = deepMerge(createDefaultWebImportSettings(), value || {})
    merged.agent.provider = normalizeProviderId(merged.agent.provider) as WebImportSettings['agent']['provider']
    return merged
  }

  function syncDraftFromCommitted(): void {
    draftSettings.value = cloneValue(settings.value)
    draftProviderConfigs.value = cloneValue(providerConfigs.value)
  }

  function toStoragePayload(): WebImportSettingsPayload {
    return {
      settings: cloneValue(settings.value),
      providerConfigs: cloneValue(providerConfigs.value)
    }
  }

  function applyLoadedPayload(payload: Partial<WebImportSettingsPayload>): void {
    settings.value = normalizeSettingsValue(payload.settings)
    providerConfigs.value = normalizeProviderConfigsValue(payload.providerConfigs)
    syncDraftFromCommitted()
  }

  function hasMeaningfulSettingsPayload(payload: {
    settings?: Partial<WebImportSettings>
    providerConfigs?: Partial<WebImportProviderConfigs>
  }): boolean {
    const normalizedSettings = normalizeSettingsValue(payload.settings)
    const normalizedProviderConfigs = normalizeProviderConfigsValue(payload.providerConfigs)

    return (
      serializeValue(normalizedSettings) !== serializeValue(createDefaultWebImportSettings()) ||
      serializeValue(normalizedProviderConfigs) !== serializeValue(createDefaultWebImportProviderConfigs())
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
      const isCombinedPayload = parsed && typeof parsed === 'object' && 'settings' in parsed
      const payload: Partial<WebImportSettingsPayload> = isCombinedPayload
        ? {
          settings: parsed.settings,
          providerConfigs: parsed.providerConfigs
        }
        : {
          settings: parsed,
          providerConfigs: createDefaultWebImportProviderConfigs()
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

      const hasStoredSettings = response.hasStoredSettings === true || hasMeaningfulSettingsPayload({
        settings: response.settings,
        providerConfigs: response.providerConfigs
      })
      if (!hasStoredSettings) return false

      applyLoadedPayload({
        settings: response.settings,
        providerConfigs: response.providerConfigs
      })
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
      const response = await saveWebImportSettings({
        settings: cloneValue(settings.value),
        providerConfigs: cloneValue(providerConfigs.value)
      })
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

    settings.value = normalizeSettingsValue(draftSettings.value)
    providerConfigs.value = normalizeProviderConfigsValue(draftProviderConfigs.value)
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
