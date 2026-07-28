import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import type {
  AgentLog,
  DownloadedImage,
  ExtractResult,
  WebImportProviderConfigs,
  WebImportSettings,
  WebImportState,
} from '@/types/webImport'
import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import {
  getV2Settings,
  saveV2SettingsTransaction,
  type V2CredentialEdit,
  type V2CredentialSummary,
  type V2ProviderSettingMutation,
} from '@/api/v2/settings'
import { deepClone } from '@/utils/deepClone'
import {
  createDefaultWebImportProviderConfigs,
  createDefaultWebImportSettings,
  useWebImportSettings,
} from './settings/modules/webImport'
import {
  parseWebImportSettingsPayload,
  serializeWebImportSettingsValue,
} from './webImportSettingsPayload'

const STORAGE_KEY_DISCLAIMER_ACCEPTED = 'webImportDisclaimerAccepted'
export { WEB_IMPORT_SETTINGS_SCHEMA_VERSION } from './webImportSettingsPayload'

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, Math.round(value)))
}

function parseCustomHeaders(value: string): Record<string, string> | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return Object.fromEntries(
        Object.entries(parsed).map(([name, headerValue]) => [name, String(headerValue)]),
      )
    }
  } catch {
    // Fall through to the user-friendly "Header: value" format.
  }
  const headers: Record<string, string> = {}
  for (const line of trimmed.split(/\r?\n/)) {
    const separator = line.indexOf(':')
    if (separator <= 0) continue
    const name = line.slice(0, separator).trim()
    const headerValue = line.slice(separator + 1).trim()
    if (name && headerValue) headers[name] = headerValue
  }
  return Object.keys(headers).length > 0 ? headers : undefined
}

export const useWebImportStore = defineStore('webImport', () => {
  const settings = ref<WebImportSettings>(createDefaultWebImportSettings())
  const providerConfigs = ref<WebImportProviderConfigs>(createDefaultWebImportProviderConfigs())

  const draftSettings = ref<WebImportSettings>(deepClone(settings.value))
  const draftProviderConfigs = ref<WebImportProviderConfigs>(deepClone(providerConfigs.value))
  const isSavingSettings = ref(false)
  const isInitializingSettings = ref(false)
  const hasLoadedBackendSettings = ref(false)
  const credentialSummaries = ref<V2CredentialSummary[]>([])
  let settingsRevision = 0
  let providerRevisions = new Map<string, number>()
  let initPromise: Promise<void> | null = null

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

  const isExtracting = computed(() => status.value === 'extracting')
  const isDownloading = computed(() => status.value === 'downloading')
  const isProcessing = computed(() => isExtracting.value || isDownloading.value)
  const selectedCount = computed(() => selectedPages.value.size)
  const downloadProgressPercent = computed(() => {
    if (downloadProgress.value.total === 0) return 0
    return clampPercent((downloadProgress.value.current / downloadProgress.value.total) * 100)
  })
  const hasUnsavedSettings = computed(() => {
    return (
      serializeWebImportSettingsValue(settings.value) !== serializeWebImportSettingsValue(draftSettings.value) ||
      serializeWebImportSettingsValue(providerConfigs.value) !== serializeWebImportSettingsValue(draftProviderConfigs.value)
    )
  })

  function syncDraftFromCommitted(): void {
    draftSettings.value = deepClone(settings.value)
    draftProviderConfigs.value = deepClone(providerConfigs.value)
  }

  function applyLoadedPayload(payload: unknown): boolean {
    const parsed = parseWebImportSettingsPayload(payload)
    if (!parsed) {
      return false
    }
    settings.value = parsed.settings
    providerConfigs.value = parsed.providerConfigs
    syncDraftFromCommitted()
    return true
  }

  function saveToStorage(): void {
    try {
      localStorage.removeItem(STORAGE_KEY_WEB_IMPORT_SETTINGS)
    } catch {
      return
    }
  }

  function loadFromStorage(): void {
    saveToStorage()
    syncDraftFromCommitted()
  }

  function hasCredential(domain: string, provider: string): boolean {
    return credentialSummaries.value.some(
      row => row.domain === domain && row.provider === provider && row.hasKey,
    )
  }

  async function loadFromBackend(): Promise<boolean> {
    try {
      const response = await getV2Settings(['web_import', 'web_import_agent', 'web_import_firecrawl', 'web_import_http'])
      const entry = response.settings.find(row => row.domain === 'web_import')
      settingsRevision = entry?.revision ?? 0
      providerRevisions = new Map(
        response.providerSettings.map(row => [`${row.domain}\u0000${row.provider}`, row.revision]),
      )
      credentialSummaries.value = response.credentials
      const loadedProviderConfigs = createDefaultWebImportProviderConfigs()
      for (const row of response.providerSettings) {
        if (row.domain !== 'web_import_agent') continue
        loadedProviderConfigs.agent[row.provider] = {
          apiKey: '',
          modelName: String(row.payload.modelName ?? ''),
          customBaseUrl: String(row.payload.customBaseUrl ?? ''),
        }
      }
      const loadedSettings = entry?.payload && Object.keys(entry.payload).length > 0
        ? entry.payload
        : createDefaultWebImportSettings()
      if (!applyLoadedPayload({
        settings: loadedSettings,
        providerConfigs: loadedProviderConfigs,
      })) {
        settings.value = createDefaultWebImportSettings()
        providerConfigs.value = loadedProviderConfigs
      }
      settings.value.firecrawl.apiKey = ''
      settings.value.agent.apiKey = ''
      settings.value.advanced.customCookie = ''
      settings.value.advanced.customHeaders = ''
      settingsMethods.restoreAgentProviderConfig(settings.value.agent.provider)
      settings.value.agent.apiKey = ''
      syncDraftFromCommitted()
      hasLoadedBackendSettings.value = true
      return true
    } catch {
      return false
    }
  }

  async function saveToBackend(): Promise<boolean> {
    try {
      const providerSettings: V2ProviderSettingMutation[] = []
      const credentialEdits: V2CredentialEdit[] = []
      const addProvider = (
        domain: string,
        provider: string,
        payload: Record<string, unknown>,
        secret: Record<string, unknown>,
      ) => {
        const identity = `${domain}\u0000${provider}`
        const existing = credentialSummaries.value.find(
          row => row.domain === domain && row.provider === provider,
        )
        const nonEmptySecret = Object.fromEntries(
          Object.entries(secret).filter(([, value]) => value !== '' && value != null),
        )
        const mutation: V2ProviderSettingMutation = {
          domain,
          provider,
          payload,
          baseRevision: providerRevisions.get(identity) ?? 0,
        }
        if (Object.keys(nonEmptySecret).length > 0) {
          const clientRef = `credential:${domain}:${provider}`
          credentialEdits.push({
            domain,
            provider,
            secret: nonEmptySecret,
            baseRevision: existing?.revision ?? 0,
            credentialId: existing?.credentialId,
            clientRef,
          })
          mutation.credentialEditRef = clientRef
        } else if (existing) {
          mutation.credentialVersionId = existing.credentialVersionId
        }
        providerSettings.push(mutation)
      }

      for (const [provider, config] of Object.entries(providerConfigs.value.agent)) {
        addProvider(
          'web_import_agent',
          provider,
          {
            modelName: config.modelName,
            customBaseUrl: config.customBaseUrl,
          },
          { api_key: config.apiKey },
        )
      }
      addProvider(
        'web_import_firecrawl',
        'firecrawl',
        {},
        { api_key: settings.value.firecrawl.apiKey },
      )
      addProvider(
        'web_import_http',
        'headers',
        {},
        {
          cookie: settings.value.advanced.customCookie,
          headers: parseCustomHeaders(settings.value.advanced.customHeaders),
        },
      )

      const payload = deepClone(settings.value)
      payload.firecrawl.apiKey = ''
      payload.agent.apiKey = ''
      payload.advanced.customCookie = ''
      payload.advanced.customHeaders = ''
      await saveV2SettingsTransaction({
        settings: [{
          domain: 'web_import',
          payload: payload as unknown as Record<string, unknown>,
          baseRevision: settingsRevision,
          schemaVersion: 1,
        }],
        providerSettings,
        credentialEdits,
      })
      return await loadFromBackend()
    } catch {
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

  function beginSettingsEdit(): void {
    syncDraftFromCommitted()
  }

  function discardSettingsChanges(): void {
    syncDraftFromCommitted()
  }

  async function saveSettings(): Promise<boolean> {
    if (isSavingSettings.value) return false

    settingsMethods.saveAgentProviderConfig(draftSettings.value.agent.provider)

    const previousSettings = deepClone(settings.value)
    const previousProviderConfigs = deepClone(providerConfigs.value)

    const parsedDraft = parseWebImportSettingsPayload({
      settings: draftSettings.value,
      providerConfigs: draftProviderConfigs.value,
    })
    if (!parsedDraft) {
      return false
    }
    settings.value = parsedDraft.settings
    providerConfigs.value = parsedDraft.providerConfigs
    isSavingSettings.value = true
    try {
      const success = await saveToBackend()
      if (!success) {
        settings.value = previousSettings
        providerConfigs.value = previousProviderConfigs
        return false
      }

      syncDraftFromCommitted()
      return true
    } finally {
      isSavingSettings.value = false
    }
  }

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
    } catch {
      // Disclaimer persistence is best-effort; the accepted state is already applied in memory.
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
    } catch {
      disclaimerAccepted.value = false
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
    hasCredential,
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
