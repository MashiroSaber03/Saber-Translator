import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import type {
  AgentLog,
  ExtractResult,
  WebImportProviderConfigs,
  WebImportSettings,
  WebImportState,
} from '@/types/webImport'
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

function hydrateBackendWebImportSettings(value: unknown): unknown {
  const payload = deepClone(value) as Record<string, unknown>
  if (payload.firecrawl && typeof payload.firecrawl === 'object') {
    (payload.firecrawl as Record<string, unknown>).apiKey = ''
  }
  if (payload.agent && typeof payload.agent === 'object') {
    (payload.agent as Record<string, unknown>).apiKey = ''
  }
  if (payload.advanced && typeof payload.advanced === 'object') {
    const advanced = payload.advanced as Record<string, unknown>
    advanced.customCookie = ''
    advanced.customHeaders = ''
  }
  return payload
}

export const useWebImportStore = defineStore('webImport', () => {
  const settings = ref<WebImportSettings>(createDefaultWebImportSettings())
  const providerConfigs = ref<WebImportProviderConfigs>(createDefaultWebImportProviderConfigs())

  const draftSettings = ref<WebImportSettings>(deepClone(settings.value))
  const draftProviderConfigs = ref<WebImportProviderConfigs>(deepClone(providerConfigs.value))
  const isSavingSettings = ref(false)
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
  const selectedPageCount = ref(0)
  const downloadProgress = ref({ current: 0, total: 0 })
  const error = ref<string | null>(null)
  const modalVisible = ref(false)
  const disclaimerAccepted = ref(false)
  const disclaimerVisible = ref(false)

  const isExtracting = computed(() => status.value === 'extracting')
  const isDownloading = computed(() => status.value === 'downloading')
  const isProcessing = computed(() => isExtracting.value || isDownloading.value)
  const selectedCount = computed(() => selectedPageCount.value)
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

  function hasCredential(domain: string, provider: string): boolean {
    return credentialSummaries.value.some(
      row => row.domain === domain && row.provider === provider && row.hasKey,
    )
  }

  async function loadFromBackend(): Promise<boolean> {
    try {
      const response = await getV2Settings(['web_import', 'web_import_agent', 'web_import_firecrawl', 'web_import_http'])
      const entry = response.settings.find(row => row.domain === 'web_import')
      if (!entry) {
        hasLoadedBackendSettings.value = false
        return false
      }
      settingsRevision = entry.revision
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
      const loadedSettings = hydrateBackendWebImportSettings(entry.payload)
      if (!applyLoadedPayload({
        settings: loadedSettings,
        providerConfigs: loadedProviderConfigs,
      })) {
        hasLoadedBackendSettings.value = false
        return false
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
      hasLoadedBackendSettings.value = false
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
          schemaVersion: 1,
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
      delete (payload.firecrawl as Partial<typeof payload.firecrawl>).apiKey
      delete (payload.agent as Partial<typeof payload.agent>).apiKey
      delete (payload.advanced as Partial<typeof payload.advanced>).customCookie
      delete (payload.advanced as Partial<typeof payload.advanced>).customHeaders
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

    initPromise = loadFromBackend().then(() => undefined)

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
    if (!hasLoadedBackendSettings.value) {
      error.value = '网页导入设置加载失败'
      return
    }
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
    if (!hasLoadedBackendSettings.value) {
      error.value = '网页导入设置加载失败'
      return
    }
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
    selectedPageCount.value = 0
    downloadProgress.value = { current: 0, total: 0 }
    error.value = null
  }

  function setUrl(newUrl: string): void {
    url.value = newUrl
  }

  function addLog(log: AgentLog): void {
    logs.value.push(log)
  }

  function setExtractResult(result: ExtractResult): void {
    extractResult.value = result
    selectedPages.value = new Set(
      result.success ? result.pages.map(page => page.pageNumber) : [],
    )
    selectedPageCount.value = selectedPages.value.size
  }

  function setPagedExtractResult(
    result: ExtractResult,
    loadedSelectedPages: Iterable<number>,
    totalSelectedCount: number,
  ): void {
    extractResult.value = result
    selectedPages.value = new Set(loadedSelectedPages)
    selectedPageCount.value = Math.max(
      0,
      Math.min(result.totalPages, totalSelectedCount),
    )
  }

  function appendExtractResultPages(
    pages: ExtractResult['pages'],
    loadedSelectedPages: Iterable<number>,
  ): void {
    if (!extractResult.value) return
    const known = new Set(extractResult.value.pages.map(page => page.pageNumber))
    extractResult.value.pages.push(...pages.filter(page => !known.has(page.pageNumber)))
    for (const pageNumber of loadedSelectedPages) {
      selectedPages.value.add(pageNumber)
    }
    selectedPages.value = new Set(selectedPages.value)
  }

  function togglePageSelection(pageNumber: number): void {
    if (selectedPages.value.has(pageNumber)) {
      selectedPages.value.delete(pageNumber)
      selectedPageCount.value = Math.max(0, selectedPageCount.value - 1)
    } else {
      selectedPages.value.add(pageNumber)
      selectedPageCount.value = Math.min(
        extractResult.value?.totalPages ?? selectedPageCount.value + 1,
        selectedPageCount.value + 1,
      )
    }
    selectedPages.value = new Set(selectedPages.value)
  }

  function setAllPageSelection(selected: boolean): void {
    if (!extractResult.value?.pages) return
    if (selected) {
      selectedPages.value = new Set(extractResult.value.pages.map((p) => p.pageNumber))
      selectedPageCount.value = extractResult.value.totalPages
    } else {
      selectedPages.value = new Set()
      selectedPageCount.value = 0
    }
  }

  function toggleSelectAll(): void {
    setAllPageSelection(selectedPageCount.value !== extractResult.value?.totalPages)
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

  const settingsMethods = useWebImportSettings(draftSettings, draftProviderConfigs)

  loadDisclaimerState()
  syncDraftFromCommitted()

  return {
    settings,
    providerConfigs,
    draftSettings,
    status,
    url,
    logs,
    extractResult,
    selectedPages,
    downloadProgress,
    error,
    modalVisible,
    disclaimerVisible,
    isDownloading,
    isProcessing,
    selectedCount,
    hasUnsavedSettings,
    isSavingSettings,
    hasCredential,
    loadFromBackend,
    saveToBackend,
    initSettings,
    openModal,
    closeModal,
    resetState,
    setUrl,
    addLog,
    setExtractResult,
    setPagedExtractResult,
    appendExtractResultPages,
    togglePageSelection,
    toggleSelectAll,
    setAllPageSelection,
    setStatus,
    setError,
    updateDownloadProgress,
    acceptDisclaimer,
    rejectDisclaimer,
    discardSettingsChanges,
    saveSettings,
    ...settingsMethods
  }
})
