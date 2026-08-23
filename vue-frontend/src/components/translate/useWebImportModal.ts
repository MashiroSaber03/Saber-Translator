import { ref, computed, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useWebImportStore } from '@/stores/webImportStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { jobsApi, type V2JobEvent } from '@/api/v2/jobs'
import type { AgentLog, WebImportEngine, WebImportResolvedEngine } from '@/types/webImport'
import {
  checkWebImportSupport,
  commitWebImportDraft,
  createWebImportDraft,
  getWebImportDraft,
  listWebImportDraftPages,
  testAgentConnection,
  testFirecrawlConnection,
  updateWebImportSelection,
  type WebImportDraftAccepted,
} from '@/api/v2/webImport'
import { getTranslationBootstrap } from '@/api/v2/content'
import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import {
  normalizeProviderId,
  providerRequiresApiKey,
  providerRequiresApiKeyForBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'
import { useAiModelDiscovery } from '@/composables/useAiModelDiscovery'
import { fetchModels as fetchV2Models } from '@/api/v2/diagnostics'
import type { WebImportSettingsActions } from './web-import/webImportSettingsActions'

export interface WebImportModalCallbacks {
  onCommitAccepted?: (accepted: WebImportDraftAccepted) => void
}

class WebImportDraftContractError extends Error {}

export function useWebImportModal(callbacks: WebImportModalCallbacks = {}) {
  const webImportStore = useWebImportStore()
  const taskCenterStore = useTaskCenterStore()
  const route = useRoute()

  const focusSourceUrlRequestId = ref(0)
  const urlInput = ref('')
  const logsExpanded = ref(true)
  const selectedEngine = ref<WebImportEngine>('auto')
  const galleryDLAvailable = ref(false)
  const galleryDLSupported = ref(false)
  const checkingSupport = ref(false)

  const settingsExpanded = ref(false)
  const activeSettingsTab = ref<'basic' | 'preprocess' | 'advanced'>('basic')
  const testingFirecrawl = ref(false)
  const testingAgent = ref(false)
  const activeDraftId = ref<string | null>(null)
  const activeDraftRevision = ref(0)
  const isLoadingMorePages = ref(false)
  const hasMorePages = ref(false)
  const draftPageIdsByNumber = new Map<number, string>()
  const selectionOverrides = new Map<string, boolean>()
  const activeDraftJobIds = new Set<string>()
  let selectAllOverride: boolean | null = null
  let selectionDirty = false
  let nextDraftPageCursor: number | null = 0
  let loadedSuccessfulPageCount = 0
  let readyDraftLoadedId: string | null = null
  let readyDraftSnapshot: Awaited<ReturnType<typeof getWebImportDraft>> | null = null
  let draftSyncGeneration = 0
  let draftPageLoadRequestId = 0
  let draftSyncTimer: ReturnType<typeof setTimeout> | null = null
  let draftPollTimer: ReturnType<typeof setTimeout> | null = null
  let pendingDraftSync: { draftId: string; generation: number } | null = null
  let draftSyncRunner: Promise<void> | null = null
  let agentLogCursor = 0
  let agentLogJobId: string | null = null
  const seenAgentLogEventIds = new Set<number>()
  let firecrawlTestRequestId = 0
  let agentTestRequestId = 0
  let stopTaskEvents: (() => void) | null = null
  const settingsActions: WebImportSettingsActions = {
    setAgentApiKey: webImportStore.setAgentApiKey,
    setAgentBaseUrl: webImportStore.setAgentBaseUrl,
    setAgentForceJsonOutput: webImportStore.setAgentForceJsonOutput,
    setAgentMaxRetries: webImportStore.setAgentMaxRetries,
    setAgentModelName: webImportStore.setAgentModelName,
    setAgentProvider: webImportStore.setAgentProvider,
    setAgentTimeout: webImportStore.setAgentTimeout,
    setAgentUseStream: webImportStore.setAgentUseStream,
    setAutoImport: webImportStore.setAutoImport,
    setBypassProxy: webImportStore.setBypassProxy,
    setCustomCookie: webImportStore.setCustomCookie,
    setCustomHeaders: webImportStore.setCustomHeaders,
    setDownloadConcurrency: webImportStore.setDownloadConcurrency,
    setDownloadDelay: webImportStore.setDownloadDelay,
    setDownloadRetries: webImportStore.setDownloadRetries,
    setDownloadTimeout: webImportStore.setDownloadTimeout,
    setDownloadUseReferer: webImportStore.setDownloadUseReferer,
    setExtractionMaxIterations: webImportStore.setExtractionMaxIterations,
    setExtractionPrompt: webImportStore.setExtractionPrompt,
    setFirecrawlApiKey: webImportStore.setFirecrawlApiKey,
    setImageAutoRotate: webImportStore.setImageAutoRotate,
    setImageCompressionEnabled: webImportStore.setImageCompressionEnabled,
    setImageCompressionQuality: webImportStore.setImageCompressionQuality,
    setImageFormatConvertEnabled: webImportStore.setImageFormatConvertEnabled,
    setImageMaxHeight: webImportStore.setImageMaxHeight,
    setImageMaxWidth: webImportStore.setImageMaxWidth,
    setImagePreprocessEnabled: webImportStore.setImagePreprocessEnabled,
    setImageTargetFormat: webImportStore.setImageTargetFormat,
    setShowAgentLogs: webImportStore.setShowAgentLogs,
  }

  const isVisible = computed(() => webImportStore.modalVisible)
  const status = computed(() => webImportStore.status)
  const logs = computed(() => webImportStore.logs)
  const extractResult = computed(() => webImportStore.extractResult)
  const selectedPages = computed(() => webImportStore.selectedPages)
  const selectedCount = computed(() => webImportStore.selectedCount)
  const downloadProgress = computed(() => webImportStore.downloadProgress)
  const error = computed(() => webImportStore.error)
  const isProcessing = computed(() => webImportStore.isProcessing)
  const draftSettings = computed(() => webImportStore.draftSettings)
  const modelDiscovery = useAiModelDiscovery({
    source: () => ({
      provider: draftSettings.value.agent.provider,
      apiKey: draftSettings.value.agent.apiKey,
      baseUrl: draftSettings.value.agent.customBaseUrl,
    }),
    fetcher: (provider, apiKey, baseUrl) =>
      fetchV2Models(provider, apiKey, baseUrl, 'web_import_agent'),
    notify: showToast,
    emptyBaseUrl: '',
  })
  const { isFetchingModels } = modelDiscovery
  const modelList = computed(() => modelDiscovery.models.value.map(model => model.id))
  const hasUnsavedSettings = computed(() => webImportStore.hasUnsavedSettings)
  const isSavingSettings = computed(() => webImportStore.isSavingSettings)
  const showAgentLogs = computed(() => draftSettings.value.ui.showAgentLogs)
  const agentProviderOptions = computed(() => [...WEB_IMPORT_AGENT_PROVIDERS])
  const supportsFetchModels = computed(() =>
    providerSupportsCapability(draftSettings.value.agent.provider, 'modelFetch')
  )
  const modelListOptions = computed(() =>
    modelList.value.map(model => ({ label: model, value: model }))
  )

  const engineDisplayName = computed(() => {
    switch (extractResult.value?.engine) {
      case 'gallery-dl':
        return 'Gallery-DL'
      case 'ai-agent':
        return 'AI Agent'
      default:
        return ''
    }
  })

  const isAllSelected = computed(() => {
    if (!extractResult.value || extractResult.value.totalPages === 0) return false
    return selectedCount.value === extractResult.value.totalPages
  })

  function resetDraftPaging(): void {
    draftPageLoadRequestId += 1
    draftPageIdsByNumber.clear()
    selectionOverrides.clear()
    selectAllOverride = null
    selectionDirty = false
    nextDraftPageCursor = 0
    loadedSuccessfulPageCount = 0
    readyDraftLoadedId = null
    readyDraftSnapshot = null
    hasMorePages.value = false
    isLoadingMorePages.value = false
  }

  function resetAgentLogState(): void {
    agentLogCursor = 0
    agentLogJobId = null
    seenAgentLogEventIds.clear()
  }

  function clearDraftPoll(): void {
    if (!draftPollTimer) return
    clearTimeout(draftPollTimer)
    draftPollTimer = null
  }

  function invalidateModalSession(): void {
    draftSyncGeneration += 1
    firecrawlTestRequestId += 1
    agentTestRequestId += 1
    testingFirecrawl.value = false
    testingAgent.value = false
    clearDraftPoll()
  }

  let checkSupportTimeout: ReturnType<typeof setTimeout> | null = null
  let focusInputTimeout: ReturnType<typeof setTimeout> | null = null
  const urlSupportGuard = useLatestRequestGuard()

  function isCurrentUrlSupport(requestId: number, url: string): boolean {
    return urlSupportGuard.isCurrent(requestId, () => urlInput.value.trim() === url)
  }

  async function checkUrlSupport(url: string) {
    urlSupportGuard.invalidate()
    if (checkSupportTimeout) {
      clearTimeout(checkSupportTimeout)
      checkSupportTimeout = null
    }

    const trimmedUrl = url.trim()
    if (!trimmedUrl || selectedEngine.value === 'ai-agent') {
      galleryDLAvailable.value = false
      galleryDLSupported.value = false
      checkingSupport.value = false
      return
    }

    const requestId = urlSupportGuard.next()
    checkSupportTimeout = setTimeout(async () => {
      checkSupportTimeout = null
      checkingSupport.value = true
      try {
        const result = await checkWebImportSupport(trimmedUrl)
        if (!isCurrentUrlSupport(requestId, trimmedUrl)) return
        galleryDLAvailable.value = result.galleryDlAvailable
        galleryDLSupported.value = result.galleryDlSupported
      } catch {
        if (!isCurrentUrlSupport(requestId, trimmedUrl)) return
        galleryDLAvailable.value = false
        galleryDLSupported.value = false
      } finally {
        if (isCurrentUrlSupport(requestId, trimmedUrl)) {
          checkingSupport.value = false
        }
      }
    }, 500)
  }

  async function handleClose() {
    const acceptedExtractContinuesInBackend =
      status.value === 'extracting' && activeDraftId.value !== null
    if (isProcessing.value && !acceptedExtractContinuesInBackend) {
      const confirmed = await confirmProductAction({
        title: '关闭网页导入',
        message: '后端任务会继续运行。确定关闭此窗口吗？',
        confirmText: '关闭',
        cancelText: '继续查看',
      })
      if (!confirmed) return
    }
    invalidateModalSession()
    if (draftSyncTimer) {
      clearTimeout(draftSyncTimer)
      draftSyncTimer = null
    }
    webImportStore.discardSettingsChanges()
    webImportStore.closeModal()
    webImportStore.resetState()
    urlInput.value = ''
    activeDraftId.value = null
    activeDraftRevision.value = 0
    resetDraftPaging()
    activeDraftJobIds.clear()
    resetAgentLogState()
  }

  async function handleSaveSettings(showSuccessFeedback = true): Promise<boolean> {
    const success = await webImportStore.saveSettings()
    if (showSuccessFeedback) {
      showToast(
        success ? '设置已保存' : webImportStore.settingsSaveError || '设置保存失败，请重试',
        success ? 'success' : 'error'
      )
    }
    return success
  }

  function handleDiscardSettings() {
    if (!hasUnsavedSettings.value) return
    webImportStore.discardSettingsChanges()
  }

  async function ensureSettingsReady(actionLabel: string): Promise<boolean> {
    if (!hasUnsavedSettings.value) {
      return true
    }

    const shouldSave = await confirmProductAction({
      title: '保存网页导入设置',
      message: `网页导入设置有未保存修改。要先保存设置后再${actionLabel}吗？`,
      confirmText: '保存并继续',
      cancelText: '不保存',
    })

    if (shouldSave) {
      const success = await handleSaveSettings(false)
      if (!success) {
        showToast(webImportStore.settingsSaveError || '设置保存失败，请重试', 'error')
      }
      return success
    }

    const shouldDiscard = await confirmProductAction({
      title: '放弃网页导入设置修改',
      message: `要继续${actionLabel}并放弃未保存修改吗？`,
      confirmText: '放弃修改',
      cancelText: '返回设置',
      tone: 'danger',
    })
    if (!shouldDiscard) {
      return false
    }

    webImportStore.discardSettingsChanges()
    return true
  }

  async function handleExtract() {
    const settingsGeneration = draftSyncGeneration
    if (!(await ensureSettingsReady('开始提取'))) {
      return
    }
    if (settingsGeneration !== draftSyncGeneration) return

    const url = urlInput.value.trim()
    if (!url) {
      showToast('请输入网址', 'warning')
      return
    }

    try {
      new URL(url)
    } catch {
      showToast('请输入有效的网址', 'warning')
      return
    }

    webImportStore.resetState()
    resetDraftPaging()
    activeDraftId.value = null
    const generation = ++draftSyncGeneration
    webImportStore.setUrl(url)
    webImportStore.setStatus('extracting')

    try {
      const bootstrap = await getTranslationBootstrap({
        bookId: typeof route.query.book === 'string' ? route.query.book : undefined,
        chapterId: typeof route.query.chapter === 'string' ? route.query.chapter : undefined,
      })
      if (generation !== draftSyncGeneration) return
      const accepted = await createWebImportDraft({
        chapterId: bootstrap.chapter.id,
        sourceUrl: url,
        engine: selectedEngine.value,
      })
      if (generation !== draftSyncGeneration) return
      activeDraftId.value = accepted.draftId
      activeDraftJobIds.clear()
      resetAgentLogState()
      webImportStore.addLog({
        timestamp: new Date().toISOString(),
        type: 'info',
        message: '网页提取任务已进入后端任务中心，可安全关闭页面。',
      })
      await syncDraft(accepted.draftId, generation)
    } catch (e) {
      if (generation === draftSyncGeneration) {
        if (e instanceof WebImportDraftContractError) {
          webImportStore.setError(e.message)
        } else if (activeDraftId.value) {
          webImportStore.setStatus('extracting')
          scheduleDraftPoll(activeDraftId.value, generation)
        } else {
          webImportStore.setError(e instanceof Error ? e.message : '提取失败')
        }
      }
    }
  }

  function togglePage(pageNumber: number) {
    const pageId = draftPageIdsByNumber.get(pageNumber)
    if (!pageId) return
    const selected = !selectedPages.value.has(pageNumber)
    webImportStore.togglePageSelection(pageNumber)
    selectionOverrides.set(pageId, selected)
    selectionDirty = true
  }

  function toggleAll() {
    const selected = !isAllSelected.value
    selectAllOverride = selected
    selectionOverrides.clear()
    selectionDirty = true
    webImportStore.setAllPageSelection(selected)
  }

  async function collectSelectedDraftPageIds(
    draftId: string,
    generation: number
  ): Promise<string[]> {
    const selectedIds: string[] = []
    let cursor = 0
    do {
      const page = await listWebImportDraftPages(draftId, { cursor, limit: 200 })
      if (!isCurrentDraft(draftId, generation)) {
        throw new Error('网页导入草稿已切换')
      }
      for (const candidate of page.items) {
        if (candidate.error) continue
        const selected =
          selectionOverrides.get(candidate.id) ?? selectAllOverride ?? candidate.selected
        if (selected) selectedIds.push(candidate.id)
      }
      cursor = page.nextCursor ?? 0
    } while (cursor > 0)
    return selectedIds
  }

  async function handleImport() {
    const settingsGeneration = draftSyncGeneration
    if (!(await ensureSettingsReady('导入图片'))) {
      return
    }
    if (settingsGeneration !== draftSyncGeneration) return

    if (!extractResult.value?.pages || selectedCount.value === 0) {
      showToast('请选择要导入的图片', 'warning')
      return
    }

    webImportStore.setStatus('downloading')
    webImportStore.updateDownloadProgress(0, selectedCount.value)

    const draftId = activeDraftId.value
    const generation = draftSyncGeneration
    let revision = activeDraftRevision.value
    try {
      if (!draftId) throw new Error('网页导入草稿不存在')
      if (selectionDirty) {
        const selectedIds = await collectSelectedDraftPageIds(draftId, generation)
        if (!isCurrentDraft(draftId, generation)) return
        const selection = await updateWebImportSelection(draftId, revision, selectedIds)
        if (!isCurrentDraft(draftId, generation)) return
        revision = selection.revision
        activeDraftRevision.value = revision
      }
      const accepted = await commitWebImportDraft(draftId, revision)
      if (!accepted.jobIds.length) {
        throw new Error('后端没有返回网页导入任务')
      }
      callbacks.onCommitAccepted?.(accepted)
      showToast('入库任务已进入后端任务中心，可安全关闭页面', 'success')
      if (isCurrentDraft(draftId, generation)) {
        webImportStore.setStatus('completed')
        await handleClose()
      }
    } catch (e) {
      if (isCurrentDraft(draftId ?? '', generation)) {
        try {
          await syncDraft(draftId ?? '', generation)
        } catch {
          // The original command error is more useful than a follow-up read error.
        }
        if (!['downloading', 'completed'].includes(status.value)) {
          webImportStore.setError(e instanceof Error ? e.message : '下载失败')
        }
      }
    }
  }

  async function loadAgentLogs(
    draft: Awaited<ReturnType<typeof getWebImportDraft>>,
    generation: number
  ): Promise<void> {
    if (draft.requestedEngine !== 'ai-agent' && draft.actualEngine !== 'ai-agent') return
    const extractJob = draft.jobs.find(job => job.kind === 'web_extract')
    if (!extractJob) return
    if (agentLogJobId !== extractJob.id) {
      agentLogJobId = extractJob.id
      agentLogCursor = 0
      seenAgentLogEventIds.clear()
    }
    const jobId = extractJob.id
    try {
      let shouldContinue = true
      while (shouldContinue) {
        const previousCursor = agentLogCursor
        const response = await jobsApi.events(jobId, {
          after: previousCursor,
          limit: 200,
        })
        if (!isCurrentDraft(draft.id, generation) || agentLogJobId !== jobId) return
        for (const event of response.items) {
          agentLogCursor = Math.max(agentLogCursor, event.eventId)
          appendAgentLogEvent(event)
        }
        shouldContinue = response.items.length === 200 && agentLogCursor > previousCursor
      }
    } catch {
      // Log retrieval must not interrupt the durable import task.
    }
  }

  function isCurrentDraft(draftId: string, generation: number): boolean {
    return generation === draftSyncGeneration && activeDraftId.value === draftId
  }

  function requireResolvedEngine(value: string | null): WebImportResolvedEngine {
    if (value === 'gallery-dl' || value === 'ai-agent') return value
    throw new WebImportDraftContractError('网页导入草稿缺少有效的实际提取引擎')
  }

  async function loadNextDraftPageBatch(
    draft: Awaited<ReturnType<typeof getWebImportDraft>>,
    generation: number,
    reset = false
  ): Promise<void> {
    const resolvedEngine = requireResolvedEngine(draft.actualEngine)
    if (isLoadingMorePages.value) return
    const cursor = reset ? 0 : nextDraftPageCursor
    if (cursor === null) return
    const requestId = ++draftPageLoadRequestId
    isLoadingMorePages.value = true
    try {
      const response = await listWebImportDraftPages(draft.id, {
        cursor,
        limit: 100,
      })
      if (!isCurrentDraft(draft.id, generation) || requestId !== draftPageLoadRequestId) return
      if (reset) {
        draftPageIdsByNumber.clear()
        loadedSuccessfulPageCount = 0
      }
      const pages: NonNullable<typeof extractResult.value>['pages'] = []
      const loadedSelected: number[] = []
      for (const candidate of response.items) {
        if (candidate.error) continue
        const pageNumber = ++loadedSuccessfulPageCount
        draftPageIdsByNumber.set(pageNumber, candidate.id)
        const selected =
          selectionOverrides.get(candidate.id) ?? selectAllOverride ?? candidate.selected
        if (selected) loadedSelected.push(pageNumber)
        pages.push({
          pageNumber,
          imageUrl: candidate.thumbnailUrl || candidate.sourceMediaUrl || '',
        })
      }
      const totalPages = Math.max(0, draft.candidateCount - draft.failedCount)
      if (reset) {
        webImportStore.setPagedExtractResult(
          {
            pages,
            totalPages,
            engine: resolvedEngine,
          },
          loadedSelected,
          draft.selectedCount
        )
      } else {
        webImportStore.appendExtractResultPages(pages, loadedSelected)
      }
      nextDraftPageCursor = response.nextCursor ?? null
      hasMorePages.value = nextDraftPageCursor !== null
      readyDraftLoadedId = draft.id
      readyDraftSnapshot = draft
    } finally {
      if (requestId === draftPageLoadRequestId) {
        isLoadingMorePages.value = false
      }
    }
  }

  async function loadMoreDraftPages(): Promise<void> {
    const draft = readyDraftSnapshot
    if (!draft || readyDraftLoadedId !== activeDraftId.value) return
    const generation = draftSyncGeneration
    try {
      await loadNextDraftPageBatch(draft, generation)
    } catch (error) {
      if (isCurrentDraft(draft.id, generation)) {
        showToast(error instanceof Error ? error.message : '加载更多图片失败', 'error')
      }
    }
  }

  async function syncDraftOnce(draftId: string, generation: number): Promise<void> {
    const draft = await getWebImportDraft(draftId)
    if (!isCurrentDraft(draftId, generation)) return
    activeDraftJobIds.clear()
    for (const job of draft.jobs) activeDraftJobIds.add(job.id)
    await loadAgentLogs(draft, generation)
    if (!isCurrentDraft(draftId, generation)) return
    activeDraftRevision.value = draft.revision
    webImportStore.updateDownloadProgress(
      draft.candidateCount - draft.failedCount,
      Math.max(draft.candidateCount, 1)
    )
    if (draft.status === 'failed') {
      clearDraftPoll()
      webImportStore.setError('后端网页提取任务失败，请在任务中心查看详情')
      return
    }
    if (draft.status === 'cancelled') {
      clearDraftPoll()
      webImportStore.setError('后端网页提取任务已取消，可以重新开始提取')
      return
    }
    if (draft.status === 'committing') {
      webImportStore.setStatus('downloading')
      scheduleDraftPoll(draftId, generation)
      return
    }
    if (draft.status === 'completed') {
      clearDraftPoll()
      webImportStore.setStatus('completed')
      return
    }
    if (draft.status !== 'ready') {
      scheduleDraftPoll(draftId, generation)
      return
    }

    if (readyDraftLoadedId !== draftId) {
      resetDraftPaging()
      await loadNextDraftPageBatch(draft, generation, true)
      if (!isCurrentDraft(draftId, generation)) return
    } else {
      readyDraftSnapshot = draft
    }
    if (!draft.autoImport) {
      clearDraftPoll()
      webImportStore.setStatus('extracted')
      return
    }
    const extractJob = draft.jobs.find(job => job.kind === 'web_extract')
    if (
      extractJob &&
      ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(extractJob.status)
    ) {
      webImportStore.setError('后端自动入库未能启动，请在任务中心查看详情')
      return
    }
    scheduleDraftPoll(draftId, generation)
  }

  function isAgentLogPayload(payload: unknown): payload is AgentLog {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return false
    const candidate = payload as Partial<AgentLog>
    const validTypes: AgentLog['type'][] = ['info', 'tool_call', 'tool_result', 'thinking', 'error']
    return (
      typeof candidate.timestamp === 'string' &&
      typeof candidate.type === 'string' &&
      validTypes.some(type => type === candidate.type) &&
      typeof candidate.message === 'string'
    )
  }

  function appendAgentLogEvent(event: V2JobEvent): void {
    if (
      event.type !== 'web_import_agent_log' ||
      seenAgentLogEventIds.has(event.eventId) ||
      !isAgentLogPayload(event.payload)
    )
      return
    seenAgentLogEventIds.add(event.eventId)
    webImportStore.addLog(event.payload)
  }

  function addAgentLog(event: V2JobEvent): void {
    if (event.type !== 'web_import_agent_log') return
    if (agentLogJobId !== event.jobId) {
      agentLogJobId = event.jobId
      agentLogCursor = 0
      seenAgentLogEventIds.clear()
    }
    appendAgentLogEvent(event)
  }

  async function runQueuedDraftSyncs(): Promise<void> {
    let latestError: unknown = null
    while (pendingDraftSync) {
      const request = pendingDraftSync
      pendingDraftSync = null
      try {
        await syncDraftOnce(request.draftId, request.generation)
        latestError = null
      } catch (error) {
        latestError = error
      }
    }
    if (latestError) throw latestError
  }

  function syncDraft(draftId: string, generation: number): Promise<void> {
    pendingDraftSync = { draftId, generation }
    if (!draftSyncRunner) {
      draftSyncRunner = (async () => {
        try {
          await runQueuedDraftSyncs()
        } finally {
          draftSyncRunner = null
        }
      })()
    }
    return draftSyncRunner
  }

  function scheduleDraftSync(): void {
    const draftId = activeDraftId.value
    if (!draftId || draftSyncTimer) return
    const generation = draftSyncGeneration
    draftSyncTimer = setTimeout(() => {
      draftSyncTimer = null
      void syncDraft(draftId, generation).catch(error => {
        handleDraftSyncFailure(error, draftId, generation)
      })
    }, 100)
  }

  function scheduleDraftPoll(draftId: string, generation: number): void {
    if (draftPollTimer || !isVisible.value || !isCurrentDraft(draftId, generation)) return
    draftPollTimer = setTimeout(() => {
      draftPollTimer = null
      void syncDraft(draftId, generation).catch(error => {
        handleDraftSyncFailure(error, draftId, generation)
      })
    }, 1000)
  }

  function handleDraftSyncFailure(error: unknown, draftId: string, generation: number): void {
    if (!isCurrentDraft(draftId, generation)) return
    if (error instanceof WebImportDraftContractError) {
      clearDraftPoll()
      webImportStore.setError(error.message)
      return
    }
    scheduleDraftPoll(draftId, generation)
  }

  function handleTaskEvent(event: V2JobEvent): void {
    if (!activeDraftJobIds.has(event.jobId)) return
    addAgentLog(event)
    scheduleDraftSync()
  }

  async function restoreActiveDraft(): Promise<void> {
    const generation = ++draftSyncGeneration
    try {
      const bootstrap = await getTranslationBootstrap({
        bookId: typeof route.query.book === 'string' ? route.query.book : undefined,
        chapterId: typeof route.query.chapter === 'string' ? route.query.chapter : undefined,
      })
      if (generation !== draftSyncGeneration || !isVisible.value) return
      const draft = bootstrap.activeWebImportDraft
      if (!draft) return
      resetDraftPaging()
      activeDraftId.value = draft.id
      activeDraftJobIds.clear()
      resetAgentLogState()
      webImportStore.setStatus('extracting')
      await syncDraft(draft.id, generation)
    } catch (error) {
      if (activeDraftId.value && isCurrentDraft(activeDraftId.value, generation)) {
        handleDraftSyncFailure(error, activeDraftId.value, generation)
      }
      // Opening the modal is still useful for starting a new draft.
    }
  }

  watch(isVisible, visible => {
    if (focusInputTimeout) {
      clearTimeout(focusInputTimeout)
      focusInputTimeout = null
    }
    if (visible) {
      focusInputTimeout = setTimeout(() => {
        focusInputTimeout = null
        focusSourceUrlRequestId.value += 1
      }, 100)
      void restoreActiveDraft()
    } else {
      invalidateModalSession()
    }
  })

  stopTaskEvents = taskCenterStore.subscribeEvents(handleTaskEvent)

  watch([urlInput, selectedEngine], ([newUrl]) => {
    void checkUrlSupport(newUrl)
  })

  watch(
    () => draftSettings.value.agent.provider,
    () => {
      modelDiscovery.invalidate()
    }
  )

  const showCustomUrl = computed(
    () => normalizeProviderId(draftSettings.value.agent.provider) === 'custom'
  )

  async function handleTestFirecrawl() {
    if (!draftSettings.value.firecrawl.apiKey) {
      showToast('请输入 Firecrawl API Key', 'warning')
      return
    }

    const apiKey = draftSettings.value.firecrawl.apiKey
    const requestId = ++firecrawlTestRequestId
    testingFirecrawl.value = true
    try {
      const result = await testFirecrawlConnection(apiKey)
      if (requestId !== firecrawlTestRequestId || draftSettings.value.firecrawl.apiKey !== apiKey)
        return
      if (result.success) {
        showToast('Firecrawl 连接成功', 'success')
      } else {
        showToast(`连接失败: ${result.message || '未知错误'}`, 'error')
      }
    } catch (e) {
      if (requestId === firecrawlTestRequestId && draftSettings.value.firecrawl.apiKey === apiKey) {
        showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
      }
    } finally {
      if (requestId === firecrawlTestRequestId) {
        testingFirecrawl.value = false
      }
    }
  }

  async function handleTestAgent() {
    if (
      providerRequiresApiKeyForBaseUrl(
        draftSettings.value.agent.provider,
        draftSettings.value.agent.customBaseUrl,
      ) &&
      !draftSettings.value.agent.apiKey
    ) {
      showToast('请输入 AI Agent API Key', 'warning')
      return
    }

    const request = {
      provider: draftSettings.value.agent.provider,
      apiKey: draftSettings.value.agent.apiKey,
      baseUrl: draftSettings.value.agent.customBaseUrl,
      modelName: draftSettings.value.agent.modelName,
    }
    const requestId = ++agentTestRequestId
    const isCurrentRequest = () =>
      requestId === agentTestRequestId &&
      request.provider === draftSettings.value.agent.provider &&
      request.apiKey === draftSettings.value.agent.apiKey &&
      request.baseUrl === draftSettings.value.agent.customBaseUrl &&
      request.modelName === draftSettings.value.agent.modelName
    testingAgent.value = true
    try {
      const result = await testAgentConnection(
        request.provider,
        request.apiKey,
        request.baseUrl,
        request.modelName
      )
      if (!isCurrentRequest()) return
      if (result.success) {
        showToast('AI Agent 连接成功', 'success')
      } else {
        showToast(`连接失败: ${result.message || '未知错误'}`, 'error')
      }
    } catch (e) {
      if (isCurrentRequest()) {
        showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
      }
    } finally {
      if (requestId === agentTestRequestId) {
        testingAgent.value = false
      }
    }
  }

  const handleFetchModels = modelDiscovery.fetchModels

  async function handleResetPrompt() {
    const generation = draftSyncGeneration
    const confirmed = await confirmProductAction({
      title: '重置提取提示词',
      message: '确定要重置为默认提示词吗？',
      confirmText: '重置',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed || generation !== draftSyncGeneration) return
    webImportStore.resetExtractionPrompt()
  }

  onUnmounted(() => {
    invalidateModalSession()
    draftPageLoadRequestId += 1
    pendingDraftSync = null
    stopTaskEvents?.()
    stopTaskEvents = null
    if (draftSyncTimer) {
      clearTimeout(draftSyncTimer)
      draftSyncTimer = null
    }
    if (checkSupportTimeout) {
      clearTimeout(checkSupportTimeout)
      checkSupportTimeout = null
    }
    urlSupportGuard.invalidate()
    checkingSupport.value = false
    modelDiscovery.invalidate()
    if (focusInputTimeout) {
      clearTimeout(focusInputTimeout)
      focusInputTimeout = null
    }
  })

  return {
    focusSourceUrlRequestId,
    urlInput,
    logsExpanded,
    selectedEngine,
    galleryDLAvailable,
    galleryDLSupported,
    checkingSupport,
    settingsExpanded,
    settingsActions,
    activeSettingsTab,
    testingFirecrawl,
    testingAgent,
    isFetchingModels,
    isLoadingMorePages,
    hasMorePages,
    modelList,
    isVisible,
    status,
    logs,
    extractResult,
    selectedPages,
    selectedCount,
    downloadProgress,
    error,
    isProcessing,
    draftSettings,
    hasUnsavedSettings,
    isSavingSettings,
    showAgentLogs,
    agentProviderOptions,
    supportsFetchModels,
    modelListOptions,
    engineDisplayName,
    isAllSelected,
    handleClose,
    handleSaveSettings,
    handleDiscardSettings,
    handleExtract,
    togglePage,
    toggleAll,
    handleImport,
    loadMoreDraftPages,
    showCustomUrl,
    handleTestFirecrawl,
    handleTestAgent,
    handleFetchModels,
    handleResetPrompt,
    providerRequiresApiKey,
  }
}
