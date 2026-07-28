import { ref, computed, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useWebImportStore } from '@/stores/webImportStore'
import { testFirecrawlConnection, testAgentConnection } from '@/api/webImport'
import type { WebImportEngine } from '@/types/webImport'
import {
  checkWebImportSupport,
  commitWebImportDraft,
  createWebImportDraft,
  getWebImportDraft,
  listAllWebImportDraftPages,
  updateWebImportSelection,
} from '@/api/v2/webImport'
import { getTranslationBootstrap } from '@/api/v2/content'
import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import { normalizeProviderId, providerRequiresApiKey, providerSupportsCapability } from '@/config/aiProviders'
import { showToast } from '@/utils/toast'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'
import { useAiModelDiscovery } from '@/composables/useAiModelDiscovery'
import { configApi } from '@/api/config'
import type { WebImportSettingsActions } from './web-import/webImportSettingsActions'

export function useWebImportModal() {
  const webImportStore = useWebImportStore()
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
  const draftPageIdsByNumber = new Map<number, string>()
  let draftPollGeneration = 0
  const settingsActions: WebImportSettingsActions = {
    setAgentApiKey: webImportStore.setAgentApiKey,
    setAgentBaseUrl: webImportStore.setAgentBaseUrl,
    setAgentForceJsonOutput: webImportStore.setAgentForceJsonOutput,
    setAgentModelName: webImportStore.setAgentModelName,
    setAgentProvider: webImportStore.setAgentProvider,
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
  const hasAgentCredential = computed(() => webImportStore.hasCredential(
    'web_import_agent',
    draftSettings.value.agent.provider,
  ))
  const hasFirecrawlCredential = computed(() => webImportStore.hasCredential(
    'web_import_firecrawl',
    'firecrawl',
  ))
  const modelDiscovery = useAiModelDiscovery({
    source: () => ({
      provider: draftSettings.value.agent.provider,
      apiKey: draftSettings.value.agent.apiKey,
      baseUrl: draftSettings.value.agent.customBaseUrl,
      hasStoredCredential: hasAgentCredential.value,
    }),
    fetcher: (provider, apiKey, baseUrl) => configApi.fetchModels(
      provider,
      apiKey,
      baseUrl,
      'web_import_agent',
    ),
    notify: showToast,
    emptyBaseUrl: '',
  })
  const { isFetchingModels } = modelDiscovery
  const modelList = computed(() => modelDiscovery.models.value.map(model => model.id))
  const hasUnsavedSettings = computed(() => webImportStore.hasUnsavedSettings)
  const isSavingSettings = computed(() => webImportStore.isSavingSettings)
  const showAgentLogs = computed(() => draftSettings.value.ui.showAgentLogs)
  const agentProviderOptions = computed(() => [...WEB_IMPORT_AGENT_PROVIDERS])
  const supportsFetchModels = computed(() => providerSupportsCapability(draftSettings.value.agent.provider, 'modelFetch'))
  const modelListOptions = computed(() => modelList.value.map(model => ({ label: model, value: model })))

  const currentEngine = computed(() => extractResult.value?.engine || null)

  const engineDisplayName = computed(() => {
    switch (currentEngine.value) {
      case 'gallery-dl': return 'Gallery-DL'
      case 'ai-agent': return 'AI Agent'
      default: return ''
    }
  })

  const isAllSelected = computed(() => {
    if (!extractResult.value?.pages) return false
    return selectedCount.value === extractResult.value.pages.length
  })

  function getPreviewUrl(originalUrl: string): string {
    return originalUrl
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
    if (!trimmedUrl) {
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
    if (isProcessing.value) {
      const confirmed = await confirmProductAction({
        title: '关闭网页导入',
        message: '后端任务会继续运行。确定关闭此窗口吗？',
        confirmText: '关闭',
        cancelText: '继续查看',
      })
      if (!confirmed) return
    }
    draftPollGeneration += 1
    webImportStore.discardSettingsChanges()
    webImportStore.closeModal()
    webImportStore.resetState()
    urlInput.value = ''
  }

  async function handleSaveSettings(showSuccessFeedback = true): Promise<boolean> {
    const success = await webImportStore.saveSettings()
    if (showSuccessFeedback) {
      showToast(success ? '设置已保存' : '设置保存失败，请重试', success ? 'success' : 'error')
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
        showToast('设置保存失败，请重试', 'error')
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
    if (!(await ensureSettingsReady('开始提取'))) {
      return
    }

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
    webImportStore.setUrl(url)
    webImportStore.setStatus('extracting')

    try {
      const bootstrap = await getTranslationBootstrap({
        bookId: typeof route.query.book === 'string' ? route.query.book : undefined,
        chapterId: typeof route.query.chapter === 'string' ? route.query.chapter : undefined,
      })
      const accepted = await createWebImportDraft({
        chapterId: bootstrap.chapter.id,
        sourceUrl: url,
        engine: selectedEngine.value,
        config: {},
      })
      activeDraftId.value = accepted.draftId
      webImportStore.addLog({
        timestamp: new Date().toISOString(),
        type: 'info',
        message: '网页提取任务已进入后端任务中心，可安全关闭页面。',
      })
      await pollDraft(accepted.draftId)
    } catch (e) {
      webImportStore.setError(e instanceof Error ? e.message : '提取失败')
    }
  }

  function togglePage(pageNumber: number) {
    webImportStore.togglePageSelection(pageNumber)
  }

  function toggleAll() {
    webImportStore.toggleSelectAll()
  }

  async function handleImport() {
    if (!(await ensureSettingsReady('导入图片'))) {
      return
    }

    if (!extractResult.value?.pages || selectedCount.value === 0) {
      showToast('请选择要导入的图片', 'warning')
      return
    }

    webImportStore.setStatus('downloading')
    webImportStore.updateDownloadProgress(0, selectedCount.value)

    try {
      if (!activeDraftId.value) throw new Error('网页导入草稿不存在')
      const selectedIds = [...selectedPages.value]
        .map(pageNumber => draftPageIdsByNumber.get(pageNumber))
        .filter((value): value is string => Boolean(value))
      const selection = await updateWebImportSelection(
        activeDraftId.value,
        activeDraftRevision.value,
        selectedIds,
      )
      activeDraftRevision.value = selection.revision
      await commitWebImportDraft(activeDraftId.value, activeDraftRevision.value)
      webImportStore.setStatus('completed')
      showToast('入库任务已进入后端任务中心，可安全关闭页面', 'success')
      await handleClose()
    } catch (e) {
      webImportStore.setError(e instanceof Error ? e.message : '下载失败')
    }
  }

  function delay(milliseconds: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
  }

  async function pollDraft(draftId: string): Promise<void> {
    const generation = ++draftPollGeneration
    while (generation === draftPollGeneration) {
      const draft = await getWebImportDraft(draftId)
      if (generation !== draftPollGeneration) return
      activeDraftRevision.value = draft.revision
      webImportStore.updateDownloadProgress(
        draft.candidateCount - draft.failedCount,
        Math.max(draft.candidateCount, 1),
      )
      if (draft.status === 'failed') {
        webImportStore.setError('后端网页提取任务失败，请在任务中心查看详情')
        return
      }
      if (draft.status === 'committing' || draft.status === 'completed') {
        webImportStore.setStatus('completed')
        return
      }
      if (draft.status === 'ready') {
        const pages = await listAllWebImportDraftPages(draftId)
        if (generation !== draftPollGeneration) return
        draftPageIdsByNumber.clear()
        const successful = pages.filter(page => !page.error)
        successful.forEach((page, index) => {
          draftPageIdsByNumber.set(index + 1, page.id)
        })
        webImportStore.setExtractResult({
          success: true,
          comicTitle: '',
          chapterTitle: '',
          pages: successful.map((page, index) => ({
            pageNumber: index + 1,
            imageUrl: page.thumbnailUrl || page.sourceMediaUrl || '',
          })),
          totalPages: successful.length,
          sourceUrl: draft.sourceUrl,
          engine: draft.actualEngine === 'gallery-dl' ? 'gallery-dl' : 'ai-agent',
        })
        webImportStore.setStatus('extracted')
        if (webImportStore.settings.ui.autoImport && successful.length > 0) {
          await handleImport()
        }
        return
      }
      await delay(750)
    }
  }

  async function restoreActiveDraft(): Promise<void> {
    try {
      const bootstrap = await getTranslationBootstrap({
        bookId: typeof route.query.book === 'string' ? route.query.book : undefined,
        chapterId: typeof route.query.chapter === 'string' ? route.query.chapter : undefined,
      })
      const draft = bootstrap.activeWebImportDraft
      if (!draft) return
      activeDraftId.value = draft.id
      webImportStore.setStatus(draft.status === 'ready' ? 'extracting' : 'extracting')
      await pollDraft(draft.id)
    } catch {
      // Opening the modal is still useful for starting a new draft.
    }
  }

  watch(isVisible, (visible) => {
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
    }
  })

  watch(urlInput, (newUrl) => {
    checkUrlSupport(newUrl)
  })

  watch(
    () => draftSettings.value.agent.provider,
    () => {
      modelDiscovery.invalidate()
    }
  )

  const showCustomUrl = computed(() => normalizeProviderId(draftSettings.value.agent.provider) === 'custom')

  async function handleTestFirecrawl() {
    if (!draftSettings.value.firecrawl.apiKey && !hasFirecrawlCredential.value) {
      showToast('请输入 Firecrawl API Key', 'warning')
      return
    }

    testingFirecrawl.value = true
    try {
      const result = await testFirecrawlConnection(draftSettings.value.firecrawl.apiKey)
      if (result.success) {
        showToast('Firecrawl 连接成功', 'success')
      } else {
        showToast(`连接失败: ${result.message || result.error || '未知错误'}`, 'error')
      }
    } catch (e) {
      showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
    } finally {
      testingFirecrawl.value = false
    }
  }

  async function handleTestAgent() {
    if (
      providerRequiresApiKey(draftSettings.value.agent.provider)
      && !draftSettings.value.agent.apiKey
      && !hasAgentCredential.value
    ) {
      showToast('请输入 AI Agent API Key', 'warning')
      return
    }

    testingAgent.value = true
    try {
      const result = await testAgentConnection(
        draftSettings.value.agent.provider,
        draftSettings.value.agent.apiKey,
        draftSettings.value.agent.customBaseUrl,
        draftSettings.value.agent.modelName
      )
      if (result.success) {
        showToast('AI Agent 连接成功', 'success')
      } else {
        showToast(`连接失败: ${result.message || result.error || '未知错误'}`, 'error')
      }
    } catch (e) {
      showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
    } finally {
      testingAgent.value = false
    }
  }

  const handleFetchModels = modelDiscovery.fetchModels

  async function handleResetPrompt() {
    const confirmed = await confirmProductAction({
      title: '重置提取提示词',
      message: '确定要重置为默认提示词吗？',
      confirmText: '重置',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed) return
    webImportStore.resetExtractionPrompt()
  }

  onUnmounted(() => {
    draftPollGeneration += 1
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
    hasAgentCredential,
    hasFirecrawlCredential,
    isSavingSettings,
    showAgentLogs,
    agentProviderOptions,
    supportsFetchModels,
    modelListOptions,
    engineDisplayName,
    isAllSelected,
    getPreviewUrl,
    handleClose,
    handleSaveSettings,
    handleDiscardSettings,
    handleExtract,
    togglePage,
    toggleAll,
    handleImport,
    showCustomUrl,
    handleTestFirecrawl,
    handleTestAgent,
    handleFetchModels,
    handleResetPrompt,
    providerRequiresApiKey,
  }
}
