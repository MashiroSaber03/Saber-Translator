import { ref, computed, onUnmounted, watch } from 'vue'
import { configApi } from '@/api/config'
import { useWebImportStore } from '@/stores/webImportStore'
import { useImageStore } from '@/stores/imageStore'
import { extractImages, downloadImages, checkGalleryDLSupport, getGalleryDLImages, testFirecrawlConnection, testAgentConnection } from '@/api/webImport'
import type { AgentLog, ExtractResult, WebImportEngine } from '@/types/webImport'
import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import { getProviderDisplayName, normalizeProviderId, providerRequiresApiKey, providerRequiresBaseUrl, providerSupportsCapability } from '@/config/aiProviders'
import { showToast } from '@/utils/toast'

export function useWebImportModal() {
  const webImportStore = useWebImportStore()
  const imageStore = useImageStore()

  // 本地状态
  const urlInput = ref('')
  const logsExpanded = ref(true)
  const selectedEngine = ref<WebImportEngine>('auto')
  const galleryDLAvailable = ref(false)
  const galleryDLSupported = ref(false)
  const checkingSupport = ref(false)

  // 设置相关状态
  const settingsExpanded = ref(false)
  const activeSettingsTab = ref<'basic' | 'preprocess' | 'advanced'>('basic')
  const testingFirecrawl = ref(false)
  const testingAgent = ref(false)
  const isFetchingModels = ref(false)
  const showFirecrawlKey = ref(false)
  const showAgentKey = ref(false)
  const modelList = ref<string[]>([])

  // 计算属性
  const isVisible = computed(() => webImportStore.modalVisible)
  const status = computed(() => webImportStore.status)
  const logs = computed(() => webImportStore.logs)
  const extractResult = computed(() => webImportStore.extractResult)
  const selectedPages = computed(() => webImportStore.selectedPages)
  const selectedCount = computed(() => webImportStore.selectedCount)
  const downloadProgress = computed(() => webImportStore.downloadProgress)
  const downloadProgressPercent = computed(() => webImportStore.downloadProgressPercent)
  const error = computed(() => webImportStore.error)
  const isProcessing = computed(() => webImportStore.isProcessing)
  const draftSettings = computed(() => webImportStore.draftSettings)
  const hasUnsavedSettings = computed(() => webImportStore.hasUnsavedSettings)
  const isSavingSettings = computed(() => webImportStore.isSavingSettings)
  const showAgentLogs = computed(() => draftSettings.value.ui.showAgentLogs)
  const agentProviderOptions = computed(() => [...WEB_IMPORT_AGENT_PROVIDERS])
  const supportsFetchModels = computed(() => providerSupportsCapability(draftSettings.value.agent.provider, 'modelFetch'))
  const modelListOptions = computed(() => modelList.value.map(model => ({ label: model, value: model })))

  // 当前使用的引擎
  const currentEngine = computed(() => extractResult.value?.engine || null)

  // 引擎显示名称
  const engineDisplayName = computed(() => {
    switch (currentEngine.value) {
      case 'gallery-dl': return 'Gallery-DL'
      case 'ai-agent': return 'AI Agent'
      default: return ''
    }
  })

  // 是否全选
  const isAllSelected = computed(() => {
    if (!extractResult.value?.pages) return false
    return selectedCount.value === extractResult.value.pages.length
  })

  // 获取预览图 URL（gallery-dl 引擎直接使用静态文件服务）
  function getPreviewUrl(originalUrl: string): string {
    // gallery-dl 引擎的图片已在本地，直接使用静态服务路径
    if (currentEngine.value === 'gallery-dl') {
      // imageUrl 格式: /api/web-import/static/temp/gallery_dl/xxx.webp
      // 直接返回，不需要代理
      return originalUrl
    }
    return originalUrl
  }

  // 检查 URL 支持（防抖）
  let checkSupportTimeout: ReturnType<typeof setTimeout> | null = null
  let focusInputTimeout: ReturnType<typeof setTimeout> | null = null
  async function checkUrlSupport(url: string) {
    if (checkSupportTimeout) {
      clearTimeout(checkSupportTimeout)
      checkSupportTimeout = null
    }
    
    if (!url.trim()) {
      galleryDLAvailable.value = false
      galleryDLSupported.value = false
      return
    }
    
    checkSupportTimeout = setTimeout(async () => {
      checkSupportTimeout = null
      checkingSupport.value = true
      try {
        const result = await checkGalleryDLSupport(url)
        galleryDLAvailable.value = result.available
        galleryDLSupported.value = result.supported
      } catch {
        galleryDLAvailable.value = false
        galleryDLSupported.value = false
      } finally {
        checkingSupport.value = false
      }
    }, 500)
  }

  // 关闭模态框
  function handleClose() {
    if (isProcessing.value) {
      if (!confirm('正在处理中，确定要关闭吗？')) return
    }
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

    const shouldSave = confirm(
      `网页导入设置有未保存修改。\n点击“确定”先保存设置后再${actionLabel}；点击“取消”则进入放弃修改确认。`
    )

    if (shouldSave) {
      const success = await handleSaveSettings(false)
      if (!success) {
        showToast('设置保存失败，请重试', 'error')
      }
      return success
    }

    const shouldDiscard = confirm(`要继续${actionLabel}并放弃未保存修改吗？`)
    if (!shouldDiscard) {
      return false
    }

    webImportStore.discardSettingsChanges()
    return true
  }

  // 开始提取
  async function handleExtract() {
    if (!(await ensureSettingsReady('开始提取'))) {
      return
    }

    const url = urlInput.value.trim()
    if (!url) {
      showToast('请输入网址', 'warning')
      return
    }

    // 验证 URL
    try {
      new URL(url)
    } catch {
      showToast('请输入有效的网址', 'warning')
      return
    }

    // 重置状态
    webImportStore.resetState()
    webImportStore.setUrl(url)
    webImportStore.setStatus('extracting')

    try {
      await extractImages(
        url,
        webImportStore.settings,
        (log: AgentLog) => {
          webImportStore.addLog(log)
        },
        (result: ExtractResult) => {
          webImportStore.setExtractResult(result)
          if (result.success) {
            webImportStore.setStatus('extracted')
          } else {
            webImportStore.setError(result.error || '提取失败')
          }
        },
        (errorMsg: string) => {
          webImportStore.setError(errorMsg)
        },
        selectedEngine.value,
        // 每收到一张图片就增量添加。
        (page) => {
          webImportStore.addPageIncremental(page)
        }
      )

      if (
        webImportStore.settings.ui.autoImport &&
        webImportStore.status === 'extracted' &&
        webImportStore.extractResult?.success
      ) {
        await handleImport()
      }
    } catch (e) {
      webImportStore.setError(e instanceof Error ? e.message : '提取失败')
    }
  }

  // 切换页面选择
  function togglePage(pageNumber: number) {
    webImportStore.togglePageSelection(pageNumber)
  }

  // 全选/取消全选
  function toggleAll() {
    webImportStore.toggleSelectAll()
  }

  // 开始下载并导入
  async function handleImport() {
    if (!(await ensureSettingsReady('导入图片'))) {
      return
    }

    if (!extractResult.value?.pages || selectedCount.value === 0) {
      showToast('请选择要导入的图片', 'warning')
      return
    }

    // 获取选中的页面
    const selectedPagesList = extractResult.value.pages.filter((p) =>
      selectedPages.value.has(p.pageNumber)
    )

    webImportStore.setStatus('downloading')
    webImportStore.updateDownloadProgress(0, selectedPagesList.length)

    // 使用提取时使用的引擎
    const engineToUse = currentEngine.value || 'ai-agent'

    try {
      // gallery-dl 引擎：图片已下载到临时目录，直接获取
      if (engineToUse === 'gallery-dl') {
        const galleryResult = await getGalleryDLImages()
        
        if (galleryResult.success && galleryResult.images.length > 0) {
          let importedCount = 0
          let processedCount = 0

          for (const page of selectedPagesList) {
            const img = galleryResult.images[page.pageNumber - 1]
            processedCount++
            if (img && img.filename && img.data) {
              imageStore.addImage(img.filename, img.data)
              importedCount++
            }
            webImportStore.updateDownloadProgress(processedCount, selectedPagesList.length)
          }

          if (importedCount === 0) {
            throw new Error('未能导入选中的图片')
          }

          webImportStore.setStatus('completed')
          showToast(`成功导入 ${importedCount} 张图片`, 'success')
          handleClose()
          return
        } else {
          throw new Error(galleryResult.error || '获取图片失败')
        }
      }
      
      // AI Agent 引擎：调用下载接口
      const result = await downloadImages(
        selectedPagesList,
        extractResult.value.sourceUrl,
        webImportStore.settings,
        engineToUse
      )

      if (result.success && result.images.length > 0) {
        webImportStore.setDownloadedImages(result.images)
        webImportStore.updateDownloadProgress(result.images.length, selectedPagesList.length)

        // 导入到 imageStore (参数顺序: fileName, dataUrl)
        for (const img of result.images) {
          imageStore.addImage(img.filename, img.dataUrl)
        }

        webImportStore.setStatus('completed')

        // 提示成功
        const failedMsg = result.failedCount > 0 ? `，${result.failedCount} 张失败` : ''
        showToast(`成功导入 ${result.images.length} 张图片${failedMsg}`, result.failedCount > 0 ? 'warning' : 'success')

        // 关闭模态框
        handleClose()
      } else {
        webImportStore.setError(result.error || '下载失败')
      }
    } catch (e) {
      webImportStore.setError(e instanceof Error ? e.message : '下载失败')
    }
  }

  // 监听模态框打开时聚焦输入框
  watch(isVisible, (visible) => {
    if (focusInputTimeout) {
      clearTimeout(focusInputTimeout)
      focusInputTimeout = null
    }
    if (visible) {
      focusInputTimeout = setTimeout(() => {
        focusInputTimeout = null
        const input = document.querySelector('.url-input') as HTMLInputElement
        input?.focus()
      }, 100)
    }
  })

  // 监听 URL 输入变化，检查 gallery-dl 支持
  watch(urlInput, (newUrl) => {
    checkUrlSupport(newUrl)
  })

  watch(
    () => draftSettings.value.agent.provider,
    () => {
      modelList.value = []
    }
  )

  const showCustomUrl = computed(() => normalizeProviderId(draftSettings.value.agent.provider) === 'custom')

  // 测试 Firecrawl 连接
  async function handleTestFirecrawl() {
    if (!draftSettings.value.firecrawl.apiKey) {
      showToast('请输入 Firecrawl API Key', 'warning')
      return
    }

    testingFirecrawl.value = true
    try {
      const result = await testFirecrawlConnection(draftSettings.value.firecrawl.apiKey)
      if (result.success) {
        showToast('Firecrawl 连接成功', 'success')
      } else {
        showToast(`连接失败: ${result.error}`, 'error')
      }
    } catch (e) {
      showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
    } finally {
      testingFirecrawl.value = false
    }
  }

  // 测试 Agent 连接
  async function handleTestAgent() {
    if (providerRequiresApiKey(draftSettings.value.agent.provider) && !draftSettings.value.agent.apiKey) {
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
        showToast(`连接失败: ${result.error}`, 'error')
      }
    } catch (e) {
      showToast(`连接失败: ${e instanceof Error ? e.message : '未知错误'}`, 'error')
    } finally {
      testingAgent.value = false
    }
  }

  async function handleFetchModels() {
    const provider = draftSettings.value.agent.provider
    const apiKey = draftSettings.value.agent.apiKey?.trim()
    const baseUrl = draftSettings.value.agent.customBaseUrl?.trim()

    if (providerRequiresApiKey(provider) && !apiKey) {
      showToast('请先填写 API Key', 'warning')
      return
    }

    if (!providerSupportsCapability(provider, 'modelFetch')) {
      showToast(`${getProviderDisplayName(provider)} 不支持自动获取模型列表`, 'warning')
      return
    }

    if (providerRequiresBaseUrl(provider) && !baseUrl) {
      showToast('自定义服务需要先填写 Base URL', 'warning')
      return
    }

    isFetchingModels.value = true
    try {
      const result = await configApi.fetchModels(provider, apiKey, baseUrl || '')
      if (result.success && result.models && result.models.length > 0) {
        modelList.value = result.models.map(model => model.id)
        showToast(`获取到 ${result.models.length} 个模型`, 'success')
      } else {
        modelList.value = []
        showToast(result.message || '未获取到可用模型', 'warning')
      }
    } catch (error: unknown) {
      modelList.value = []
      showToast(error instanceof Error ? error.message : '获取模型列表失败', 'error')
    } finally {
      isFetchingModels.value = false
    }
  }

  // 重置提示词
  function handleResetPrompt() {
    if (confirm('确定要重置为默认提示词吗？')) {
      webImportStore.resetExtractionPrompt()
    }
  }

  onUnmounted(() => {
    if (checkSupportTimeout) {
      clearTimeout(checkSupportTimeout)
      checkSupportTimeout = null
    }
    if (focusInputTimeout) {
      clearTimeout(focusInputTimeout)
      focusInputTimeout = null
    }
  })

  return {
    webImportStore,
    urlInput,
    logsExpanded,
    selectedEngine,
    galleryDLAvailable,
    galleryDLSupported,
    checkingSupport,
    settingsExpanded,
    activeSettingsTab,
    testingFirecrawl,
    testingAgent,
    isFetchingModels,
    showFirecrawlKey,
    showAgentKey,
    modelList,
    isVisible,
    status,
    logs,
    extractResult,
    selectedPages,
    selectedCount,
    downloadProgress,
    downloadProgressPercent,
    error,
    isProcessing,
    draftSettings,
    hasUnsavedSettings,
    isSavingSettings,
    showAgentLogs,
    agentProviderOptions,
    supportsFetchModels,
    modelListOptions,
    currentEngine,
    engineDisplayName,
    isAllSelected,
    getPreviewUrl,
    handleClose,
    handleSaveSettings,
    handleDiscardSettings,
    ensureSettingsReady,
    handleExtract,
    togglePage,
    toggleAll,
    handleImport,
    showCustomUrl,
    handleTestFirecrawl,
    handleTestAgent,
    handleFetchModels,
    handleResetPrompt,
    getProviderDisplayName,
    providerRequiresApiKey,
    providerRequiresBaseUrl,
  }
}
