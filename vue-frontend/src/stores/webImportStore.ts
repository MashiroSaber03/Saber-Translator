/**
 * 网页导入状态管理 Store
 * 管理网页导入设置和运行时状态
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { WebImportSettings, WebImportState, AgentLog, ExtractResult, DownloadedImage } from '@/types/webImport'
import { STORAGE_KEY_WEB_IMPORT_SETTINGS } from '@/constants'
import { createDefaultWebImportSettings, useWebImportSettings } from './settings/modules/webImport'

// ============================================================
// Store 定义
// ============================================================

export const useWebImportStore = defineStore('webImport', () => {
  // ============================================================
  // 设置状态
  // ============================================================

  /** 网页导入设置 */
  const settings = ref<WebImportSettings>(createDefaultWebImportSettings())

  // ============================================================
  // 运行时状态
  // ============================================================

  /** 导入状态 */
  const status = ref<WebImportState['status']>('idle')

  /** 当前 URL */
  const url = ref('')

  /** Agent 日志 */
  const logs = ref<AgentLog[]>([])

  /** 提取结果 */
  const extractResult = ref<ExtractResult | null>(null)

  /** 已选页面 */
  const selectedPages = ref<Set<number>>(new Set())

  /** 下载进度 */
  const downloadProgress = ref({ current: 0, total: 0 })

  /** 已下载图片 */
  const downloadedImages = ref<DownloadedImage[]>([])

  /** 错误信息 */
  const error = ref<string | null>(null)

  /** 模态框是否可见 */
  const modalVisible = ref(false)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否正在提取 */
  const isExtracting = computed(() => status.value === 'extracting')

  /** 是否正在下载 */
  const isDownloading = computed(() => status.value === 'downloading')

  /** 是否正在处理中 */
  const isProcessing = computed(() => isExtracting.value || isDownloading.value)

  /** 已选页面数量 */
  const selectedCount = computed(() => selectedPages.value.size)

  /** 下载进度百分比 */
  const downloadProgressPercent = computed(() => {
    if (downloadProgress.value.total === 0) return 0
    return Math.round((downloadProgress.value.current / downloadProgress.value.total) * 100)
  })

  // ============================================================
  // localStorage 持久化
  // ============================================================

  function saveToStorage(): void {
    try {
      localStorage.setItem(STORAGE_KEY_WEB_IMPORT_SETTINGS, JSON.stringify(settings.value))
    } catch (e) {
      console.error('保存网页导入设置失败:', e)
    }
  }

  function loadFromStorage(): void {
    try {
      const data = localStorage.getItem(STORAGE_KEY_WEB_IMPORT_SETTINGS)
      if (data) {
        const parsed = JSON.parse(data)
        // 深度合并，确保新增字段不丢失
        settings.value = deepMerge(createDefaultWebImportSettings(), parsed)
      }
    } catch (e) {
      console.error('加载网页导入设置失败:', e)
    }
  }

  /** 深度合并对象 */
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

  // ============================================================
  // 运行时状态操作
  // ============================================================

  /** 打开模态框 */
  function openModal(): void {
    modalVisible.value = true
  }

  /** 关闭模态框 */
  function closeModal(): void {
    modalVisible.value = false
  }

  /** 重置运行时状态 */
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

  /** 设置 URL */
  function setUrl(newUrl: string): void {
    url.value = newUrl
  }

  /** 添加日志 */
  function addLog(log: AgentLog): void {
    logs.value.push(log)
  }

  /** 清空日志 */
  function clearLogs(): void {
    logs.value = []
  }

  /** 设置提取结果 */
  function setExtractResult(result: ExtractResult): void {
    // 兼容分片推送模式：如果已经有增量添加的页面，不覆盖
    if (extractResult.value && extractResult.value.pages.length > 0) {
      // 只更新元数据，保留现有的 pages
      extractResult.value.comicTitle = result.comicTitle
      extractResult.value.chapterTitle = result.chapterTitle
      extractResult.value.totalPages = result.totalPages
      extractResult.value.sourceUrl = result.sourceUrl
      extractResult.value.referer = result.referer
      extractResult.value.engine = result.engine
      extractResult.value.success = result.success
      extractResult.value.error = result.error
    } else {
      // 首次设置，直接覆盖
      extractResult.value = result
      if (result.success && result.pages) {
        // 默认全选
        selectedPages.value = new Set(result.pages.map((p) => p.pageNumber))
      }
    }
  }

  /** 切换页面选择 */
  function togglePageSelection(pageNumber: number): void {
    if (selectedPages.value.has(pageNumber)) {
      selectedPages.value.delete(pageNumber)
    } else {
      selectedPages.value.add(pageNumber)
    }
    // 触发响应式更新
    selectedPages.value = new Set(selectedPages.value)
  }

  /** 全选/取消全选 */
  function toggleSelectAll(): void {
    if (!extractResult.value?.pages) return

    if (selectedPages.value.size === extractResult.value.pages.length) {
      selectedPages.value = new Set()
    } else {
      selectedPages.value = new Set(extractResult.value.pages.map((p) => p.pageNumber))
    }
  }

  /** 设置状态 */
  function setStatus(newStatus: WebImportState['status']): void {
    status.value = newStatus
  }

  /** 设置错误 */
  function setError(errorMsg: string | null): void {
    error.value = errorMsg
    if (errorMsg) {
      status.value = 'error'
    }
  }

  /** 更新下载进度 */
  function updateDownloadProgress(current: number, total: number): void {
    downloadProgress.value = { current, total }
  }

  /** 设置下载结果 */
  function setDownloadedImages(images: DownloadedImage[]): void {
    downloadedImages.value = images
  }

  /** 增量添加页面（分片推送模式） */
  function addPageIncremental(page: { pageNumber: number; imageUrl: string; localPath?: string }): void {
    if (!extractResult.value) {
      // 如果还没有提取结果，先创建一个
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

    // 添加新页面
    extractResult.value.pages.push(page)
    extractResult.value.totalPages = extractResult.value.pages.length

    // 自动选中新页面
    selectedPages.value.add(page.pageNumber)
    selectedPages.value = new Set(selectedPages.value)  // 触发响应式更新
  }

  // ============================================================
  // 导入设置模块方法
  // ============================================================

  const settingsMethods = useWebImportSettings(settings, saveToStorage)

  // ============================================================
  // 初始化
  // ============================================================

  loadFromStorage()

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 设置
    settings,
    // 运行时状态
    status,
    url,
    logs,
    extractResult,
    selectedPages,
    downloadProgress,
    downloadedImages,
    error,
    modalVisible,
    // 计算属性
    isExtracting,
    isDownloading,
    isProcessing,
    selectedCount,
    downloadProgressPercent,
    // 持久化
    saveToStorage,
    loadFromStorage,
    // 运行时操作
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
    // 设置方法
    ...settingsMethods
  }
})
