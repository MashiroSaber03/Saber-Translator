/**
 * 会话状态管理 Store
 * 管理翻译会话的保存、加载、书籍/章节上下文
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { persistAllPages } from '@/composables/translation/core/persistenceService'
import type { SessionListItem } from '@/types/api'
import type { ImageData } from '@/types/image'
import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { OcrResult } from '@/types/ocr'
import { normalizeImageTextStyleFields } from '@/defaults/textStyleDefaults'
import { getTextlinesPerBubbleFromStates } from '@/utils/bubbleFactory'

/**
 * 会话数据接口（用于保存和加载）
 */
export interface SessionData {
  /** 会话名称 */
  name: string
  /** 版本号 */
  version: string
  /** 保存时间 */
  savedAt: string
  /** 图片数量 */
  imageCount: number
  /** UI 设置 */
  ui_settings: Record<string, unknown>
  /** 图片数据数组 */
  images: Array<{
    originalDataURL: string
    translatedDataURL?: string
    cleanImageData?: string
    bubbleStates?: unknown[]
    fileName: string
    [key: string]: unknown
  }>
  /** 当前图片索引 */
  currentImageIndex: number
}

// ============================================================
// 类型定义
// ============================================================

/**
 * 会话上下文（书架模式）
 */
export interface SessionContext {
  /** 当前书籍ID */
  bookId: string | null
  /** 当前章节ID */
  chapterId: string | null
  /** 当前书籍标题 */
  bookTitle: string | null
  /** 当前章节标题 */
  chapterTitle: string | null
}

/**
 * 会话保存选项
 */
export interface SessionSaveOptions {
  /** 会话名称 */
  name: string
  /** 是否为书架模式 */
  isBookshelfMode?: boolean
}

interface BatchSaveState {
  isInProgress: boolean
  totalCount: number
  currentIndex: number
  sessionId: string | null
}

function isBubbleTextline(value: unknown): value is BubbleTextline {
  if (!value || typeof value !== 'object') return false

  const textline = value as Partial<BubbleTextline>
  return (
    Array.isArray(textline.polygon) &&
    (textline.direction === 'h' || textline.direction === 'v') &&
    typeof textline.confidence === 'number'
  )
}

function readTextlinesPerBubble(value: unknown): BubbleTextline[][] | undefined {
  if (!Array.isArray(value)) return undefined

  return value.map((group) => (
    Array.isArray(group) ? group.filter(isBubbleTextline) : []
  ))
}

// ============================================================
// Store 定义
// ============================================================

export const useSessionStore = defineStore('session', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 当前会话名称 */
  const currentSessionName = ref<string | null>(null)

  /** 会话上下文（书架模式） */
  const context = ref<SessionContext>({
    bookId: null,
    chapterId: null,
    bookTitle: null,
    chapterTitle: null
  })

  /** 会话列表 */
  const sessionList = ref<SessionListItem[]>([])

  /** 加载状态 */
  const isLoading = ref(false)

  /** 加载进度信息 */
  const loadingProgress = ref({
    current: 0,
    total: 0,
    message: '',
  })

  /** 错误信息 */
  const error = ref<string | null>(null)

  /** 是否正在保存 */
  const isSaving = ref(false)

  const batchSaveState = ref<BatchSaveState>({
    isInProgress: false,
    totalCount: 0,
    currentIndex: 0,
    sessionId: null,
  })

  let progressClearTimer: ReturnType<typeof setTimeout> | null = null

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否为书架模式 */
  const isBookshelfMode = computed(() => {
    return context.value.bookId !== null && context.value.chapterId !== null
  })

  /** 当前书籍ID */
  const currentBookId = computed(() => context.value.bookId)

  /** 当前章节ID */
  const currentChapterId = computed(() => context.value.chapterId)

  const batchSaveProgress = computed(() => {
    if (!batchSaveState.value.isInProgress || batchSaveState.value.totalCount <= 0) {
      return 0
    }

    return Math.round((batchSaveState.value.currentIndex / batchSaveState.value.totalCount) * 100)
  })

  // ============================================================
  // 上下文管理方法
  // ============================================================

  /**
   * 设置会话上下文（书架模式）
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   * @param bookTitle - 书籍标题
   * @param chapterTitle - 章节标题
   */
  function setContext(
    bookId: string | null,
    chapterId: string | null,
    bookTitle: string | null = null,
    chapterTitle: string | null = null
  ): void {
    context.value = {
      bookId,
      chapterId,
      bookTitle,
      chapterTitle
    }
  }

  /**
   * 设置书籍/章节上下文
   * @param bookId - 书籍ID
   * @param chapterId - 章节ID
   * @param bookTitle - 书籍标题
   * @param chapterTitle - 章节标题
   */
  function setBookChapterContext(
    bookId: string,
    chapterId: string,
    bookTitle: string,
    chapterTitle: string
  ): void {
    setContext(bookId, chapterId, bookTitle, chapterTitle)
  }

  /**
   * 清除会话上下文
   */
  function clearContext(): void {
    context.value = {
      bookId: null,
      chapterId: null,
      bookTitle: null,
      chapterTitle: null
    }
  }

  /**
   * 从 URL 参数解析上下文
   * @param searchParams - URL 查询参数
   */
  function parseContextFromUrl(searchParams: URLSearchParams): void {
    const bookId = searchParams.get('book')
    const chapterId = searchParams.get('chapter')

    if (bookId && chapterId) {
      setContext(bookId, chapterId)
    }
  }

  // ============================================================
  // 会话名称管理
  // ============================================================

  /**
   * 设置当前会话名称
   * @param name - 会话名称
   */
  function setSessionName(name: string | null): void {
    currentSessionName.value = name
  }

  /**
   * 清除当前会话名称
   */
  function clearSessionName(): void {
    currentSessionName.value = null
  }

  // ============================================================
  // 会话列表管理
  // ============================================================

  /**
   * 设置会话列表
   * @param list - 会话列表
   */
  function setSessionList(list: SessionListItem[]): void {
    sessionList.value = list
  }

  /**
   * 添加会话到列表
   * @param session - 会话信息
   */
  function addToSessionList(session: SessionListItem): void {
    // 检查是否已存在
    const existingIndex = sessionList.value.findIndex(s => s.name === session.name)
    if (existingIndex >= 0) {
      // 更新现有会话
      sessionList.value[existingIndex] = session
    } else {
      // 添加新会话
      sessionList.value.unshift(session)
    }
  }

  /**
   * 从列表中移除会话
   * @param name - 会话名称
   */
  function removeFromSessionList(name: string): void {
    const index = sessionList.value.findIndex(s => s.name === name)
    if (index >= 0) {
      sessionList.value.splice(index, 1)
    }
  }

  /**
   * 重命名会话
   * @param oldName - 旧名称
   * @param newName - 新名称
   */
  function renameInSessionList(oldName: string, newName: string): void {
    const session = sessionList.value.find(s => s.name === oldName)
    if (session) {
      session.name = newName
    }
  }

  // ============================================================
  // 加载/保存状态管理
  // ============================================================

  /**
   * 设置加载状态
   * @param loading - 是否正在加载
   */
  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  /**
   * 设置保存状态
   * @param saving - 是否正在保存
   */
  function setSaving(saving: boolean): void {
    isSaving.value = saving
  }

  /**
   * 设置错误信息
   * @param message - 错误信息
   */
  function setError(message: string | null): void {
    error.value = message
  }

  function startBatchSave(totalCount: number, sessionId: string): void {
    batchSaveState.value = {
      isInProgress: true,
      totalCount,
      currentIndex: 0,
      sessionId,
    }
  }

  function updateBatchSaveProgress(currentIndex: number): void {
    batchSaveState.value.currentIndex = currentIndex
  }

  function completeBatchSave(): void {
    batchSaveState.value = {
      isInProgress: false,
      totalCount: 0,
      currentIndex: 0,
      sessionId: null,
    }
  }

  function clearProgressClearTimer(): void {
    if (progressClearTimer) {
      clearTimeout(progressClearTimer)
      progressClearTimer = null
    }
  }

  function clearLoadingProgress(): void {
    loadingProgress.value = { current: 0, total: 0, message: '' }
  }

  function scheduleLoadingProgressClear(delayMs: number): void {
    clearProgressClearTimer()
    progressClearTimer = setTimeout(() => {
      clearLoadingProgress()
      progressClearTimer = null
    }, delayMs)
  }

  function createSessionData(
    name: string,
    images: ImageData[],
    currentImageIndex: number,
    uiSettings: Record<string, unknown>
  ): SessionData {
    return {
      name,
      version: '2.0',
      savedAt: new Date().toISOString(),
      imageCount: images.length,
      currentImageIndex,
      ui_settings: uiSettings,
      images: images.map((image) => ({
        ...image,
        originalDataURL: image.originalDataURL,
        translatedDataURL: image.translatedDataURL || undefined,
        cleanImageData: image.cleanImageData || undefined,
        bubbleStates: image.bubbleStates || undefined,
        fileName: image.fileName,
      })),
    }
  }

  // ============================================================
  // 重置方法
  // ============================================================

  /**
   * 重置所有状态
   */
  function reset(): void {
    clearProgressClearTimer()
    currentSessionName.value = null
    context.value = {
      bookId: null,
      chapterId: null,
      bookTitle: null,
      chapterTitle: null
    }
    sessionList.value = []
    isLoading.value = false
    isSaving.value = false
    error.value = null
    clearLoadingProgress()
    completeBatchSave()
  }

  // ============================================================
  // 章节会话管理（书架模式）
  // ============================================================

  /**
   * 将图片 URL 转换为 Base64
   */
  async function imageUrlToBase64(url: string | null): Promise<string | null> {
    if (!url || typeof url !== 'string') return null
    // 如果已经是 Base64，直接返回
    if (url.startsWith('data:')) return url
    // 如果不是 API URL，返回 null
    if (!url.startsWith('/api/')) return null

    try {
      const response = await fetch(url)
      if (!response.ok) return null

      const blob = await response.blob()
      return new Promise((resolve) => {
        const reader = new FileReader()
        reader.onloadend = () => resolve(reader.result as string)
        reader.onerror = () => resolve(null)
        reader.readAsDataURL(blob)
      })
    } catch (error) {
      console.error(`转换图片 URL 失败: ${url}`, error)
      return null
    }
  }

  /**
   * 将会话中的所有图片 URL 转换为 Base64
   */
  async function convertImagesToBase64(
    images: ImageData[],
    progressCallback?: (current: number, total: number) => void
  ): Promise<void> {
    const total = images.length

    for (let i = 0; i < total; i++) {
      const img = images[i]
      if (!img) continue
      if (progressCallback) progressCallback(i + 1, total)

      // 转换原图
      if (img.originalDataURL && img.originalDataURL.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.originalDataURL)
        if (base64) img.originalDataURL = base64
      }

      // 转换翻译图
      if (img.translatedDataURL && img.translatedDataURL.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.translatedDataURL)
        if (base64) img.translatedDataURL = base64
      }

      // 转换干净背景（cleanImageData 存储的是纯 Base64，不带 data: 前缀）
      if (img.cleanImageData && img.cleanImageData.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.cleanImageData)
        if (base64) {
          // 移除 data:image/png;base64, 前缀
          img.cleanImageData = base64.replace(/^data:image\/\w+;base64,/, '')
        }
      }
    }
  }

  async function loadSessionByPath(sessionPath: string): Promise<boolean> {
    clearProgressClearTimer()
    setLoading(true)
    setError(null)
    loadingProgress.value = { current: 0, total: 0, message: '正在加载...' }

    try {
      const { useSettingsStore } = await import('@/stores/settings')
      const imageStore = useImageStore()
      const settingsStore = useSettingsStore()
      const bubbleStore = useBubbleStore()

      const { loadSessionByPath } = await import('@/api/session')
      const response = await loadSessionByPath(sessionPath)

      if (!response.success || !response.session) {
        throw new Error(response.error || '加载会话失败')
      }

      const sessionData = response.session

      // 转换会话数据为 ImageData 格式
      if (sessionData.images && sessionData.images.length > 0) {
        const images: ImageData[] = sessionData.images.map((img, index) => {
          const restoredTextlines = readTextlinesPerBubble(img.textlinesPerBubble)
          const bubbleStates = (img.bubbleStates !== undefined && img.bubbleStates !== null)
            ? (img.bubbleStates as BubbleState[]).map((state, bubbleIndex) => ({
                ...state,
                textlines: state.textlines && state.textlines.length > 0
                  ? state.textlines
                  : restoredTextlines?.[bubbleIndex] ?? []
              }))
            : null

          return ({
          id: `session-${index}-${Date.now()}`,
          originalDataURL: img.originalDataURL,
          translatedDataURL: img.translatedDataURL || null,
          cleanImageData: img.cleanImageData || null,
          // 图片尺寸（可选）
          width: (img.width as number) || undefined,
          height: (img.height as number) || undefined,
          // 保留 bubbleStates 的 null 语义：null/undefined 表示需要自动检测，[] 表示用户主动清空了文本框。
          bubbleStates: bubbleStates,
          bubbleCoords: bubbleStates
            ? bubbleStates.map((state) => state.coords)
            : (img.bubbleCoords !== undefined ? (img.bubbleCoords as BubbleCoords[]) : undefined),
          bubbleAngles: bubbleStates
            ? bubbleStates.map((state) => state.rotationAngle || 0)
            : (img.bubbleAngles !== undefined ? (img.bubbleAngles as number[]) : undefined),
          originalTexts: bubbleStates
            ? bubbleStates.map((state) => state.originalText || '')
            : (img.originalTexts !== undefined ? (img.originalTexts as string[]) : undefined),
          bubbleTexts: bubbleStates
            ? bubbleStates.map((state) => state.translatedText || '')
            : (img.bubbleTexts !== undefined ? (img.bubbleTexts as string[]) : undefined),
          textboxTexts: bubbleStates
            ? bubbleStates.map((state) => state.textboxText || '')
            : (img.textboxTexts !== undefined ? (img.textboxTexts as string[]) : undefined),
          textlinesPerBubble: bubbleStates
            ? getTextlinesPerBubbleFromStates(bubbleStates)
            : restoredTextlines,
          // 恢复手动标注标记
          isManuallyAnnotated: Boolean(img.isManuallyAnnotated),
          // 恢复文件夹路径信息
          relativePath: (img.relativePath as string) || undefined,
          folderPath: (img.folderPath as string) || undefined,
          ocrResults: bubbleStates
            ? bubbleStates.map((state, bubbleIndex) => state.ocrResult || ((img.ocrResults as OcrResult[] | undefined)?.[bubbleIndex] ?? {
                text: state.originalText || '',
                confidence: null,
                confidenceSupported: false,
                engine: '',
                primaryEngine: '',
                fallbackUsed: false
              }))
            : (img.ocrResults !== undefined ? (img.ocrResults as OcrResult[]) : undefined),
          fileName: img.fileName || `image-${index + 1}.png`,
          translationStatus: (img.translationStatus as 'pending' | 'processing' | 'completed' | 'failed') || 'pending',
          translationFailed: Boolean(img.translationFailed),
          hasUnsavedChanges: false,
          ...normalizeImageTextStyleFields(img as unknown as Partial<ImageData>),
          // 双掩膜系统字段
          textMask: (img.textMask as string) || null,
          userMask: (img.userMask as string) || null,
        })})

        // 将图片 URL 转换为 Base64，用于 Canvas 操作和翻译功能。
        if (images.length > 0) {
          loadingProgress.value = { current: 0, total: images.length, message: '正在加载图片...' }

          await convertImagesToBase64(images, (current, total) => {
            loadingProgress.value = { current, total, message: `加载图片 ${current}/${total}...` }
          })

          loadingProgress.value = { current: images.length, total: images.length, message: '加载完成' }

          scheduleLoadingProgressClear(500)
        }

        // 设置图片到 imageStore
        imageStore.setImages(images)

        // 设置当前图片索引
        let newIndex = 0
        if (typeof sessionData.currentImageIndex === 'number') {
          newIndex = sessionData.currentImageIndex
          if (newIndex >= images.length || newIndex < 0) {
            newIndex = images.length > 0 ? 0 : -1
          }
        }
        imageStore.setCurrentImageIndex(newIndex)

        // 恢复当前图片的气泡状态到 bubbleStore（skipSync=true 避免冗余同步）。
        // 无气泡状态时使用本地清空，保留 null 和 [] 的语义区分。
        const currentImage = images[newIndex]
        if (currentImage && currentImage.bubbleStates && currentImage.bubbleStates.length > 0) {
          bubbleStore.setBubbles(currentImage.bubbleStates, true)
        } else {
          bubbleStore.clearBubblesLocal()
        }

      }

      // 恢复 UI 设置到 settingsStore
      const uiSettings = sessionData.ui_settings
      if (uiSettings) {
        // 恢复语言设置
        if (uiSettings.targetLanguage || uiSettings.sourceLanguage) {
          settingsStore.updateSettings({
            targetLanguage: (uiSettings.targetLanguage as string) || undefined,
            sourceLanguage: (uiSettings.sourceLanguage as string) || undefined,
          })
        }

        // 恢复文字样式设置
        const inpaintValue = uiSettings.useInpaintingMethod as string
        type ValidInpaintMethod = 'solid' | 'lama_mpe' | 'litelama'
        const validInpaintMethods: ValidInpaintMethod[] = ['solid', 'lama_mpe', 'litelama']
        const inpaintMethod: ValidInpaintMethod = validInpaintMethods.includes(inpaintValue as ValidInpaintMethod)
          ? (inpaintValue as ValidInpaintMethod)
          : settingsStore.settings.textStyle.inpaintMethod

        settingsStore.updateTextStyle({
          fontSize: (uiSettings.fontSize as number) || settingsStore.settings.textStyle.fontSize,
          autoFontSize: (uiSettings.autoFontSize as boolean) ?? settingsStore.settings.textStyle.autoFontSize,
          fontFamily: (uiSettings.fontFamily as string) || settingsStore.settings.textStyle.fontFamily,
          layoutDirection: (uiSettings.layoutDirection as 'vertical' | 'horizontal' | 'auto') || settingsStore.settings.textStyle.layoutDirection,
          textColor: (uiSettings.textColor as string) || settingsStore.settings.textStyle.textColor,
          fillColor: (uiSettings.fillColor as string) || settingsStore.settings.textStyle.fillColor,
          inpaintMethod,
          strokeEnabled: (uiSettings.strokeEnabled as boolean) ?? settingsStore.settings.textStyle.strokeEnabled,
          strokeColor: (uiSettings.strokeColor as string) || settingsStore.settings.textStyle.strokeColor,
          strokeWidth: (uiSettings.strokeWidth as number) || settingsStore.settings.textStyle.strokeWidth,
          lineSpacing: (uiSettings.lineSpacing as number) ?? settingsStore.settings.textStyle.lineSpacing,
          textAlign: (uiSettings.textAlign as 'start' | 'center' | 'end') || settingsStore.settings.textStyle.textAlign,
          useAutoTextColor: (uiSettings.useAutoTextColor as boolean) ?? settingsStore.settings.textStyle.useAutoTextColor,
        })

      }

      // 设置当前会话名称
      setSessionName(sessionPath)

      return true
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : '加载会话失败'
      setError(errorMsg)
      clearProgressClearTimer()
      console.error(`按路径加载会话失败: ${sessionPath}`, e)
      throw e
    } finally {
      setLoading(false)
    }
  }

  /**
   * 保存章节会话（使用新的单页存储 API，逐页保存）
   * @param bookId - 书籍 ID
   * @param chapterId - 章节 ID
   * @returns 是否保存成功
   */
  async function saveChapterSession(bookId: string, chapterId: string): Promise<boolean> {
    // 检查参数
    if (!bookId || !chapterId) {
      return false
    }

    // 获取 imageStore 和 settingsStore
    const { useSettingsStore } = await import('@/stores/settings')
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()

    // 检查是否有图片数据
    const allImages = Array.isArray(imageStore.images) ? imageStore.images : []

    if (!allImages || allImages.length === 0) {
      return false
    }

    // 构建章节会话路径
    const sessionPath = `bookshelf/${bookId}/chapters/${chapterId}/session`

    clearProgressClearTimer()
    setSaving(true)
    setError(null)

    const totalImages = allImages.length
    loadingProgress.value = { current: 0, total: totalImages, message: `准备保存 ${totalImages} 张图片...` }

    try {
      const { createPipelineRuntime, hydrateTaskContextFromImage } = await import('@/composables/translation/core/runtime')

      const runtime = createPipelineRuntime('standard', {
        settingsSnapshot: settingsStore.settings,
        autoSaveEnabled: true,
        sessionPath,
        bookId,
        chapterId,
      })

      const contexts = allImages.map((image, index) => hydrateTaskContextFromImage(index, image, 'standard', runtime))

      await persistAllPages(contexts, runtime, {
        includeOriginal: true,
        includeDerivedImagesFromSource: true,
        currentImageIndex: imageStore.currentImageIndex,
        onProgress: (current, total) => {
          loadingProgress.value = {
            current,
            total,
            message: `保存图片 ${current}/${total}...`
          }
        }
      })

      for (let index = 0; index < allImages.length; index++) {
        imageStore.updateImageByIndex(index, { hasUnsavedChanges: false })
      }

      loadingProgress.value = { current: totalImages, total: totalImages, message: '保存完成' }

      scheduleLoadingProgressClear(1000)

      return true

    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : '保存章节会话失败'
      setError(errorMsg)
      console.error('保存失败:', e)
      clearProgressClearTimer()
      clearLoadingProgress()
      return false
    } finally {
      setSaving(false)
    }
  }

  // ============================================================
  // 返回 Store 接口
  // ============================================================

  return {
    // 状态
    currentSessionName,
    context,
    sessionList,
    isLoading,
    isSaving,
    error,
    loadingProgress,
    batchSaveState,

    // 计算属性
    isBookshelfMode,
    currentBookId,
    currentChapterId,
    batchSaveProgress,

    // 上下文管理
    setContext,
    clearContext,
    parseContextFromUrl,

    // 会话名称管理
    setSessionName,
    clearSessionName,

    // 会话列表管理
    setSessionList,
    addToSessionList,
    removeFromSessionList,
    renameInSessionList,

    // 加载/保存状态
    setLoading,
    setSaving,
    setError,
    startBatchSave,
    updateBatchSaveProgress,
    completeBatchSave,
    createSessionData,

    // 图片转换工具
    imageUrlToBase64,

    // 章节会话管理
    saveChapterSession,
    loadSessionByPath,
    setBookChapterContext,

    // 重置
    reset
  }
})
