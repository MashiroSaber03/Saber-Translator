/**
 * 会话状态管理 Store
 * 管理翻译会话的保存、加载、书籍/章节上下文
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SessionListItem } from '@/types/api'
import type { ImageData } from '@/types/image'

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
    console.log(`会话上下文已设置: 书籍=${bookId}, 章节=${chapterId}`)
  }

  /**
   * 设置书籍/章节上下文（别名，兼容 main.js 迁移）
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
    console.log('会话上下文已清除')
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
    console.log(`当前会话名称: ${name}`)
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
    console.log(`会话列表已更新，共 ${list.length} 个会话`)
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

  // ============================================================
  // 会话数据序列化
  // ============================================================

  /**
   * 创建会话数据对象
   * @param name - 会话名称
   * @param images - 图片数据数组
   * @param currentIndex - 当前图片索引
   * @param uiSettings - UI 设置
   * @returns 会话数据对象
   */
  function createSessionData(
    name: string,
    images: ImageData[],
    currentIndex: number,
    uiSettings: Record<string, unknown>
  ): SessionData {
    return {
      name,
      version: '2.0',
      savedAt: new Date().toISOString(),
      imageCount: images.length,
      ui_settings: uiSettings,
      images: images.map(img => ({
        originalDataURL: img.originalDataURL,
        translatedDataURL: img.translatedDataURL || undefined,
        cleanImageData: img.cleanImageData || undefined,
        bubbleStates: img.bubbleStates || undefined,
        fileName: img.fileName,
        fontSize: img.fontSize,
        autoFontSize: img.autoFontSize,
        fontFamily: img.fontFamily,
        layoutDirection: img.layoutDirection,
        textColor: img.textColor,
        fillColor: img.fillColor,
        inpaintMethod: img.inpaintMethod,
        strokeEnabled: img.strokeEnabled,
        strokeColor: img.strokeColor,
        strokeWidth: img.strokeWidth,
        translationStatus: img.translationStatus,
        translationFailed: img.translationFailed
      })),
      currentImageIndex: currentIndex
    }
  }

  // ============================================================
  // 重置方法
  // ============================================================

  /**
   * 重置所有状态
   */
  function reset(): void {
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
    console.log('会话状态已重置')
  }

  // ============================================================
  // 章节会话管理（书架模式）
  // ============================================================

  /**
   * 将图片 URL 转换为 Base64
   * 复刻原版 imageUrlToBase64 逻辑
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
   * 复刻原版 convertImagesToBase64 逻辑
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
    setLoading(true)
    setError(null)
    loadingProgress.value = { current: 0, total: 0, message: '正在加载...' }
    
    try {
      // 获取 store 实例
      const { useImageStore } = await import('@/stores/imageStore')
      const { useSettingsStore } = await import('@/stores/settingsStore')
      const { useBubbleStore } = await import('@/stores/bubbleStore')
      const imageStore = useImageStore()
      const settingsStore = useSettingsStore()
      const bubbleStore = useBubbleStore()
      
      // 调用 API 按路径加载会话
      const { loadSessionByPathApi } = await import('@/api/session')
      const response = await loadSessionByPathApi(sessionPath)
      
      if (!response.success || !response.session) {
        throw new Error(response.error || '加载会话失败')
      }
      
      const sessionData = response.session
      
      // 转换会话数据为 ImageData 格式
      if (sessionData.images && sessionData.images.length > 0) {
        const images: ImageData[] = sessionData.images.map((img, index) => ({
          id: `session-${index}-${Date.now()}`,
          originalDataURL: img.originalDataURL,
          translatedDataURL: img.translatedDataURL || null,
          cleanImageData: img.cleanImageData || null,
          bubbleStates: (img.bubbleStates || []) as import('@/types/bubble').BubbleState[],
          fileName: img.fileName || `image-${index + 1}.png`,
          translationStatus: (img.translationStatus as 'pending' | 'processing' | 'completed' | 'failed') || 'pending',
          translationFailed: Boolean(img.translationFailed),
          hasUnsavedChanges: false,
          fontSize: (img.fontSize as number) || 24,
          autoFontSize: (img.autoFontSize as boolean) ?? true,
          fontFamily: (img.fontFamily as string) || 'Microsoft YaHei',
          layoutDirection: (img.layoutDirection as 'vertical' | 'horizontal') || 'vertical',
          textColor: (img.textColor as string) || '#000000',
          fillColor: (img.fillColor as string) || '#FFFFFF',
          inpaintMethod: (img.inpaintMethod as import('@/types/bubble').InpaintMethod) || 'litelama',
          strokeEnabled: (img.strokeEnabled as boolean) ?? false,
          strokeColor: (img.strokeColor as string) || '#FFFFFF',
          strokeWidth: (img.strokeWidth as number) || 2,
        }))
        
        // 将图片 URL 转换为 Base64（用于 Canvas 操作和翻译功能）
        // 复刻原版逻辑：显示进度并逐张转换
        if (images.length > 0) {
          console.log('正在加载图片...')
          loadingProgress.value = { current: 0, total: images.length, message: '正在加载图片...' }
          
          await convertImagesToBase64(images, (current, total) => {
            const progress = (current / total) * 100
            loadingProgress.value = { current, total, message: `加载图片 ${current}/${total}...` }
            console.log(`加载图片 ${current}/${total}... (${progress.toFixed(0)}%)`)
          })
          
          loadingProgress.value = { current: images.length, total: images.length, message: '加载完成' }
          console.log('图片加载完成，已转换为 Base64')
          
          // 延迟清除进度信息
          setTimeout(() => {
            loadingProgress.value = { current: 0, total: 0, message: '' }
          }, 500)
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
        
        // 恢复当前图片的气泡状态到 bubbleStore（skipSync=true 避免冗余同步）
        // 【修复】使用 clearBubblesLocal 保持 null 和 [] 的语义区分
        const currentImage = images[newIndex]
        if (currentImage && currentImage.bubbleStates && currentImage.bubbleStates.length > 0) {
          bubbleStore.setBubbles(currentImage.bubbleStates, true)
        } else {
          bubbleStore.clearBubblesLocal()
        }
        
        console.log(`按路径加载会话成功: ${sessionPath}, 共 ${images.length} 张图片`)
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
          layoutDirection: (uiSettings.layoutDirection as 'vertical' | 'horizontal') || settingsStore.settings.textStyle.layoutDirection,
          textColor: (uiSettings.textColor as string) || settingsStore.settings.textStyle.textColor,
          fillColor: (uiSettings.fillColor as string) || settingsStore.settings.textStyle.fillColor,
          inpaintMethod,
          strokeEnabled: (uiSettings.strokeEnabled as boolean) ?? settingsStore.settings.textStyle.strokeEnabled,
          strokeColor: (uiSettings.strokeColor as string) || settingsStore.settings.textStyle.strokeColor,
          strokeWidth: (uiSettings.strokeWidth as number) || settingsStore.settings.textStyle.strokeWidth,
        })
        
        console.log('UI 设置已恢复')
      }
      
      // 设置当前会话名称
      setSessionName(sessionPath)
      
      return true
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : '加载会话失败'
      setError(errorMsg)
      console.error(`按路径加载会话失败: ${sessionPath}`, e)
      throw e
    } finally {
      setLoading(false)
    }
  }

  /**
   * 保存章节会话（分批保存，复刻原版 session.js 的 saveChapterSession + saveSessionInBatches）
   * @param bookId - 书籍 ID
   * @param chapterId - 章节 ID
   * @returns 是否保存成功
   */
  async function saveChapterSession(bookId: string, chapterId: string): Promise<boolean> {
    // 检查参数
    if (!bookId || !chapterId) {
      console.log('未在书籍/章节模式，不支持保存')
      return false
    }

    console.log(`保存章节会话（分批）: book=${bookId}, chapter=${chapterId}`)

    // 获取 imageStore 和 settingsStore
    const { useImageStore } = await import('@/stores/imageStore')
    const { useSettingsStore } = await import('@/stores/settingsStore')
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()

    // 检查是否有图片数据
    const allImages = Array.isArray(imageStore.images) ? imageStore.images : []
    
    if (!allImages || allImages.length === 0) {
      console.log('没有图片数据可保存')
      return false
    }

    // 构建会话路径（格式必须与后端 bookshelf_manager.get_chapter_session_path 一致）
    const sessionPath = `bookshelf/${bookId}/${chapterId}`

    setSaving(true)
    setError(null)

    // 更新进度显示
    loadingProgress.value = { current: 0, total: 100, message: '准备保存...' }

    try {
      // 导入 API
      const { batchSaveStartApi, batchSaveImageApi, batchSaveCompleteApi } = await import('@/api/session')

      const totalImages = allImages.length

      // 步骤1: 收集元数据（仅渲染相关设置，AI设置使用全局配置）
      loadingProgress.value = { current: 5, total: 100, message: '收集元数据...' }

      const { textStyle, targetLanguage, sourceLanguage } = settingsStore.settings
      const uiSettings: Record<string, unknown> = {
        targetLanguage: targetLanguage,
        sourceLanguage: sourceLanguage,
        fontSize: textStyle.fontSize,
        autoFontSize: textStyle.autoFontSize,
        fontFamily: textStyle.fontFamily,
        layoutDirection: textStyle.layoutDirection,
        textColor: textStyle.textColor,
        useInpaintingMethod: textStyle.inpaintMethod,
        fillColor: textStyle.fillColor,
        strokeEnabled: textStyle.strokeEnabled,
        strokeColor: textStyle.strokeColor,
        strokeWidth: textStyle.strokeWidth,
      }

      // 图片元数据（不含 Base64 数据）
      const imagesMeta = allImages.map((img) => {
        const meta: Record<string, unknown> = {}
        for (const key of Object.keys(img)) {
          if (key !== 'originalDataURL' && key !== 'translatedDataURL' && key !== 'cleanImageData') {
            meta[key] = img[key as keyof typeof img]
          }
        }
        meta.hasOriginalData = !!img.originalDataURL
        meta.hasTranslatedData = !!img.translatedDataURL
        meta.hasCleanData = !!img.cleanImageData
        return meta
      })

      const metadata = {
        ui_settings: uiSettings,
        images_meta: imagesMeta,
        currentImageIndex: imageStore.currentImageIndex
      }

      // 步骤2: 调用开始保存 API
      loadingProgress.value = { current: 10, total: 100, message: '初始化保存...' }

      const startResponse = await batchSaveStartApi(sessionPath, metadata)
      if (!startResponse.success) {
        throw new Error(startResponse.error || '初始化保存失败')
      }

      const sessionFolder = startResponse.session_folder
      if (!sessionFolder) {
        throw new Error('未获取到会话文件夹路径')
      }
      console.log(`会话文件夹: ${sessionFolder}`)

      // 步骤3: 逐张保存图片
      let savedCount = 0
      let failedCount = 0

      const isBase64Data = (data: string | null | undefined): boolean => {
        return !!data && typeof data === 'string' && data.startsWith('data:')
      }

      for (let i = 0; i < totalImages; i++) {
        const img = allImages[i]
        if (!img) continue
        
        const progress = 10 + (i / totalImages) * 80
        loadingProgress.value = { 
          current: Math.round(progress), 
          total: 100, 
          message: `保存图片 ${i + 1}/${totalImages}...` 
        }

        if (isBase64Data(img.originalDataURL)) {
          try {
            const resp = await batchSaveImageApi(sessionFolder, i, 'original', img.originalDataURL)
            if (!resp.success) {
              console.error(`保存图片 ${i} original 失败:`, resp.error)
              failedCount++
            }
          } catch (e) {
            console.error(`保存图片 ${i} original 出错:`, e)
            failedCount++
          }
        }

        if (isBase64Data(img.translatedDataURL)) {
          try {
            const resp = await batchSaveImageApi(sessionFolder, i, 'translated', img.translatedDataURL!)
            if (!resp.success) {
              console.error(`保存图片 ${i} translated 失败:`, resp.error)
              failedCount++
            }
          } catch (e) {
            console.error(`保存图片 ${i} translated 出错:`, e)
            failedCount++
          }
        }

        if (img.cleanImageData && typeof img.cleanImageData === 'string' && !img.cleanImageData.startsWith('/api/')) {
          try {
            const resp = await batchSaveImageApi(sessionFolder, i, 'clean', img.cleanImageData)
            if (!resp.success) {
              console.error(`保存图片 ${i} clean 失败:`, resp.error)
              failedCount++
            }
          } catch (e) {
            console.error(`保存图片 ${i} clean 出错:`, e)
            failedCount++
          }
        }

        savedCount++
      }

      // 步骤4: 完成保存
      loadingProgress.value = { current: 95, total: 100, message: '完成保存...' }

      const completeResponse = await batchSaveCompleteApi(sessionFolder, imagesMeta)
      if (!completeResponse.success) {
        throw new Error(completeResponse.error || '完成保存失败')
      }

      loadingProgress.value = { current: 100, total: 100, message: '保存完成' }
      console.log(`分批保存完成: ${savedCount} 张图片, ${failedCount} 个失败`)

      setTimeout(() => {
        loadingProgress.value = { current: 0, total: 0, message: '' }
      }, 1000)

      return true

    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : '保存章节会话失败'
      setError(errorMsg)
      console.error('分批保存失败:', e)
      loadingProgress.value = { current: 0, total: 0, message: '' }
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

    // 计算属性
    isBookshelfMode,
    currentBookId,
    currentChapterId,

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

    // 会话数据
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
