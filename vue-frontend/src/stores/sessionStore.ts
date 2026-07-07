import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { persistAllPages } from '@/composables/translation/core/persistenceService'
import type { SessionListItem } from '@/types/api'
import type { ImageData } from '@/types/image'
import { readBlobAsDataUrl } from '@/utils/dataUrl'
import { applySessionUiSettings } from '@/stores/sessionUiSettings'
import { hydrateSessionImages } from '@/stores/sessionImageHydration'

export interface SessionData {
  name: string
  version: string
  savedAt: string
  imageCount: number
  ui_settings: Record<string, unknown>
  images: Array<{
    originalDataURL: string
    translatedDataURL?: string
    cleanImageData?: string
    bubbleStates?: unknown[]
    fileName: string
    [key: string]: unknown
  }>
  currentImageIndex: number
}

export interface SessionContext {
  bookId: string | null
  chapterId: string | null
  bookTitle: string | null
  chapterTitle: string | null
}

export interface SessionSaveOptions {
  name: string
  isBookshelfMode?: boolean
}

interface BatchSaveState {
  isInProgress: boolean
  totalCount: number
  currentIndex: number
  sessionId: string | null
}

export const useSessionStore = defineStore('session', () => {
  const currentSessionName = ref<string | null>(null)
  const context = ref<SessionContext>({
    bookId: null,
    chapterId: null,
    bookTitle: null,
    chapterTitle: null
  })

  const sessionList = ref<SessionListItem[]>([])
  const isLoading = ref(false)
  const loadingProgress = ref({
    current: 0,
    total: 0,
    message: '',
  })

  const error = ref<string | null>(null)
  const isSaving = ref(false)

  const batchSaveState = ref<BatchSaveState>({
    isInProgress: false,
    totalCount: 0,
    currentIndex: 0,
    sessionId: null,
  })

  let progressClearTimer: ReturnType<typeof setTimeout> | null = null
  let sessionLoadRequestId = 0

  const isBookshelfMode = computed(() => {
    return context.value.bookId !== null && context.value.chapterId !== null
  })

  const currentBookId = computed(() => context.value.bookId)
  const currentChapterId = computed(() => context.value.chapterId)

  const batchSaveProgress = computed(() => {
    if (!batchSaveState.value.isInProgress || batchSaveState.value.totalCount <= 0) {
      return 0
    }

    return Math.round((batchSaveState.value.currentIndex / batchSaveState.value.totalCount) * 100)
  })

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

  function setBookChapterContext(
    bookId: string,
    chapterId: string,
    bookTitle: string,
    chapterTitle: string
  ): void {
    setContext(bookId, chapterId, bookTitle, chapterTitle)
  }

  function clearContext(): void {
    context.value = {
      bookId: null,
      chapterId: null,
      bookTitle: null,
      chapterTitle: null
    }
  }

  function parseContextFromUrl(searchParams: URLSearchParams): void {
    const bookId = searchParams.get('book')
    const chapterId = searchParams.get('chapter')

    if (bookId && chapterId) {
      setContext(bookId, chapterId)
    }
  }

  function setSessionName(name: string | null): void {
    currentSessionName.value = name
  }

  function clearSessionName(): void {
    currentSessionName.value = null
  }

  function setSessionList(list: SessionListItem[]): void {
    sessionList.value = list
  }

  function addToSessionList(session: SessionListItem): void {
    const existingIndex = sessionList.value.findIndex(s => s.name === session.name)
    if (existingIndex >= 0) {
      sessionList.value[existingIndex] = session
    } else {
      sessionList.value.unshift(session)
    }
  }

  function removeFromSessionList(name: string): void {
    const index = sessionList.value.findIndex(s => s.name === name)
    if (index >= 0) {
      sessionList.value.splice(index, 1)
    }
  }

  function renameInSessionList(currentName: string, newName: string): void {
    const session = sessionList.value.find(s => s.name === currentName)
    if (session) {
      session.name = newName
    }
  }

  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  function setSaving(saving: boolean): void {
    isSaving.value = saving
  }

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

  function isActiveSessionLoad(requestId: number): boolean {
    return requestId === sessionLoadRequestId
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

  function reset(): void {
    sessionLoadRequestId += 1
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

  async function imageUrlToBase64(url: string | null): Promise<string | null> {
    if (!url || typeof url !== 'string') return null
    if (url.startsWith('data:')) return url
    if (!url.startsWith('/api/')) return null

    try {
      const response = await fetch(url)
      if (!response.ok) return null

      const blob = await response.blob()
      return await readBlobAsDataUrl(blob)
    } catch {
      return null
    }
  }

  async function convertImagesToBase64(
    images: ImageData[],
    progressCallback?: (current: number, total: number) => void
  ): Promise<void> {
    const total = images.length

    for (let i = 0; i < total; i++) {
      const img = images[i]
      if (!img) continue
      if (progressCallback) progressCallback(i + 1, total)

      if (img.originalDataURL && img.originalDataURL.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.originalDataURL)
        if (base64) img.originalDataURL = base64
      }

      if (img.translatedDataURL && img.translatedDataURL.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.translatedDataURL)
        if (base64) img.translatedDataURL = base64
      }

      if (img.cleanImageData && img.cleanImageData.startsWith('/api/')) {
        const base64 = await imageUrlToBase64(img.cleanImageData)
        if (base64) {
          img.cleanImageData = base64.replace(/^data:image\/\w+;base64,/, '')
        }
      }
    }
  }

  async function loadSessionByPath(sessionPath: string): Promise<boolean> {
    const requestId = ++sessionLoadRequestId
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
      if (!isActiveSessionLoad(requestId)) return false

      if (!response.success || !response.session) {
        throw new Error(response.error || '加载会话失败')
      }

      const sessionData = response.session

      if (sessionData.images && sessionData.images.length > 0) {
        const images = hydrateSessionImages(sessionData.images)

        if (images.length > 0) {
          if (!isActiveSessionLoad(requestId)) return false
          loadingProgress.value = { current: 0, total: images.length, message: '正在加载图片...' }

          await convertImagesToBase64(images, (current, total) => {
            if (isActiveSessionLoad(requestId)) {
              loadingProgress.value = { current, total, message: `加载图片 ${current}/${total}...` }
            }
          })

          if (!isActiveSessionLoad(requestId)) return false
          loadingProgress.value = { current: images.length, total: images.length, message: '加载完成' }

          scheduleLoadingProgressClear(500)
        }

        if (!isActiveSessionLoad(requestId)) return false
        imageStore.setImages(images)

        let newIndex = 0
        if (typeof sessionData.currentImageIndex === 'number') {
          newIndex = sessionData.currentImageIndex
          if (newIndex >= images.length || newIndex < 0) {
            newIndex = images.length > 0 ? 0 : -1
          }
        }
        imageStore.setCurrentImageIndex(newIndex)

        // 无气泡状态时使用本地清空，保留 null 和 [] 的语义区分。
        const currentImage = images[newIndex]
        if (currentImage && currentImage.bubbleStates && currentImage.bubbleStates.length > 0) {
          bubbleStore.setBubbles(currentImage.bubbleStates, true)
        } else {
          bubbleStore.clearBubblesLocal()
        }

      }

      const uiSettings = sessionData.ui_settings
      if (uiSettings) {
        if (!isActiveSessionLoad(requestId)) return false
        applySessionUiSettings(uiSettings, settingsStore)
      }

      if (!isActiveSessionLoad(requestId)) return false
      setSessionName(sessionPath)

      return true
    } catch (e) {
      if (!isActiveSessionLoad(requestId)) return false
      const errorMsg = e instanceof Error ? e.message : '加载会话失败'
      setError(errorMsg)
      clearProgressClearTimer()
      throw e
    } finally {
      if (isActiveSessionLoad(requestId)) {
        setLoading(false)
      }
    }
  }

  async function saveChapterSession(bookId: string, chapterId: string): Promise<boolean> {
    if (!bookId || !chapterId) {
      return false
    }

    const { useSettingsStore } = await import('@/stores/settings')
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()

    const allImages = Array.isArray(imageStore.images) ? imageStore.images : []

    if (!allImages || allImages.length === 0) {
      return false
    }

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
      clearProgressClearTimer()
      clearLoadingProgress()
      return false
    } finally {
      setSaving(false)
    }
  }

  return {
    currentSessionName,
    context,
    sessionList,
    isLoading,
    isSaving,
    error,
    loadingProgress,
    batchSaveState,

    isBookshelfMode,
    currentBookId,
    currentChapterId,
    batchSaveProgress,

    setContext,
    clearContext,
    parseContextFromUrl,

    setSessionName,
    clearSessionName,

    setSessionList,
    addToSessionList,
    removeFromSessionList,
    renameInSessionList,

    setLoading,
    setSaving,
    setError,
    startBatchSave,
    updateBatchSaveProgress,
    completeBatchSave,
    createSessionData,

    imageUrlToBase64,

    saveChapterSession,
    loadSessionByPath,
    setBookChapterContext,

    reset
  }
})
