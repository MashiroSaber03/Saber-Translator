import { getCurrentInstance, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useEditMode } from '@/composables/useEditMode'
import { showToast } from '@/utils/toast'
import type { V2Font } from '@/api/v2/settings'
import {
  getPageDocument,
  getTranslationBootstrap,
  updateChapterSettingsMemory,
  updateLastVisitedPage,
} from '@/api/v2/content'
import { restoreTranslationFromBootstrap } from '@/composables/useTranslationPipeline'
import {
  pageSummaryToImage,
} from '@/adapters/v2ContentAdapter'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import {
  flushPageDocument,
  isPageDocumentRegistered,
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'

export interface InitState {
  isInitializing: boolean
  isInitialized: boolean
  initError: string | null
  fontList: V2Font[]
  promptNames: string[]
  textboxPromptNames: string[]
  currentBookId: string | null
  currentChapterId: string | null
  currentBookTitle: string | null
  currentChapterTitle: string | null
  isBookshelfMode: boolean
}

export function useTranslateInit() {
  const route = useRoute()
  const settingsStore = useSettingsStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const editMode = useEditMode()

  const isInitializing = ref(false)
  const isInitialized = ref(false)
  const initError = ref<string | null>(null)
  const fontList = ref<V2Font[]>([])
  const promptNames = ref<string[]>([])
  const textboxPromptNames = ref<string[]>([])
  const currentBookId = ref<string | null>(null)
  const currentChapterId = ref<string | null>(null)
  const currentBookTitle = ref<string | null>(null)
  const currentChapterTitle = ref<string | null>(null)
  const isBookshelfMode = ref(false)
  const isSwitchingImage = ref(false)
  let switchImageFlagTimer: ReturnType<typeof setTimeout> | null = null
  let isOwnerAlive = true
  let bookContextRequestId = 0
  let pageDocumentRequestId = 0
  let pageDocumentAbortController: AbortController | null = null
  let navigationRevision = 0
  let navigationWriteChain = Promise.resolve()
  let settingsMemoryChapterId: string | null = null
  let settingsMemoryRevision = 0
  let lastSettingsMemoryFingerprint = ''
  let settingsMemoryWriteTimer: ReturnType<typeof setTimeout> | null = null
  let pendingSettingsMemoryWrite: {
    chapterId: string
    payload: Record<string, unknown>
  } | null = null
  let isWritingSettingsMemory = false

  function clearSwitchImageFlagTimer(): void {
    if (switchImageFlagTimer) {
      clearTimeout(switchImageFlagTimer)
      switchImageFlagTimer = null
    }
  }

  function resetSwitchImageFlag(): void {
    clearSwitchImageFlagTimer()
    isSwitchingImage.value = false
  }

  function markOwnerUnmounted(): void {
    if (settingsMemoryWriteTimer) {
      clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = null
      void flushSettingsMemoryWrite()
    }
    isOwnerAlive = false
    bookContextRequestId += 1
    pageDocumentRequestId += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    resetSwitchImageFlag()
    settingsStore.clearChapterWorkState(settingsMemoryChapterId ?? undefined)
  }

  if (getCurrentInstance()) {
    onUnmounted(markOwnerUnmounted)
  }

  watch(
    () => settingsStore.chapterWorkStatePayload(),
    payload => {
      const chapterId = settingsMemoryChapterId
      if (!chapterId) return
      const fingerprint = JSON.stringify(payload)
      if (fingerprint === lastSettingsMemoryFingerprint) return
      pendingSettingsMemoryWrite = { chapterId, payload }
      if (settingsMemoryWriteTimer) clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = setTimeout(() => {
        settingsMemoryWriteTimer = null
        void flushSettingsMemoryWrite()
      }, 400)
    },
    { deep: true },
  )

  async function initializeApp(force: boolean = false): Promise<void> {
    // SPA 场景下重新进入翻译页时，需要重新加载书籍/章节上下文。
    if (!force && (isInitializing.value || isInitialized.value)) {
      await initializeBookChapterContext()
      return
    }

    isInitializing.value = true
    initError.value = null

    try {
      settingsStore.initSettings()
      await initializeBookChapterContext()

      isInitialized.value = true
    } catch (error) {
      initError.value = error instanceof Error ? error.message : '初始化失败'
    } finally {
      isInitializing.value = false
    }
  }

  async function initializeBookChapterContext(): Promise<void> {
    const requestId = ++bookContextRequestId
    const bookId = route.query.book as string | undefined
    const chapterId = route.query.chapter as string | undefined

    try {
      const bootstrap = await getTranslationBootstrap({
        bookId: bookId && chapterId ? bookId : undefined,
        chapterId: bookId && chapterId ? chapterId : undefined,
      })
      if (!isOwnerAlive || requestId !== bookContextRequestId) return

      if (!settingsStore.hydrateFromBackendDocument(bootstrap.settings)) {
        throw new Error(settingsStore.backendError || '后端设置加载失败')
      }
      settingsStore.clearChapterWorkState(settingsMemoryChapterId ?? undefined)
      settingsMemoryChapterId = bootstrap.chapter.id
      settingsMemoryRevision = bootstrap.chapter.settingsMemoryRevision
      if (!settingsStore.hydrateChapterWorkState(
        bootstrap.chapter.id,
        bootstrap.chapter.settingsMemory,
      )) {
        throw new Error('后端章节工作态设置格式无效')
      }
      lastSettingsMemoryFingerprint = JSON.stringify(
        settingsStore.chapterWorkStatePayload(),
      )
      pendingSettingsMemoryWrite = null
      if (settingsMemoryWriteTimer) {
        clearTimeout(settingsMemoryWriteTimer)
        settingsMemoryWriteTimer = null
      }
      settingsStore.hydrateResourceCatalogs(bootstrap.fonts, bootstrap.prompts)
      fontList.value = bootstrap.fonts
      const translationPrompts = bootstrap.prompts.filter(
        prompt => prompt.type === 'translate',
      )
      const textboxPrompts = bootstrap.prompts.filter(
        prompt => prompt.type === 'textbox',
      )
      promptNames.value = translationPrompts.map(prompt => prompt.name)
      textboxPromptNames.value = textboxPrompts.map(prompt => prompt.name)
      const translateFactory = translationPrompts.find(prompt => prompt.isFactoryDefault)
      if (!settingsStore.settings.translatePrompt && translateFactory) {
        settingsStore.setTranslatePrompt(translateFactory.content)
      }
      const textboxFactory = textboxPrompts.find(prompt => prompt.isFactoryDefault)
      if (!settingsStore.settings.textboxPrompt && textboxFactory) {
        settingsStore.setTextboxPrompt(textboxFactory.content)
      }

      currentBookId.value = bootstrap.book.id
      currentChapterId.value = bootstrap.chapter.id
      currentBookTitle.value = bootstrap.book.title
      currentChapterTitle.value = bootstrap.chapter.title
      isBookshelfMode.value = bootstrap.book.kind === 'library'
      bookTranslationConstraintsStore.loadBookConstraints(
        bootstrap.book.id,
        bootstrap.constraints.payload as Partial<BookTranslationConstraints>,
      )
      imageStore.setImages(bootstrap.pages.items.map(pageSummaryToImage))
      restoreTranslationFromBootstrap(bootstrap.activeJobs, imageStore)
      navigationRevision = bootstrap.navigation.revision

      const lastVisitedIndex = bootstrap.navigation.lastVisitedPageId
        ? imageStore.images.findIndex(image => image.id === bootstrap.navigation.lastVisitedPageId)
        : -1
      const initialIndex = lastVisitedIndex >= 0 ? lastVisitedIndex : 0
      if (imageStore.imageCount > 0) {
        await switchImage(initialIndex, false)
      } else {
        bubbleStore.clearBubblesLocal()
      }

      if (typeof document !== 'undefined') {
        document.title = `${bootstrap.chapter.title} - ${bootstrap.book.title} - Saber-Translator`
      }
    } catch {
      if (!isOwnerAlive || requestId !== bookContextRequestId) return
      bookTranslationConstraintsStore.resetBookConstraints()
      showToast('加载后端章节数据失败', 'error')
    }
  }

  async function flushSettingsMemoryWrite(): Promise<void> {
    if (isWritingSettingsMemory) return
    isWritingSettingsMemory = true
    try {
      while (pendingSettingsMemoryWrite) {
        const pending = pendingSettingsMemoryWrite
        pendingSettingsMemoryWrite = null
        if (pending.chapterId !== settingsMemoryChapterId) continue
        try {
          const updated = await updateChapterSettingsMemory(
            pending.chapterId,
            pending.payload,
            settingsMemoryRevision,
          )
          settingsMemoryRevision = updated.revision
          lastSettingsMemoryFingerprint = JSON.stringify(updated.payload)
        } catch (error) {
          showToast(
            `章节工作态设置未保存：${error instanceof Error ? error.message : '未知错误'}`,
            'warning',
          )
        }
      }
    } finally {
      isWritingSettingsMemory = false
      if (pendingSettingsMemoryWrite && !settingsMemoryWriteTimer) {
        settingsMemoryWriteTimer = setTimeout(() => {
          settingsMemoryWriteTimer = null
          void flushSettingsMemoryWrite()
        }, 0)
      }
    }
  }

  function queueLastVisitedPageWrite(chapterId: string, pageId: string): void {
    navigationWriteChain = navigationWriteChain
      .then(async () => {
        if (!isOwnerAlive || currentChapterId.value !== chapterId) return
        const updated = await updateLastVisitedPage(
          chapterId,
          pageId,
          navigationRevision,
        )
        if (isOwnerAlive && currentChapterId.value === chapterId) {
          navigationRevision = updated.revision
        }
      })
      .catch(error => {
        if (!isOwnerAlive || currentChapterId.value !== chapterId) return
        showToast(
          `记录最后访问页失败：${error instanceof Error ? error.message : '未知错误'}`,
          'warning',
        )
      })
  }

  async function switchImage(
    index: number,
    persistNavigation: boolean = true,
  ): Promise<void> {
    if (index < 0 || index >= imageStore.imageCount) {
      return
    }

    clearSwitchImageFlagTimer()
    isSwitchingImage.value = true

    if (editMode.isActive.value) {
      editMode.exitEditModeWithoutRender()
    }

    const currentImage = imageStore.currentImage
    if (
      currentImage
      && index !== imageStore.currentImageIndex
      && currentImage.documentRevision !== undefined
      && isPageDocumentRegistered(currentImage.id)
    ) {
      imageStore.updateCurrentImageProperty('bubbleStates', bubbleStore.bubbles)
      queuePageDocumentSave(
        currentImage.id,
        currentImage.documentRevision,
        bubbleStore.bubbles,
      )
      try {
        await flushPageDocument(currentImage.id)
      } catch (error) {
        resetSwitchImageFlag()
        showToast(
          `当前页写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
          'error',
        )
        return
      }
    }

    imageStore.setCurrentImageIndex(index)
    const newImage = imageStore.currentImage

    if (!newImage) {
      resetSwitchImageFlag()
      return
    }
    if (persistNavigation && currentChapterId.value) {
      queueLastVisitedPageWrite(currentChapterId.value, newImage.id)
    }

    bubbleStore.clearBubblesLocal()
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = new AbortController()
    const pageRequestId = ++pageDocumentRequestId
    const pageId = newImage.id
    void getPageDocument(pageId, pageDocumentAbortController.signal)
      .then(document => {
        if (
          !isOwnerAlive
          || pageRequestId !== pageDocumentRequestId
          || imageStore.currentImage?.id !== pageId
        ) return
        const bubbles = registerPageDocument(document)
        imageStore.updateCurrentImage({
          bubbleStates: bubbles,
          documentRevision: document.documentRevision,
          hasUnsavedChanges: false,
        })
        bubbleStore.setBubbles(bubbles, true)
      })
      .catch(error => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        if (pageRequestId === pageDocumentRequestId) {
          showToast('加载当前页编辑数据失败', 'error')
        }
      })
      .finally(() => {
        if (pageRequestId !== pageDocumentRequestId) return
        switchImageFlagTimer = setTimeout(() => {
          switchImageFlagTimer = null
          isSwitchingImage.value = false
        }, 100)
      })
  }

  async function goToPrevious(): Promise<void> {
    if (imageStore.canGoPrevious) {
      await switchImage(imageStore.currentImageIndex - 1)
    }
  }

  async function goToNext(): Promise<void> {
    if (imageStore.canGoNext) {
      await switchImage(imageStore.currentImageIndex + 1)
    }
  }

  function setupAutoInit(): void {
    onMounted(async () => {
      await initializeApp()
    })
  }

  return {
    isInitializing,
    isInitialized,
    initError,
    fontList,
    promptNames,
    textboxPromptNames,
    currentBookId,
    currentChapterId,
    currentBookTitle,
    currentChapterTitle,
    isBookshelfMode,
    isSwitchingImage,

    initializeApp,
    initializeBookChapterContext,

    switchImage,
    goToPrevious,
    goToNext,
    editMode,

    setupAutoInit
  }
}
