import { getCurrentInstance, onMounted, onUnmounted, ref } from 'vue'
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
} from '@/api/v2/content'
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
    isOwnerAlive = false
    bookContextRequestId += 1
    pageDocumentRequestId += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    resetSwitchImageFlag()
  }

  if (getCurrentInstance()) {
    onUnmounted(markOwnerUnmounted)
  }

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

      const lastVisitedIndex = bootstrap.navigation.lastVisitedPageId
        ? imageStore.images.findIndex(image => image.id === bootstrap.navigation.lastVisitedPageId)
        : -1
      const initialIndex = lastVisitedIndex >= 0 ? lastVisitedIndex : 0
      if (imageStore.imageCount > 0) {
        await switchImage(initialIndex)
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

  async function switchImage(index: number): Promise<void> {
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
