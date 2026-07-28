import { getCurrentInstance, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useEditMode } from '@/composables/useEditMode'
import { showToast } from '@/utils/toast'
import { getFontList, getPrompts, getTextboxPrompts } from '@/api/config'
import { reloadTextStyleDefaultsFromBackend } from '@/defaults/textStyleDefaults'
import {
  getPageDocument,
  getTranslationBootstrap,
} from '@/api/v2/content'
import {
  pageSummaryToImage,
} from '@/adapters/v2ContentAdapter'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import { registerPageDocument } from '@/services/pageDocumentPersistence'

import type { FontInfo } from '@/types/api'

export interface InitState {
  isInitializing: boolean
  isInitialized: boolean
  initError: string | null
  fontList: FontInfo[]
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
  const fontList = ref<FontInfo[]>([])
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
      await initializeSettings()
      await initializeFontList()
      await initializePromptSettings()
      await initializeTextboxPromptSettings()
      await initializeBookChapterContext()

      isInitialized.value = true
    } catch (error) {
      initError.value = error instanceof Error ? error.message : '初始化失败'
    } finally {
      isInitializing.value = false
    }
  }

  async function initializeSettings(): Promise<void> {
    await reloadTextStyleDefaultsFromBackend()
    settingsStore.initSettings()

    try {
      await settingsStore.loadFromBackend()
    } catch {
      // Backend settings are optional; the current browser settings remain active.
    }
  }

  async function initializeFontList(): Promise<void> {
    try {
      const response = await getFontList()
      if (response.fonts && response.fonts.length > 0) {
        fontList.value = response.fonts
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // Font discovery is optional during startup; users can still work with saved font settings.
    }
  }

  async function initializePromptSettings(): Promise<void> {
    try {
      const response = await getPrompts()
      if (response.prompt_names !== undefined) {
        promptNames.value = response.prompt_names || []

        if (!settingsStore.settings.translatePrompt && response.default_prompt_content) {
          settingsStore.setTranslatePrompt(response.default_prompt_content)
        }
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // Prompt discovery is optional during startup; existing prompt settings remain active.
    }
  }

  async function initializeTextboxPromptSettings(): Promise<void> {
    try {
      const response = await getTextboxPrompts()
      if (response.prompt_names !== undefined) {
        textboxPromptNames.value = response.prompt_names || []

        if (!settingsStore.settings.textboxPrompt && response.default_prompt_content) {
          settingsStore.setTextboxPrompt(response.default_prompt_content)
        }
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // Textbox prompt discovery is optional during startup; existing prompt settings remain active.
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
        switchImage(initialIndex)
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

  function switchImage(index: number): void {
    if (index < 0 || index >= imageStore.imageCount) {
      return
    }

    clearSwitchImageFlagTimer()
    isSwitchingImage.value = true

    if (editMode.isActive.value) {
      editMode.exitEditModeWithoutRender()
    }

    const currentImage = imageStore.currentImage
    if (currentImage && bubbleStore.bubbles.length > 0) {
      imageStore.updateCurrentImageProperty('bubbleStates', bubbleStore.bubbles)
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

  function goToPrevious(): void {
    if (imageStore.canGoPrevious) {
      switchImage(imageStore.currentImageIndex - 1)
    }
  }

  function goToNext(): void {
    if (imageStore.canGoNext) {
      switchImage(imageStore.currentImageIndex + 1)
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
    initializeSettings,
    initializeFontList,
    initializePromptSettings,
    initializeTextboxPromptSettings,
    initializeBookChapterContext,

    switchImage,
    goToPrevious,
    goToNext,
    editMode,

    setupAutoInit
  }
}
