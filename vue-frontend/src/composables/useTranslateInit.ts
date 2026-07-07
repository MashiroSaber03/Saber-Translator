import { getCurrentInstance, onMounted, onUnmounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useEditMode } from '@/composables/useEditMode'
import { showToast } from '@/utils/toast'
import { getFontList, getPrompts, getTextboxPrompts } from '@/api/config'
import { reloadTextStyleDefaultsFromBackend } from '@/defaults/textStyleDefaults'
import { getBookDetail } from '@/api/bookshelf'
import { cleanupGpu } from '@/api/system'

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
  const sessionStore = useSessionStore()
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
    resetSwitchImageFlag()
  }

  function isActiveBookContextRequest(
    requestId: number,
    bookId: string,
    chapterId: string
  ): boolean {
    return (
      isOwnerAlive &&
      requestId === bookContextRequestId &&
      route.query.book === bookId &&
      route.query.chapter === chapterId
    )
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
      await initializeGpu()
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

  async function initializeGpu(): Promise<void> {
    try {
      const response = await cleanupGpu()
      if (!response.success) {
        showToast(response.error || 'GPU 清理失败', 'warning')
      }
    } catch {
      // GPU cleanup is a best-effort startup task and should not block the page.
    }
  }

  async function initializeBookChapterContext(): Promise<void> {
    const requestId = ++bookContextRequestId
    const bookId = route.query.book as string | undefined
    const chapterId = route.query.chapter as string | undefined

    if (!bookId || !chapterId) {
      isBookshelfMode.value = false
      currentBookId.value = null
      currentChapterId.value = null
      currentBookTitle.value = null
      currentChapterTitle.value = null
      sessionStore.clearContext()
      bookTranslationConstraintsStore.resetBookConstraints()
      return
    }

    isBookshelfMode.value = true
    currentBookId.value = bookId
    currentChapterId.value = chapterId

    try {
      const bookResponse = await getBookDetail(bookId)
      if (!isActiveBookContextRequest(requestId, bookId, chapterId)) {
        return
      }

      if (!bookResponse.success || !bookResponse.book) {
        bookTranslationConstraintsStore.resetBookConstraints()
        showToast('书籍不存在', 'warning')
        return
      }

      const book = bookResponse.book
      const chapter = book.chapters?.find(c => c.id === chapterId)

      if (!chapter) {
        bookTranslationConstraintsStore.resetBookConstraints()
        showToast('章节不存在', 'warning')
        return
      }

      currentBookTitle.value = book.title
      currentChapterTitle.value = chapter.title
      bookTranslationConstraintsStore.loadBookConstraints(bookId, book.translation_constraints)

      sessionStore.setBookChapterContext(bookId, chapterId, book.title, chapter.title)

      if (typeof document !== 'undefined') {
        document.title = `${chapter.title} - ${book.title} - Saber-Translator`
      }

      const hasData = chapter.page_count && chapter.page_count > 0
      if (chapter.session_path && hasData) {
        try {
          if (!isActiveBookContextRequest(requestId, bookId, chapterId)) {
            return
          }
          await sessionStore.loadSessionByPath(chapter.session_path)
          if (!isActiveBookContextRequest(requestId, bookId, chapterId)) {
            return
          }
          showToast(`已加载章节: ${chapter.title}`, 'success')
        } catch {
          // 会话不可用时保持当前章节上下文，等待用户重新保存。
        }
      }

    } catch {
      if (!isActiveBookContextRequest(requestId, bookId, chapterId)) {
        return
      }
      bookTranslationConstraintsStore.resetBookConstraints()
      showToast('加载书籍信息失败', 'error')
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

    // 使用 clearBubblesLocal 保持 null 和 [] 的语义区分：
    //   - bubbleStates === null: 从未处理过，翻译时应自动检测
    //   - bubbleStates === []: 用户主动清空，翻译时应跳过（避免"框复活"）
    if (newImage.bubbleStates && newImage.bubbleStates.length > 0) {
      bubbleStore.setBubbles(newImage.bubbleStates, true)
    } else {
      // 仅清除本地状态，避免把 null 错误地写成 []。
      bubbleStore.clearBubblesLocal()
    }

    // 当 currentImage 变化时，watch 会调用 syncImageToSidebar

    switchImageFlagTimer = setTimeout(() => {
      switchImageFlagTimer = null
      isSwitchingImage.value = false
    }, 100)
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
