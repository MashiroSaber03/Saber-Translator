import { getCurrentInstance, nextTick, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useSettingsStore } from '@/stores/settings'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { showToast } from '@/utils/toast'
import { isRequestCanceled } from '@/api/client'
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
import {
  flushPageDocument,
  isPageDocumentRegistered,
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { parseCompleteTextStyleSettings } from '@/defaults/textStyleDefaults'

export type TranslationRouteContext =
  | { kind: 'quick' }
  | { kind: 'library'; bookId: string; chapterId: string }

export function parseTranslationRouteContext(
  query: { book?: unknown; chapter?: unknown },
): TranslationRouteContext | null {
  if (query.book === undefined && query.chapter === undefined) {
    return { kind: 'quick' }
  }
  if (
    typeof query.book === 'string'
    && query.book.length > 0
    && typeof query.chapter === 'string'
    && query.chapter.length > 0
  ) {
    return {
      kind: 'library',
      bookId: query.book,
      chapterId: query.chapter,
    }
  }
  return null
}

export function useTranslateInit() {
  const route = useRoute()
  const settingsStore = useSettingsStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()

  const isInitializing = ref(false)
  const isInitialized = ref(false)
  const isContextReady = ref(false)
  const fontList = ref<V2Font[]>([])
  const currentBookId = ref<string | null>(null)
  const currentChapterId = ref<string | null>(null)
  const currentBookTitle = ref<string | null>(null)
  const currentChapterTitle = ref<string | null>(null)
  const isBookshelfMode = ref(false)
  const isSwitchingImage = ref(false)
  let isOwnerAlive = true
  let bookContextRequestId = 0
  let pageDocumentRequestId = 0
  let pageDocumentAbortController: AbortController | null = null
  let navigationWriteChain = Promise.resolve()
  let settingsMemoryChapterId: string | null = null
  let settingsMemoryRevision = 0
  let lastSettingsMemoryFingerprint = ''
  let settingsMemoryWriteTimer: ReturnType<typeof setTimeout> | null = null
  let pendingSettingsMemoryWrite: {
    chapterId: string
    fingerprint: string
    payload: Record<string, unknown>
  } | null = null
  let activeSettingsMemoryWrite: {
    chapterId: string
    fingerprint: string
  } | null = null
  let settingsMemoryWritePromise: Promise<boolean> | null = null

  function clearLoadedChapterContext(): void {
    isContextReady.value = false
    pageDocumentRequestId += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    if (settingsMemoryWriteTimer) {
      clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = null
    }
    settingsStore.clearChapterWorkState(settingsMemoryChapterId ?? undefined)
    settingsMemoryChapterId = null
    settingsMemoryRevision = 0
    lastSettingsMemoryFingerprint = ''
    pendingSettingsMemoryWrite = null
    currentBookId.value = null
    currentChapterId.value = null
    currentBookTitle.value = null
    currentChapterTitle.value = null
    isBookshelfMode.value = false
    bookTranslationConstraintsStore.resetBookConstraints()
    imageStore.clearImages()
    bubbleStore.clearBubblesLocal()
  }

  function markOwnerUnmounted(): void {
    if (settingsMemoryWriteTimer) {
      clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = null
    }
    void flushChapterWorkState()
    isOwnerAlive = false
    isContextReady.value = false
    bookContextRequestId += 1
    pageDocumentRequestId += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    isSwitchingImage.value = false
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
      if (!stageChapterWorkState(chapterId, payload)) {
        if (settingsMemoryWriteTimer) {
          clearTimeout(settingsMemoryWriteTimer)
          settingsMemoryWriteTimer = null
        }
        return
      }
      if (settingsMemoryWriteTimer) clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = setTimeout(() => {
        settingsMemoryWriteTimer = null
        void flushSettingsMemoryWrite()
      }, 400)
    },
    { deep: true },
  )

  function stageChapterWorkState(
    chapterId: string,
    payload: Record<string, unknown>,
  ): boolean {
    if (chapterId !== settingsMemoryChapterId) return false
    const fingerprint = JSON.stringify(payload)
    const baselineFingerprint = (
      activeSettingsMemoryWrite?.chapterId === chapterId
        ? activeSettingsMemoryWrite.fingerprint
        : lastSettingsMemoryFingerprint
    )
    if (fingerprint === baselineFingerprint) {
      if (pendingSettingsMemoryWrite?.chapterId === chapterId) {
        pendingSettingsMemoryWrite = null
      }
      return false
    }
    if (
      pendingSettingsMemoryWrite?.chapterId === chapterId
      && pendingSettingsMemoryWrite.fingerprint === fingerprint
    ) return true
    pendingSettingsMemoryWrite = { chapterId, fingerprint, payload }
    return true
  }

  async function initializeApp(): Promise<boolean> {
    // SPA 场景下重新进入翻译页时，需要重新加载书籍/章节上下文。
    if (isInitializing.value || isInitialized.value) {
      return initializeBookChapterContext()
    }

    isInitializing.value = true

    try {
      settingsStore.initSettings()
      const initialized = await initializeBookChapterContext()
      isInitialized.value = initialized
      return initialized
    } finally {
      isInitializing.value = false
    }
  }

  async function initializeBookChapterContext(): Promise<boolean> {
    isContextReady.value = false
    const requestId = ++bookContextRequestId
    pageDocumentRequestId += 1
    pageDocumentAbortController?.abort()
    pageDocumentAbortController = null
    isSwitchingImage.value = false
    const routeContext = parseTranslationRouteContext(route.query)
    if (!routeContext) {
      clearLoadedChapterContext()
      showToast('翻译页面地址无效：book 与 chapter 必须同时且各自只提供一个值', 'error')
      return false
    }

    try {
      if (
        settingsMemoryChapterId
        && !(await flushChapterWorkState())
      ) {
        throw new Error('当前章节工作态设置尚未写入后端')
      }
      const bootstrap = await getTranslationBootstrap({
        ...(routeContext.kind === 'library'
          ? {
              bookId: routeContext.bookId,
              chapterId: routeContext.chapterId,
            }
          : {}),
      })
      if (!isOwnerAlive || requestId !== bookContextRequestId) return false
      if (
        routeContext.kind === 'library'
          ? (
              bootstrap.book.kind !== 'library'
              || bootstrap.book.id !== routeContext.bookId
              || bootstrap.chapter.id !== routeContext.chapterId
            )
          : bootstrap.book.kind !== 'quick_workspace'
      ) {
        throw new Error('后端返回了其他翻译工作区的数据')
      }
      if (bootstrap.pages.nextCursor !== null) {
        throw new Error('后端翻译工作区页面列表不完整')
      }
      if (bootstrap.pages.items.some(page => page.chapterId !== bootstrap.chapter.id)) {
        throw new Error('后端翻译工作区包含其他章节的页面')
      }
      const pageIds = new Set(bootstrap.pages.items.map(page => page.id))
      if (
        bootstrap.navigation.lastVisitedPageId
        && !pageIds.has(bootstrap.navigation.lastVisitedPageId)
      ) {
        throw new Error('后端最后访问页不属于当前章节')
      }

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
        bootstrap.constraints.payload,
        bootstrap.constraints.revision,
      )
      imageStore.setImages(bootstrap.pages.items.map(pageSummaryToImage))
      restoreTranslationFromBootstrap(bootstrap.activeJobs, imageStore)

      const lastVisitedIndex = bootstrap.navigation.lastVisitedPageId
        ? imageStore.images.findIndex(image => image.id === bootstrap.navigation.lastVisitedPageId)
        : -1
      const initialIndex = lastVisitedIndex >= 0 ? lastVisitedIndex : 0
      if (imageStore.imageCount > 0) {
        const switched = await switchImage(initialIndex, false)
        if (!isOwnerAlive || requestId !== bookContextRequestId) return false
        if (!switched) {
          clearLoadedChapterContext()
          return false
        }
      } else {
        bubbleStore.clearBubblesLocal()
      }

      if (typeof document !== 'undefined') {
        document.title = `${bootstrap.chapter.title} - ${bootstrap.book.title} - Saber-Translator`
      }
      isContextReady.value = true
      return true
    } catch (error) {
      if (!isOwnerAlive || requestId !== bookContextRequestId) return false
      clearLoadedChapterContext()
      const message = error instanceof Error ? error.message : '未知错误'
      showToast(`加载后端章节数据失败：${message}`, 'error')
      return false
    }
  }

  function flushSettingsMemoryWrite(): Promise<boolean> {
    if (settingsMemoryWritePromise) return settingsMemoryWritePromise
    settingsMemoryWritePromise = persistPendingSettingsMemory().finally(() => {
      settingsMemoryWritePromise = null
    })
    return settingsMemoryWritePromise
  }

  async function persistPendingSettingsMemory(): Promise<boolean> {
    while (pendingSettingsMemoryWrite) {
      const pending = pendingSettingsMemoryWrite
      pendingSettingsMemoryWrite = null
      if (pending.chapterId !== settingsMemoryChapterId) continue
      activeSettingsMemoryWrite = {
        chapterId: pending.chapterId,
        fingerprint: pending.fingerprint,
      }
      try {
        const updated = await updateChapterSettingsMemory(
          pending.chapterId,
          pending.payload,
          settingsMemoryRevision,
        )
        if (pending.chapterId === settingsMemoryChapterId) {
          settingsMemoryRevision = updated.revision
          lastSettingsMemoryFingerprint = JSON.stringify(updated.payload)
        }
      } catch (error) {
        if (
          pending.chapterId === settingsMemoryChapterId
          && !pendingSettingsMemoryWrite
        ) {
          pendingSettingsMemoryWrite = pending
        }
        showToast(
          `章节工作态设置未保存：${error instanceof Error ? error.message : '未知错误'}`,
          'warning',
        )
        return false
      } finally {
        activeSettingsMemoryWrite = null
      }
    }
    return true
  }

  async function flushChapterWorkState(): Promise<boolean> {
    const chapterId = settingsMemoryChapterId
    if (!chapterId) return true
    const payload = settingsStore.chapterWorkStatePayload()
    stageChapterWorkState(chapterId, payload)
    if (settingsMemoryWriteTimer) {
      clearTimeout(settingsMemoryWriteTimer)
      settingsMemoryWriteTimer = null
    }
    return flushSettingsMemoryWrite()
  }

  function queueLastVisitedPageWrite(chapterId: string, pageId: string): void {
    navigationWriteChain = navigationWriteChain
      .then(async () => {
        if (!isOwnerAlive || currentChapterId.value !== chapterId) return
        await updateLastVisitedPage(chapterId, pageId)
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
  ): Promise<boolean> {
    if (index < 0 || index >= imageStore.imageCount) {
      return false
    }

    const target = imageStore.images[index]
    if (!target) return false
    pageDocumentAbortController?.abort()
    const controller = new AbortController()
    pageDocumentAbortController = controller
    const pageRequestId = ++pageDocumentRequestId
    isSwitchingImage.value = true

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
        if (pageRequestId === pageDocumentRequestId) {
          showToast(
            `当前页写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
            'error',
          )
          await nextTick()
          if (pageRequestId === pageDocumentRequestId) {
            isSwitchingImage.value = false
          }
        }
        return false
      }
    }
    if (pageRequestId !== pageDocumentRequestId) return false

    const pageId = target.id
    try {
      const document = await getPageDocument(
        pageId,
        controller.signal,
      )
      if (
        !isOwnerAlive
        || pageRequestId !== pageDocumentRequestId
        || imageStore.images[index]?.id !== pageId
      ) return false
      if (document.pageId !== pageId || document.chapterId !== target.chapterId) {
        throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
      }
      const pageTextStyle = parseCompleteTextStyleSettings({
        ...document.pageStyleDefaults,
        ...(document.defaultFontId
          ? { fontFamily: document.defaultFontId }
          : {}),
      })
      const bubbles = registerPageDocument(document)
      imageStore.setCurrentImageIndex(index)
      if (imageStore.currentImage?.id !== pageId) return false
      imageStore.updateCurrentImage({
        ...pageTextStyle,
        bubbleStates: bubbles,
        documentRevision: document.documentRevision,
        hasUnsavedChanges: false,
      })
      settingsStore.updateTextStyle(pageTextStyle)
      bubbleStore.setBubbles(bubbles, true)
      if (persistNavigation && currentChapterId.value) {
        queueLastVisitedPageWrite(currentChapterId.value, pageId)
      }
      return true
    } catch (error) {
      if (isRequestCanceled(error)) return false
      if (pageRequestId === pageDocumentRequestId) {
        showToast(
          `加载当前页编辑数据失败：${error instanceof Error ? error.message : '未知错误'}`,
          'error',
        )
      }
      return false
    } finally {
      if (pageRequestId === pageDocumentRequestId) {
        await nextTick()
        if (pageRequestId === pageDocumentRequestId) {
          isSwitchingImage.value = false
        }
      }
    }
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

  return {
    fontList,
    currentBookId,
    currentChapterId,
    currentBookTitle,
    currentChapterTitle,
    isBookshelfMode,
    isContextReady,
    isSwitchingImage,

    initializeApp,
    initializeBookChapterContext,
    flushChapterWorkState,

    switchImage,
    goToPrevious,
    goToNext,
  }
}
