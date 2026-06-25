/**
 * 翻译页面初始化组合式函数
 * 负责翻译页面启动初始化
 * 
 * 功能：
 * - 页面加载时初始化当前设置缓存
 * - 初始化提示词状态
 * - 初始化字体列表
 * - 初始化主题状态
 * - 初始化插件状态
 * - URL参数解析（书架模式自动加载章节会话）
 */

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

// ============================================================
// 类型定义
// ============================================================

/** 初始化状态 */
export interface InitState {
  /** 是否正在初始化 */
  isInitializing: boolean
  /** 初始化是否完成 */
  isInitialized: boolean
  /** 初始化错误信息 */
  initError: string | null
  /** 字体列表 */
  fontList: FontInfo[]
  /** 提示词名称列表 */
  promptNames: string[]
  /** 文本框提示词名称列表 */
  textboxPromptNames: string[]
  /** 当前书籍ID（书架模式） */
  currentBookId: string | null
  /** 当前章节ID（书架模式） */
  currentChapterId: string | null
  /** 当前书籍标题 */
  currentBookTitle: string | null
  /** 当前章节标题 */
  currentChapterTitle: string | null
  /** 是否为书架模式 */
  isBookshelfMode: boolean
}

// ============================================================
// 组合式函数
// ============================================================

/**
 * 翻译页面初始化组合式函数
 */
export function useTranslateInit() {
  const route = useRoute()
  const settingsStore = useSettingsStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()
  const imageStore = useImageStore()
  const sessionStore = useSessionStore()
  const bubbleStore = useBubbleStore()
  const editMode = useEditMode()

  // ============================================================
  // 状态定义
  // ============================================================

  /** 是否正在初始化 */
  const isInitializing = ref(false)

  /** 初始化是否完成 */
  const isInitialized = ref(false)

  /** 初始化错误信息 */
  const initError = ref<string | null>(null)

  /** 字体列表 */
  const fontList = ref<FontInfo[]>([])

  /** 提示词名称列表 */
  const promptNames = ref<string[]>([])

  /** 文本框提示词名称列表 */
  const textboxPromptNames = ref<string[]>([])

  /** 当前书籍ID（书架模式） */
  const currentBookId = ref<string | null>(null)

  /** 当前章节ID（书架模式） */
  const currentChapterId = ref<string | null>(null)

  /** 当前书籍标题 */
  const currentBookTitle = ref<string | null>(null)

  /** 当前章节标题 */
  const currentChapterTitle = ref<string | null>(null)

  /** 是否为书架模式 */
  const isBookshelfMode = ref(false)
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
    window._isChangingFromSwitchImage = false
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

  // ============================================================
  // 初始化方法
  // ============================================================

  /**
   * 初始化应用
   * 负责翻译页面启动初始化
   * 
   * @param force - 是否强制重新初始化（用于 SPA 场景下重新进入页面）
   */
  async function initializeApp(force: boolean = false): Promise<void> {
    // SPA 场景下重新进入翻译页时，需要重新加载书籍/章节上下文。
    if (!force && (isInitializing.value || isInitialized.value)) {
      // 即使跳过完整初始化，也需要重新处理 URL 参数（书架模式）
      await initializeBookChapterContext()
      return
    }

    isInitializing.value = true
    initError.value = null

    try {
      // 1. 初始化设置（后端配置覆盖当前浏览器缓存）
      await initializeSettings()

      // 2. 初始化字体列表
      await initializeFontList()

      // 3. 初始化提示词设置
      await initializePromptSettings()
      await initializeTextboxPromptSettings()

      // 4. 清理 GPU 资源（确保显存状态干净）
      await initializeGpu()

      // 5. 处理书籍/章节 URL 参数
      await initializeBookChapterContext()

      isInitialized.value = true
    } catch (error) {
      initError.value = error instanceof Error ? error.message : '初始化失败'
    } finally {
      isInitializing.value = false
    }
  }

  /**
   * 初始化设置
   * 优先从后端加载设置（config/user_settings.json）
   * 后端无数据时保留当前浏览器设置缓存
   */
  async function initializeSettings(): Promise<void> {
    // 先重新拉取一次最新的文字样式默认值
    await reloadTextStyleDefaultsFromBackend()

    // 先读取当前浏览器设置缓存。
    settingsStore.initSettings()

    // 尝试从后端加载设置（会覆盖浏览器缓存中的值）
    try {
      await settingsStore.loadFromBackend()
    } catch {
      // Backend settings are optional; the current browser settings remain active.
    }
  }

  /**
   * 初始化字体列表
   * 从后端获取系统字体列表
   */
  async function initializeFontList(): Promise<void> {
    try {
      const response = await getFontList()
      // 后端 API 直接返回 { fonts: [...] }，不包含 success 字段
      if (response.fonts && response.fonts.length > 0) {
        fontList.value = response.fonts
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // 字体列表获取失败不阻止初始化
    }
  }

  /**
   * 初始化翻译提示词设置
   * 从后端获取提示词列表和默认内容
   */
  async function initializePromptSettings(): Promise<void> {
    try {
      const response = await getPrompts()
      // 后端 API 直接返回 { prompt_names: [...], default_prompt_content: "..." }
      if (response.prompt_names !== undefined) {
        promptNames.value = response.prompt_names || []

        // 如果当前没有设置提示词，使用默认提示词
        if (!settingsStore.settings.translatePrompt && response.default_prompt_content) {
          settingsStore.setTranslatePrompt(response.default_prompt_content)
        }
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // 提示词获取失败不阻止初始化
    }
  }

  /**
   * 初始化文本框提示词设置
   * 从后端获取文本框提示词列表和默认内容
   */
  async function initializeTextboxPromptSettings(): Promise<void> {
    try {
      const response = await getTextboxPrompts()
      // 后端 API 直接返回 { prompt_names: [...], default_prompt_content: "..." }
      if (response.prompt_names !== undefined) {
        textboxPromptNames.value = response.prompt_names || []

        // 如果当前没有设置文本框提示词，使用默认提示词
        if (!settingsStore.settings.textboxPrompt && response.default_prompt_content) {
          settingsStore.setTextboxPrompt(response.default_prompt_content)
        }
      } else if (response.error) {
        showToast(response.error, 'warning')
      }
    } catch {
      // 提示词获取失败不阻止初始化
    }
  }

  /**
   * 初始化 GPU 资源
   * 清理显存并卸载已加载的模型，确保 GPU 状态干净
   */
  async function initializeGpu(): Promise<void> {
    try {
      const response = await cleanupGpu()
      if (!response.success) {
        showToast(response.error || 'GPU 清理失败', 'warning')
      }
    } catch {
      // GPU 清理失败不阻止初始化
    }
  }


  /**
   * 初始化书籍/章节上下文
   * 从 URL 参数中读取 book 和 chapter，加载对应的会话数据
   */
  async function initializeBookChapterContext(): Promise<void> {
    const requestId = ++bookContextRequestId
    const bookId = route.query.book as string | undefined
    const chapterId = route.query.chapter as string | undefined

    if (!bookId || !chapterId) {
      isBookshelfMode.value = false
      // 进入独立模式时清空书籍/章节上下文，避免跨会话残留
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
      // 获取书籍和章节信息
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

      // 设置书籍/章节上下文
      currentBookTitle.value = book.title
      currentChapterTitle.value = chapter.title
      bookTranslationConstraintsStore.loadBookConstraints(bookId, book.translation_constraints)

      // 更新 sessionStore 的上下文
      sessionStore.setBookChapterContext(bookId, chapterId, book.title, chapter.title)

      // 更新页面标题
      if (typeof document !== 'undefined') {
        document.title = `${chapter.title} - ${book.title} - Saber-Translator`
      }

      // 尝试加载章节的会话数据（仅当章节有已保存的图片时才尝试加载）
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

  // ============================================================
  // 图片切换逻辑
  // ============================================================

  /**
   * 切换显示的图片
   * 切换当前显示图片并同步页面状态
   * @param index - 要显示的图片索引
   */
  function switchImage(index: number): void {
    if (index < 0 || index >= imageStore.imageCount) {
      return
    }

    // 设置全局标记，表示当前正在进行切换图片操作
    // 这个标记用于避免在切换图片时触发不必要的重渲染
    clearSwitchImageFlagTimer()
    window._isChangingFromSwitchImage = true

    // 如果在编辑模式，退出编辑模式但不触发重渲染
    if (editMode.isActive.value) {
      editMode.exitEditModeWithoutRender()
    }

    // 保存当前图片的气泡状态（如果有气泡）
    const currentImage = imageStore.currentImage
    if (currentImage && bubbleStore.bubbles.length > 0) {
      // 将当前气泡状态保存到图片数据中
      imageStore.updateCurrentImageProperty('bubbleStates', bubbleStore.bubbles)
    }

    // 设置新的当前索引
    imageStore.setCurrentImageIndex(index)
    const newImage = imageStore.currentImage

    if (!newImage) {
      resetSwitchImageFlag()
      return
    }

    // 加载新图片的气泡状态（skipSync=true 避免冗余同步）
    // 使用 clearBubblesLocal 保持 null 和 [] 的语义区分：
    //   - bubbleStates === null: 从未处理过，翻译时应自动检测
    //   - bubbleStates === []: 用户主动清空，翻译时应跳过（避免"框复活"）
    if (newImage.bubbleStates && newImage.bubbleStates.length > 0) {
      bubbleStore.setBubbles(newImage.bubbleStates, true)
    } else {
      // 使用 clearBubblesLocal 仅清除本地状态，不同步到 imageStore
      // 这样不会把 null 错误地写成 []
      bubbleStore.clearBubblesLocal()
    }

    // 注意：图片设置同步由 TranslateView.vue 的 watch 自动处理
    // 当 currentImage 变化时，watch 会调用 syncImageToSidebar

    // 重置切换图片操作的标记
    switchImageFlagTimer = setTimeout(() => {
      switchImageFlagTimer = null
      window._isChangingFromSwitchImage = false
    }, 100)
  }

  /**
   * 切换到上一张图片
   */
  function goToPrevious(): void {
    if (imageStore.canGoPrevious) {
      switchImage(imageStore.currentImageIndex - 1)
    }
  }

  /**
   * 切换到下一张图片
   */
  function goToNext(): void {
    if (imageStore.canGoNext) {
      switchImage(imageStore.currentImageIndex + 1)
    }
  }

  // ============================================================
  // 生命周期
  // ============================================================

  /**
   * 组件挂载时自动初始化
   */
  function setupAutoInit(): void {
    onMounted(async () => {
      await initializeApp()
    })
  }

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
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

    // 初始化方法
    initializeApp,
    initializeSettings,
    initializeFontList,
    initializePromptSettings,
    initializeTextboxPromptSettings,
    initializeBookChapterContext,

    // 图片切换方法
    switchImage,
    goToPrevious,
    goToNext,
    // 编辑模式相关
    editMode,

    // 生命周期
    setupAutoInit
  }
}

// ============================================================
// 全局类型扩展
// ============================================================

// 扩展 Window 接口以支持全局标记
declare global {
  interface Window {
    /** 是否正在切换图片（用于避免重渲染） */
    _isChangingFromSwitchImage?: boolean
  }
}
