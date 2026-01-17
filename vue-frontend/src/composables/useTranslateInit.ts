/**
 * 翻译页面初始化组合式函数
 * 迁移自 main.js 的 initializeApp 函数
 * 
 * 功能：
 * - 页面加载时初始化所有设置（从 localStorage 恢复）
 * - 初始化提示词状态
 * - 初始化字体列表
 * - 初始化主题状态
 * - 初始化插件状态
 * - URL参数解析（书架模式自动加载章节会话）
 * 
 * Requirements: 7.3, 10.2
 */

import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useSettingsStore } from '@/stores/settingsStore'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useEditMode } from '@/composables/useEditMode'
import { showToast } from '@/utils/toast'
import { getFontList, getPrompts, getTextboxPrompts } from '@/api/config'
import { getBookDetail } from '@/api/bookshelf'

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
  /** 字体列表（支持新旧两种格式） */
  fontList: FontInfo[] | string[]
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

  /** 字体列表（支持新旧两种格式） */
  const fontList = ref<FontInfo[] | string[]>([])

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

  // ============================================================
  // 初始化方法
  // ============================================================

  /**
   * 初始化应用
   * 迁移自 main.js 的 initializeApp 函数
   * 
   * @param force - 是否强制重新初始化（用于 SPA 场景下重新进入页面）
   */
  async function initializeApp(force: boolean = false): Promise<void> {
    // 【修复】支持强制重新初始化
    // SPA 场景：用户从书架返回翻译页，需要重新加载书籍/章节上下文
    if (!force && (isInitializing.value || isInitialized.value)) {
      console.log('[TranslateInit] 已经初始化，仅重新加载上下文')
      // 即使跳过完整初始化，也需要重新处理 URL 参数（书架模式）
      await initializeBookChapterContext()
      return
    }

    console.log('[TranslateInit] 开始初始化应用...')
    isInitializing.value = true
    initError.value = null

    try {
      // 1. 初始化设置（优先从后端加载，备选 localStorage）
      await initializeSettings()

      // 2. 初始化字体列表
      await initializeFontList()

      // 3. 初始化提示词设置
      await initializePromptSettings()
      await initializeTextboxPromptSettings()

      // 4. 处理书籍/章节 URL 参数
      await initializeBookChapterContext()

      console.log('[TranslateInit] 应用初始化完成')
      isInitialized.value = true
    } catch (error) {
      console.error('[TranslateInit] 应用初始化失败:', error)
      initError.value = error instanceof Error ? error.message : '初始化失败'
    } finally {
      isInitializing.value = false
    }
  }

  /**
   * 初始化设置
   * 优先从后端加载设置（config/user_settings.json）
   * 如果后端无数据，则从 localStorage 恢复
   */
  async function initializeSettings(): Promise<void> {
    console.log('[TranslateInit] 初始化设置...')

    // 先从 localStorage 加载（作为基础/备份）
    settingsStore.initSettings()

    // 尝试从后端加载设置（会覆盖 localStorage 中的值）
    try {
      const loaded = await settingsStore.loadFromBackend()
      if (loaded) {
        console.log('[TranslateInit] 已从后端加载设置')
      } else {
        console.log('[TranslateInit] 后端无设置，使用 localStorage 数据')
      }
    } catch (error) {
      console.warn('[TranslateInit] 从后端加载设置失败，使用 localStorage 数据:', error)
    }

    console.log('[TranslateInit] 设置初始化完成')
  }

  /**
   * 初始化字体列表
   * 从后端获取系统字体列表
   */
  async function initializeFontList(): Promise<void> {
    console.log('[TranslateInit] 初始化字体列表...')

    try {
      const response = await getFontList()
      // 后端 API 直接返回 { fonts: [...] }，不包含 success 字段
      if (response.fonts && response.fonts.length > 0) {
        fontList.value = response.fonts
        console.log(`[TranslateInit] 已加载 ${response.fonts.length} 个字体`)
      } else if (response.error) {
        console.warn('[TranslateInit] 获取字体列表失败:', response.error)
      } else {
        console.warn('[TranslateInit] 字体列表为空')
      }
    } catch (error) {
      console.error('[TranslateInit] 获取字体列表失败:', error)
      // 字体列表获取失败不阻止初始化
    }
  }

  /**
   * 初始化翻译提示词设置
   * 从后端获取提示词列表和默认内容
   */
  async function initializePromptSettings(): Promise<void> {
    console.log('[TranslateInit] 初始化翻译提示词设置...')

    try {
      const response = await getPrompts()
      // 后端 API 直接返回 { prompt_names: [...], default_prompt_content: "..." }
      if (response.prompt_names !== undefined) {
        promptNames.value = response.prompt_names || []

        // 如果当前没有设置提示词，使用默认提示词
        if (!settingsStore.settings.translatePrompt && response.default_prompt_content) {
          settingsStore.setTranslatePrompt(response.default_prompt_content)
        }

        console.log(`[TranslateInit] 已加载 ${promptNames.value.length} 个翻译提示词`)
      } else if (response.error) {
        console.warn('[TranslateInit] 获取翻译提示词失败:', response.error)
      }
    } catch (error) {
      console.error('[TranslateInit] 获取翻译提示词失败:', error)
      // 提示词获取失败不阻止初始化
    }
  }

  /**
   * 初始化文本框提示词设置
   * 从后端获取文本框提示词列表和默认内容
   */
  async function initializeTextboxPromptSettings(): Promise<void> {
    console.log('[TranslateInit] 初始化文本框提示词设置...')

    try {
      const response = await getTextboxPrompts()
      // 后端 API 直接返回 { prompt_names: [...], default_prompt_content: "..." }
      if (response.prompt_names !== undefined) {
        textboxPromptNames.value = response.prompt_names || []

        // 如果当前没有设置文本框提示词，使用默认提示词
        if (!settingsStore.settings.textboxPrompt && response.default_prompt_content) {
          settingsStore.setTextboxPrompt(response.default_prompt_content)
        }

        console.log(`[TranslateInit] 已加载 ${textboxPromptNames.value.length} 个文本框提示词`)
      } else if (response.error) {
        console.warn('[TranslateInit] 获取文本框提示词失败:', response.error)
      }
    } catch (error) {
      console.error('[TranslateInit] 获取文本框提示词失败:', error)
      // 提示词获取失败不阻止初始化
    }
  }


  /**
   * 初始化书籍/章节上下文
   * 从 URL 参数中读取 book 和 chapter，加载对应的会话数据
   */
  async function initializeBookChapterContext(): Promise<void> {
    const bookId = route.query.book as string | undefined
    const chapterId = route.query.chapter as string | undefined

    if (!bookId || !chapterId) {
      console.log('[TranslateInit] 未指定书籍/章节参数，使用独立模式')
      isBookshelfMode.value = false
      // 【修复】清空书籍/章节上下文，避免残留旧数据
      currentBookId.value = null
      currentChapterId.value = null
      currentBookTitle.value = null
      currentChapterTitle.value = null
      sessionStore.clearContext()
      return
    }

    console.log(`[TranslateInit] 检测到书籍/章节参数: book=${bookId}, chapter=${chapterId}`)
    isBookshelfMode.value = true
    currentBookId.value = bookId
    currentChapterId.value = chapterId

    try {
      // 获取书籍和章节信息
      const bookResponse = await getBookDetail(bookId)

      if (!bookResponse.success || !bookResponse.book) {
        console.warn('[TranslateInit] 书籍不存在:', bookId)
        showToast('书籍不存在', 'warning')
        return
      }

      const book = bookResponse.book
      const chapter = book.chapters?.find(c => c.id === chapterId)

      if (!chapter) {
        console.warn('[TranslateInit] 章节不存在:', chapterId)
        showToast('章节不存在', 'warning')
        return
      }

      // 设置书籍/章节上下文
      currentBookTitle.value = book.title
      currentChapterTitle.value = chapter.title

      // 更新 sessionStore 的上下文
      sessionStore.setBookChapterContext(bookId, chapterId, book.title, chapter.title)

      // 更新页面标题
      if (typeof document !== 'undefined') {
        document.title = `${chapter.title} - ${book.title} - Saber-Translator`
      }

      // 尝试加载章节的会话数据
      if (chapter.session_path) {
        console.log(`[TranslateInit] 尝试加载章节会话: ${chapter.session_path}`)
        try {
          await sessionStore.loadSessionByPath(chapter.session_path)
          showToast(`已加载章节: ${chapter.title}`, 'success')
        } catch {
          console.log('[TranslateInit] 章节会话数据不存在或加载失败，将创建新会话')
        }
      }

    } catch (error) {
      console.error('[TranslateInit] 加载书籍/章节信息失败:', error)
      showToast('加载书籍信息失败', 'error')
    }
  }

  // ============================================================
  // 图片切换逻辑
  // ============================================================

  /**
   * 切换显示的图片
   * 迁移自 main.js 的 switchImage 函数
   * @param index - 要显示的图片索引
   */
  function switchImage(index: number): void {
    if (index < 0 || index >= imageStore.imageCount) {
      console.warn(`[TranslateInit] 无效的图片索引: ${index}`)
      return
    }

    // 设置全局标记，表示当前正在进行切换图片操作
    // 这个标记用于避免在切换图片时触发不必要的重渲染
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
      console.warn('[TranslateInit] 切换到的图片不存在')
      window._isChangingFromSwitchImage = false
      return
    }

    console.log(`[TranslateInit] 切换到图片: ${index}, ${newImage.fileName}`)

    // 加载新图片的气泡状态（skipSync=true 避免冗余同步）
    // 【修复P0】使用 clearBubblesLocal 而非 clearBubbles，保持 null 和 [] 的语义区分：
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
    setTimeout(() => {
      window._isChangingFromSwitchImage = false
      console.log('[TranslateInit] 已重置切换图片操作标记')
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
