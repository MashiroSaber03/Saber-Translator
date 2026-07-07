import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent } from 'vue'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useTranslateInit } from '@/composables/useTranslateInit'
import { useImageStore } from '@/stores/imageStore'
import { useSessionStore } from '@/stores/sessionStore'
import { useSettingsStore } from '@/stores/settings'
import { getFontList, getPrompts, getTextboxPrompts } from '@/api/config'
import { cleanupGpu } from '@/api/system'
import { reloadTextStyleDefaultsFromBackend } from '@/defaults/textStyleDefaults'
import { getBookDetail } from '@/api/bookshelf'

const routeState = vi.hoisted(() => ({
  query: {} as Record<string, string | undefined>,
}))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
}))

vi.mock('@/api/config', () => ({
  getFontList: vi.fn(),
  getPrompts: vi.fn(),
  getTextboxPrompts: vi.fn(),
}))

vi.mock('@/api/system', () => ({
  cleanupGpu: vi.fn(),
}))

vi.mock('@/api/bookshelf', () => ({
  getBookDetail: vi.fn(),
}))

vi.mock('@/defaults/textStyleDefaults', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/defaults/textStyleDefaults')>()
  return {
    ...actual,
    reloadTextStyleDefaultsFromBackend: vi.fn(),
  }
})

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

describe('useTranslateInit', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    routeState.query = {}

    vi.mocked(reloadTextStyleDefaultsFromBackend).mockResolvedValue(undefined)
    vi.mocked(getFontList).mockResolvedValue({
      fonts: [
        {
          file_name: 'default.ttf',
          display_name: 'Default',
          path: '/fonts/default.ttf',
          is_default: true,
        },
      ],
    })
    vi.mocked(getPrompts).mockResolvedValue({
      prompt_names: ['Default'],
      default_prompt_content: '',
    })
    vi.mocked(getTextboxPrompts).mockResolvedValue({
      prompt_names: ['Default textbox'],
      default_prompt_content: '',
    })
    vi.mocked(cleanupGpu).mockResolvedValue({
      success: true,
      unloaded_models: ['ocr'],
      memory_allocated_mb: 0,
      memory_reserved_mb: 0,
    })
    vi.mocked(getBookDetail).mockResolvedValue({
      success: false,
      error: 'not configured',
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('does not write routine console logs during successful initialization and image switching', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const translateInit = useTranslateInit()
    const settingsStore = useSettingsStore()
    const sessionStore = useSessionStore()
    const imageStore = useImageStore()

    vi.spyOn(settingsStore, 'initSettings').mockImplementation(() => {})
    vi.spyOn(settingsStore, 'loadFromBackend').mockResolvedValue(true)
    vi.spyOn(sessionStore, 'clearContext').mockImplementation(() => {})

    await translateInit.initializeApp()
    imageStore.addImage('first.png', 'data:image/png;base64,first')
    imageStore.addImage('second.png', 'data:image/png;base64,second')
    translateInit.switchImage(1)
    await vi.advanceTimersByTimeAsync(100)

    expect(translateInit.isSwitchingImage.value).toBe(false)
    expect(consoleLog).not.toHaveBeenCalled()
  })

  it('clears the image-switching flag when the owner unmounts', () => {
    const imageStore = useImageStore()
    imageStore.addImage('first.png', 'data:image/png;base64,first')
    imageStore.addImage('second.png', 'data:image/png;base64,second')

    let translateInit!: ReturnType<typeof useTranslateInit>
    const Harness = defineComponent({
      setup() {
        translateInit = useTranslateInit()
        return () => null
      },
    })

    const wrapper = mount(Harness)
    translateInit.switchImage(1)

    expect(translateInit.isSwitchingImage.value).toBe(true)

    wrapper.unmount()
    expect(translateInit.isSwitchingImage.value).toBe(false)
  })

  it('ignores stale bookshelf context responses after the route leaves bookshelf mode', async () => {
    let resolveBookDetail!: (value: Awaited<ReturnType<typeof getBookDetail>>) => void
    vi.mocked(getBookDetail).mockImplementationOnce(() => new Promise((resolve) => {
      resolveBookDetail = resolve
    }))

    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    const translateInit = useTranslateInit()
    const sessionStore = useSessionStore()
    const setContext = vi.spyOn(sessionStore, 'setBookChapterContext')

    const pendingContext = translateInit.initializeBookChapterContext()
    routeState.query = {}
    await translateInit.initializeBookChapterContext()

    resolveBookDetail({
      success: true,
      book: {
        id: 'book-1',
        title: 'Stale Book',
        chapters: [{
          id: 'chapter-1',
          title: 'Stale Chapter',
          order: 1,
          imageCount: 0,
          hasSession: false,
        }],
        createdAt: '2026-06-25T00:00:00.000Z',
        updatedAt: '2026-06-25T00:00:00.000Z',
      },
    })
    await pendingContext

    expect(translateInit.isBookshelfMode.value).toBe(false)
    expect(translateInit.currentBookId.value).toBeNull()
    expect(translateInit.currentChapterId.value).toBeNull()
    expect(translateInit.currentBookTitle.value).toBeNull()
    expect(translateInit.currentChapterTitle.value).toBeNull()
    expect(setContext).not.toHaveBeenCalled()
  })

  it('keeps initialization source comments focused on current behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/composables/useTranslateInit.ts'), 'utf8')

    for (const staleNarration of [
      '翻译页面初始化组合式函数',
      '// ============================================================',
      '类型定义',
      '状态定义',
      '初始化方法',
      '图片切换逻辑',
      '生命周期',
      '返回',
      '@param',
      '1. 初始化设置',
      '2. 初始化字体列表',
      '3. 初始化提示词设置',
      '4. 清理 GPU 资源',
      '5. 处理书籍/章节 URL 参数',
    ]) {
      expect(source).not.toContain(staleNarration)
    }

    expect(source).toContain('null 和 [] 的语义区分')
    expect(source).toContain('Backend settings are optional')
  })
})
