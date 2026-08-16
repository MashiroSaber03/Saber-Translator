import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslateInit } from '@/composables/useTranslateInit'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const mocks = vi.hoisted(() => ({
  getPageDocument: vi.fn(),
  getTranslationBootstrap: vi.fn(),
  showToast: vi.fn(),
  updateChapterSettingsMemory: vi.fn(),
  updateLastVisitedPage: vi.fn(),
}))

const routeState = vi.hoisted(() => ({
  query: {} as Record<string, string | string[] | null | undefined>,
}))

vi.mock('vue-router', () => ({
  useRoute: () => routeState,
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
  getTranslationBootstrap: mocks.getTranslationBootstrap,
  updateChapterSettingsMemory: mocks.updateChapterSettingsMemory,
  updateLastVisitedPage: mocks.updateLastVisitedPage,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.showToast,
  useToast: () => ({
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  }),
}))

function bootstrap(
  bookId: string,
  chapterId: string,
  kind: 'library' | 'quick_workspace' = 'library',
) {
  return {
    activeJobs: [],
    activeWebImportDraft: null,
    book: { id: bookId, kind, title: `Book ${bookId}` },
    chapter: {
      id: chapterId,
      pageOrderRevision: 1,
      settingsMemory: {},
      settingsMemoryRevision: 1,
      settingsMemorySchemaVersion: 1,
      title: `Chapter ${chapterId}`,
    },
    constraints: { payload: {}, revision: 1, schemaVersion: 1 },
    navigation: { lastVisitedPageId: null, revision: 1 },
    settings: {
      settings: [
        {
          domain: 'translation',
          payload: createDefaultSettings() as unknown as Record<string, unknown>,
          revision: 1,
          schemaVersion: 6,
        },
        {
          domain: 'text_style_defaults',
          payload: createDefaultSettings().textStyle as unknown as Record<string, unknown>,
          revision: 1,
          schemaVersion: 1,
        },
        {
          domain: 'workflow_preferences',
          payload: {
            rememberWorkflowModeEnabled: false,
            lastWorkflowMode: 'translate-current',
          },
          revision: 1,
          schemaVersion: 1,
        },
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    },
    fonts: [],
    prompts: [],
    pages: {
      items: [{
        chapterId,
        cleanUrl: null,
        detectionState: 'unprocessed',
        documentRevision: 1,
        height: 1600,
        id: 'page-1',
        logicalSourcePath: '001.png',
        ordinal: 0,
        renderStatus: 'not_rendered',
        renderedRevision: null,
        sourceRevision: 1,
        sourceUrl: '/api/v2/assets/source-1',
        thumbnailSourceUrl: '/api/v2/assets/thumb-1',
        translatedUrl: null,
        width: 1200,
      }],
      nextCursor: null,
      pageOrderRevision: 1,
    },
  }
}

describe('useTranslateInit', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    routeState.query = {}
    mocks.getTranslationBootstrap.mockResolvedValue(
      bootstrap('quick', 'quick-chapter', 'quick_workspace'),
    )
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    mocks.getPageDocument.mockImplementation(async (pageId: string) => ({
      bubbles: [],
      chapterId: typeof routeState.query.chapter === 'string'
        ? routeState.query.chapter
        : 'quick-chapter',
      defaultFontId: fontFamily,
      documentRevision: 1,
      pageId,
      pageStyleDefaults,
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
    }))
    mocks.updateLastVisitedPage.mockImplementation(
      async (chapterId: string, pageId: string) => ({
        chapterId,
        lastVisitedPageId: pageId,
        revision: 1,
      }),
    )
    mocks.updateChapterSettingsMemory.mockImplementation(
      async (chapterId: string, payload: Record<string, unknown>, baseRevision: number) => ({
        chapterId,
        payload,
        revision: baseRevision + 1,
      }),
    )
  })

  it('hydrates page metadata with backend URLs and loads only the current page document', async () => {
    await useTranslateInit().initializeApp()
    const imageStore = useImageStore()

    expect(mocks.getTranslationBootstrap).toHaveBeenCalledWith({})
    expect(imageStore.images).toHaveLength(1)
    expect(imageStore.images[0]?.sourceAssetUrl).toBe('/api/v2/assets/source-1')
    expect(imageStore.images[0]?.thumbnailSourceUrl).toBe('/api/v2/assets/thumb-1')
    expect(mocks.getPageDocument).toHaveBeenCalledTimes(1)
    expect(mocks.getPageDocument).toHaveBeenCalledWith('page-1', expect.any(AbortSignal))
  })

  it('does not finish initialization before the authoritative page style is loaded', async () => {
    const defaults = createDefaultSettings().textStyle
    const { fontFamily, ...pageStyleDefaults } = defaults
    let resolveDocument!: (value: Record<string, unknown>) => void
    mocks.getPageDocument.mockImplementationOnce(() => new Promise(resolve => {
      resolveDocument = resolve
    }))

    const state = useTranslateInit()
    const initialization = state.initializeApp()
    await vi.waitFor(() => {
      expect(mocks.getPageDocument).toHaveBeenCalledTimes(1)
    })

    const imageStore = useImageStore()
    expect(imageStore.currentImage?.bubbleStates).toBeNull()

    resolveDocument({
      bubbles: [],
      chapterId: 'quick-chapter',
      defaultFontId: fontFamily,
      documentRevision: 4,
      pageId: 'page-1',
      pageStyleDefaults: {
        ...pageStyleDefaults,
        layoutDirection: 'horizontal',
        textColor: '#123456',
        useAutoTextColor: true,
      },
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
    })
    await initialization

    expect(imageStore.currentImage).toMatchObject({
      bubbleStates: [],
      documentRevision: 4,
      layoutDirection: 'horizontal',
      textColor: '#123456',
      useAutoTextColor: true,
    })
    expect(useSettingsStore().settings.textStyle).toMatchObject({
      layoutDirection: 'horizontal',
      textColor: '#123456',
      useAutoTextColor: true,
    })
  })

  it('requests the selected library chapter through the v2 bootstrap', async () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    mocks.getTranslationBootstrap.mockResolvedValue(
      bootstrap('book-1', 'chapter-1'),
    )

    const state = useTranslateInit()
    await state.initializeBookChapterContext()

    expect(mocks.getTranslationBootstrap).toHaveBeenCalledWith({
      bookId: 'book-1',
      chapterId: 'chapter-1',
    })
    expect(state.isBookshelfMode.value).toBe(true)
    expect(state.currentBookId.value).toBe('book-1')
    expect(state.currentChapterId.value).toBe('chapter-1')
  })

  it.each([
    { book: 'book-1' },
    { chapter: 'chapter-1' },
    { book: ['book-1', 'book-2'], chapter: 'chapter-1' },
    { book: '', chapter: 'chapter-1' },
  ])('rejects an incomplete or non-scalar translation route: %o', async query => {
    routeState.query = query
    const state = useTranslateInit()

    await state.initializeBookChapterContext()

    expect(mocks.getTranslationBootstrap).not.toHaveBeenCalled()
    expect(state.currentBookId.value).toBeNull()
    expect(useImageStore().images).toEqual([])
    expect(mocks.showToast).toHaveBeenCalledWith(
      '翻译页面地址无效：book 与 chapter 必须同时且各自只提供一个值',
      'error',
    )
  })

  it('rejects a bootstrap that belongs to a different requested chapter', async () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    mocks.getTranslationBootstrap.mockResolvedValue(
      bootstrap('book-1', 'other-chapter'),
    )
    const state = useTranslateInit()

    await state.initializeBookChapterContext()

    expect(state.currentBookId.value).toBeNull()
    expect(useImageStore().images).toEqual([])
    expect(mocks.showToast).toHaveBeenCalledWith(
      '加载后端章节数据失败：后端返回了其他翻译工作区的数据',
      'error',
    )
  })

  it('clears a previously loaded chapter when the next authoritative bootstrap fails', async () => {
    const state = useTranslateInit()
    await expect(state.initializeBookChapterContext()).resolves.toBe(true)
    expect(state.currentBookId.value).toBe('quick')
    expect(useImageStore().images).toHaveLength(1)

    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    mocks.getTranslationBootstrap.mockRejectedValueOnce(new Error('backend unavailable'))

    await expect(state.initializeBookChapterContext()).resolves.toBe(false)

    expect(state.currentBookId.value).toBeNull()
    expect(state.currentChapterId.value).toBeNull()
    expect(useImageStore().images).toEqual([])
    expect(mocks.showToast).toHaveBeenCalledWith(
      '加载后端章节数据失败：backend unavailable',
      'error',
    )
  })

  it('serializes last-write-wins page navigation updates', async () => {
    const payload = bootstrap('quick', 'quick-chapter', 'quick_workspace')
    payload.pages.items.push({
      ...payload.pages.items[0]!,
      id: 'page-2',
      logicalSourcePath: '002.png',
      ordinal: 1,
      sourceUrl: '/api/v2/assets/source-2',
      thumbnailSourceUrl: '/api/v2/assets/thumb-2',
    })
    payload.pages.pageOrderRevision = 2
    mocks.getTranslationBootstrap.mockResolvedValue(payload)

    const state = useTranslateInit()
    await state.initializeApp()
    await state.switchImage(1)
    await state.switchImage(0)

    await vi.waitFor(() => {
      expect(mocks.updateLastVisitedPage).toHaveBeenCalledTimes(2)
    })
    expect(mocks.updateLastVisitedPage).toHaveBeenNthCalledWith(
      1,
      'quick-chapter',
      'page-2',
    )
    expect(mocks.updateLastVisitedPage).toHaveBeenNthCalledWith(
      2,
      'quick-chapter',
      'page-1',
    )
  })

  it('keeps the current page selected when the target document cannot be loaded', async () => {
    const payload = bootstrap('quick', 'quick-chapter', 'quick_workspace')
    payload.pages.items.push({
      ...payload.pages.items[0]!,
      id: 'page-2',
      logicalSourcePath: '002.png',
      ordinal: 1,
      sourceUrl: '/api/v2/assets/source-2',
      thumbnailSourceUrl: '/api/v2/assets/thumb-2',
    })
    payload.pages.pageOrderRevision = 2
    mocks.getTranslationBootstrap.mockResolvedValue(payload)
    const state = useTranslateInit()
    await expect(state.initializeApp()).resolves.toBe(true)
    mocks.getPageDocument.mockRejectedValueOnce(new Error('document unavailable'))

    await expect(state.switchImage(1)).resolves.toBe(false)

    expect(useImageStore().currentImage?.id).toBe('page-1')
    expect(state.isSwitchingImage.value).toBe(false)
    expect(mocks.updateLastVisitedPage).not.toHaveBeenCalled()
    expect(mocks.showToast).toHaveBeenCalledWith(
      '加载当前页编辑数据失败：document unavailable',
      'error',
    )
  })

  it('clears a bootstrap that cannot load its initial page document', async () => {
    mocks.getPageDocument.mockRejectedValueOnce(new Error('document unavailable'))
    const state = useTranslateInit()

    await expect(state.initializeApp()).resolves.toBe(false)

    expect(state.currentBookId.value).toBeNull()
    expect(state.currentChapterId.value).toBeNull()
    expect(useImageStore().images).toEqual([])
    expect(mocks.showToast).toHaveBeenCalledWith(
      '加载当前页编辑数据失败：document unavailable',
      'error',
    )
  })

  it('ignores a stale bootstrap response after navigation changes', async () => {
    let resolveFirst!: (value: ReturnType<typeof bootstrap>) => void
    mocks.getTranslationBootstrap
      .mockImplementationOnce(() => new Promise(resolve => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce(bootstrap('quick', 'quick-chapter', 'quick_workspace'))

    routeState.query = { book: 'old-book', chapter: 'old-chapter' }
    const state = useTranslateInit()
    const stale = state.initializeBookChapterContext()
    routeState.query = {}
    await state.initializeBookChapterContext()
    resolveFirst(bootstrap('old-book', 'old-chapter'))
    await stale

    expect(state.currentBookId.value).toBe('quick')
    expect(state.currentChapterId.value).toBe('quick-chapter')
    expect(state.isBookshelfMode.value).toBe(false)
  })

  it('invalidates an active page read as soon as a newer chapter refresh starts', async () => {
    const initial = bootstrap('quick', 'quick-chapter', 'quick_workspace')
    initial.pages.items.push({
      ...initial.pages.items[0]!,
      id: 'page-2',
      logicalSourcePath: '002.png',
      ordinal: 1,
      sourceUrl: '/api/v2/assets/source-2',
      thumbnailSourceUrl: '/api/v2/assets/thumb-2',
    })
    initial.pages.pageOrderRevision = 2
    mocks.getTranslationBootstrap.mockResolvedValueOnce(initial)
    const state = useTranslateInit()
    await expect(state.initializeApp()).resolves.toBe(true)

    let resolveOldPage!: (value: Record<string, unknown>) => void
    mocks.getPageDocument.mockImplementationOnce(() => new Promise(resolve => {
      resolveOldPage = resolve
    }))
    const staleSwitch = state.switchImage(1)
    await vi.waitFor(() => {
      expect(mocks.getPageDocument).toHaveBeenCalledTimes(2)
    })

    let resolveRefresh!: (value: ReturnType<typeof bootstrap>) => void
    mocks.getTranslationBootstrap.mockImplementationOnce(() => new Promise(resolve => {
      resolveRefresh = resolve
    }))
    const refresh = state.initializeBookChapterContext()
    await vi.waitFor(() => {
      expect(mocks.getTranslationBootstrap).toHaveBeenCalledTimes(2)
    })

    resolveOldPage({
      bubbles: [],
      chapterId: 'quick-chapter',
      defaultFontId: null,
      documentRevision: 1,
      pageId: 'page-2',
      pageStyleDefaults: createDefaultSettings().textStyle,
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
    })
    await expect(staleSwitch).resolves.toBe(false)
    expect(useImageStore().currentImage?.id).toBe('page-1')

    const refreshed = bootstrap('quick', 'quick-chapter', 'quick_workspace')
    Object.assign(refreshed.pages.items[0]!, {
      id: 'page-new',
      logicalSourcePath: 'new.png',
      sourceUrl: '/api/v2/assets/source-new',
      thumbnailSourceUrl: '/api/v2/assets/thumb-new',
    })
    resolveRefresh(refreshed)

    await expect(refresh).resolves.toBe(true)
    expect(useImageStore().currentImage?.id).toBe('page-new')
  })

  it('restores backend active job progress without overwriting completed pages', async () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    const payload = bootstrap('book-1', 'chapter-1')
    Object.assign(payload.pages.items[0], {
      documentRevision: 2,
      renderedRevision: 2,
      renderStatus: 'ready',
      translatedUrl: '/api/v2/assets/translated-1',
    })
    payload.activeJobs = [{
      id: 'job-1',
      kind: 'translation',
      status: 'running',
      queueRank: 1,
      pages: [{ pageId: 'page-1', status: 'completed' }],
      progress: {
        executionMode: 'parallel',
        jobStatus: 'running',
        totalItems: 1,
        completedItems: 0,
        failedItems: 0,
        skippedItems: 0,
        cancelledItems: 0,
        pools: [{
          kind: 'ocr',
          total: 1,
          completed: 0,
          failed: 0,
          skipped: 0,
          cancelled: 0,
          waiting: 0,
          processing: 1,
          lockWaiting: false,
          current: [],
        }],
      },
    }]
    mocks.getTranslationBootstrap.mockResolvedValue(payload)

    await useTranslateInit().initializeApp()
    const translation = useTranslation()

    expect(translation.progress.value.isInProgress).toBe(true)
    expect(translation.progress.value.executionMode).toBe('parallel')
    expect(translation.progress.value.pools[0]?.kind).toBe('ocr')
    expect(useImageStore().images[0]?.translationStatus).toBe('completed')
  })

  it('hydrates and CAS-persists chapter-scoped non-style work state', async () => {
    routeState.query = { book: 'book-1', chapter: 'chapter-1' }
    const payload = bootstrap('book-1', 'chapter-1')
    payload.chapter.settingsMemory = {
      targetLanguage: 'en',
    }
    mocks.getTranslationBootstrap.mockResolvedValue(payload)

    await useTranslateInit().initializeApp()
    const settingsStore = useSettingsStore()
    expect(settingsStore.settings.targetLanguage).toBe('en')
    expect(settingsStore.chapterWorkStatePayload()).not.toHaveProperty('textStyle')

    settingsStore.settings.targetLanguage = 'ja'
    await vi.waitFor(() => {
      expect(mocks.updateChapterSettingsMemory).toHaveBeenCalled()
    })
    expect(mocks.updateChapterSettingsMemory).toHaveBeenLastCalledWith(
      'chapter-1',
      expect.objectContaining({ targetLanguage: 'ja' }),
      1,
    )
    const savedPayload = mocks.updateChapterSettingsMemory.mock.calls.at(-1)?.[1]
    expect(savedPayload).not.toHaveProperty('textStyle')
    expect(JSON.stringify(savedPayload)).not.toContain('apiKey')
  })

  it('flushes the latest chapter work state without waiting for the debounce window', async () => {
    const state = useTranslateInit()
    await state.initializeApp()
    const settingsStore = useSettingsStore()
    mocks.updateChapterSettingsMemory.mockClear()
    settingsStore.settings.targetLanguage = 'ja'

    expect(await state.flushChapterWorkState()).toBe(true)
    expect(mocks.updateChapterSettingsMemory).toHaveBeenLastCalledWith(
      'quick-chapter',
      expect.objectContaining({ targetLanguage: 'ja' }),
      expect.any(Number),
    )
  })

  it('persists a value restored while the previous value is still in flight', async () => {
    const state = useTranslateInit()
    await state.initializeApp()
    const settingsStore = useSettingsStore()
    const originalTargetLanguage = settingsStore.settings.targetLanguage
    let resolveFirst!: () => void
    mocks.updateChapterSettingsMemory.mockReset()
    mocks.updateChapterSettingsMemory
      .mockImplementationOnce((
        chapterId: string,
        payload: Record<string, unknown>,
        baseRevision: number,
      ) => new Promise(resolve => {
        resolveFirst = () => resolve({
          chapterId,
          payload,
          revision: baseRevision + 1,
        })
      }))
      .mockImplementation(async (
        chapterId: string,
        payload: Record<string, unknown>,
        baseRevision: number,
      ) => ({
        chapterId,
        payload,
        revision: baseRevision + 1,
      }))

    settingsStore.settings.targetLanguage = 'ja'
    const firstFlush = state.flushChapterWorkState()
    await vi.waitFor(() => {
      expect(mocks.updateChapterSettingsMemory).toHaveBeenCalledTimes(1)
    })

    settingsStore.settings.targetLanguage = originalTargetLanguage
    const finalFlush = state.flushChapterWorkState()
    resolveFirst()

    expect(await firstFlush).toBe(true)
    expect(await finalFlush).toBe(true)
    expect(mocks.updateChapterSettingsMemory).toHaveBeenCalledTimes(2)
    expect(mocks.updateChapterSettingsMemory).toHaveBeenLastCalledWith(
      'quick-chapter',
      expect.objectContaining({ targetLanguage: originalTargetLanguage }),
      2,
    )
  })
})
