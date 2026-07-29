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
  updateChapterSettingsMemory: vi.fn(),
  updateLastVisitedPage: vi.fn(),
}))

const routeState = vi.hoisted(() => ({
  query: {} as Record<string, string | undefined>,
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
  showToast: vi.fn(),
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
          schemaVersion: 3,
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
        detectionState: 'not_started',
        documentRevision: 1,
        height: 1600,
        id: 'page-1',
        logicalSourcePath: '001.png',
        ordinal: 0,
        renderStatus: 'idle',
        renderedRevision: null,
        sourceRevision: 1,
        sourceUrl: '/api/v2/assets/source-1',
        thumbnailSourceUrl: '/api/v2/assets/thumb-1',
        thumbnailTranslatedUrl: null,
        translatedUrl: null,
        width: 1200,
      }],
      nextCursor: null,
      total: 1,
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
    mocks.getPageDocument.mockResolvedValue({
      bubbles: [],
      chapterId: 'quick-chapter',
      defaultFontId: null,
      documentRevision: 1,
      pageId: 'page-1',
      pageStyleDefaults: {},
      pageStyleSchemaVersion: 1,
      renderedRevision: null,
      sourceRevision: 1,
    })
    mocks.updateLastVisitedPage.mockImplementation(
      async (chapterId: string, pageId: string, baseRevision: number) => ({
        chapterId,
        lastVisitedPageId: pageId,
        revision: baseRevision + 1,
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

    expect(mocks.getTranslationBootstrap).toHaveBeenCalledWith({
      bookId: undefined,
      chapterId: undefined,
    })
    expect(imageStore.images).toHaveLength(1)
    expect(imageStore.images[0]?.sourceAssetUrl).toBe('/api/v2/assets/source-1')
    expect(imageStore.images[0]?.thumbnailSourceUrl).toBe('/api/v2/assets/thumb-1')
    expect(mocks.getPageDocument).toHaveBeenCalledTimes(1)
    expect(mocks.getPageDocument).toHaveBeenCalledWith('page-1', expect.any(AbortSignal))
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

  it('serializes last-visited page writes with the backend navigation revision', async () => {
    const payload = bootstrap('quick', 'quick-chapter', 'quick_workspace')
    payload.pages.items.push({
      ...payload.pages.items[0]!,
      id: 'page-2',
      logicalSourcePath: '002.png',
      ordinal: 1,
      sourceUrl: '/api/v2/assets/source-2',
      thumbnailSourceUrl: '/api/v2/assets/thumb-2',
    })
    payload.pages.total = 2
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
      1,
    )
    expect(mocks.updateLastVisitedPage).toHaveBeenNthCalledWith(
      2,
      'quick-chapter',
      'page-1',
      2,
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

  it('restores backend active job progress and page scope after refresh', async () => {
    const payload = bootstrap('book-1', 'chapter-1')
    payload.activeJobs = [{
      id: 'job-1',
      kind: 'translation',
      status: 'running',
      queueRank: 1,
      pageIds: ['page-1'],
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
  })

  it('hydrates and CAS-persists chapter-scoped non-style work state', async () => {
    const payload = bootstrap('book-1', 'chapter-1')
    payload.chapter.settingsMemory = {
      sourceLanguage: 'english',
      targetLanguage: 'zh',
    }
    mocks.getTranslationBootstrap.mockResolvedValue(payload)

    await useTranslateInit().initializeApp()
    const settingsStore = useSettingsStore()
    expect(settingsStore.settings.sourceLanguage).toBe('english')
    expect(settingsStore.chapterWorkStatePayload()).not.toHaveProperty('textStyle')

    settingsStore.settings.sourceLanguage = 'korean'
    await vi.waitFor(() => {
      expect(mocks.updateChapterSettingsMemory).toHaveBeenCalled()
    })
    expect(mocks.updateChapterSettingsMemory).toHaveBeenLastCalledWith(
      'chapter-1',
      expect.objectContaining({ sourceLanguage: 'korean' }),
      1,
    )
    const savedPayload = mocks.updateChapterSettingsMemory.mock.calls.at(-1)?.[1]
    expect(savedPayload).not.toHaveProperty('textStyle')
    expect(JSON.stringify(savedPayload)).not.toContain('apiKey')
  })
})
