import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslation } from '@/composables/useTranslationPipeline'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({
  createChapterRemoveTextJob: vi.fn(),
  createChapterTranslationJob: vi.fn(),
  getPageDocument: vi.fn(),
  getPageSummary: vi.fn(),
  jobsList: vi.fn(),
  jobsRetryFailed: vi.fn(),
  listChapterPages: vi.fn(),
  toast: {
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock('@/api/v2/translation', () => ({
  createChapterRemoveTextJob: mocks.createChapterRemoveTextJob,
  createChapterTranslationJob: mocks.createChapterTranslationJob,
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
  getPageSummary: mocks.getPageSummary,
  listChapterPages: mocks.listChapterPages,
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    cancel: vi.fn(),
    list: mocks.jobsList,
    retryFailed: mocks.jobsRetryFailed,
  },
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => mocks.toast,
}))

describe('useTranslationPipeline', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.createChapterRemoveTextJob.mockResolvedValue({
      batchId: 'remove-batch-1',
      jobIds: ['remove-job-1'],
      status: 'queued',
    })
    mocks.createChapterTranslationJob.mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-1'],
      status: 'queued',
    })
    mocks.jobsList.mockResolvedValue({ items: [], queueRevision: 1 })
    mocks.jobsRetryFailed.mockResolvedValue({
      batchId: 'retry-batch',
      jobIds: ['retry-job'],
      status: 'queued',
      sourceJobId: 'source-job',
      retryMode: 'current',
      failedOnly: true,
    })
    mocks.listChapterPages.mockResolvedValue({ items: [] })
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      documentRevision: 2,
      defaultFontId: fontFamily,
      pageStyleDefaults,
      bubbles: [],
    })
  })

  it('submits selected pages to one durable backend job without changing navigation', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    addTestImage(imageStore, '002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-1',
      id: 'page-2',
    })
    imageStore.setCurrentImageIndex(0)

    const result = await useTranslation().translateSelectedImages({ pages: [2] })

    expect(result).toBe(true)
    expect(mocks.createChapterTranslationJob).toHaveBeenCalledWith(
      'chapter-1',
      ['page-2'],
      expect.objectContaining({ mode: 'standard' }),
    )
    expect(imageStore.currentImageIndex).toBe(0)
    expect(imageStore.images[1]?.translationStatus).toBe('processing')
    expect(mocks.toast.success).toHaveBeenCalledWith(
      '任务已加入后端任务中心，可安全关闭页面',
    )
  })

  it('does not create a job when chapter settings preflight fails', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    const beforeCreateJob = vi.fn().mockResolvedValue(false)

    const result = await useTranslation({ beforeCreateJob }).translateCurrentImage()

    expect(result).toBe(false)
    expect(beforeCreateJob).toHaveBeenCalledOnce()
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
  })

  it('uses the dedicated backend remove-text job endpoint', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    settingsStore.settings.parallel.enabled = true
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })

    const result = await useTranslation().removeTextOnly()

    expect(result).toBe(true)
    expect(mocks.createChapterRemoveTextJob).toHaveBeenCalledWith(
      'chapter-1',
      ['page-1'],
      'parallel',
    )
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
  })

  it('rejects pages that are not owned by one backend chapter', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    addTestImage(imageStore, '002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-2',
      id: 'page-2',
    })

    const result = await useTranslation().translateAllImages()

    expect(result).toBe(false)
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
  })

  it('refreshes a completed page immediately without reloading the chapter', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 1,
      id: 'page-1',
    })
    addTestImage(imageStore, '002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-1',
      documentRevision: 1,
      id: 'page-2',
    })
    imageStore.setCurrentImageIndex(0)
    mocks.getPageSummary.mockResolvedValue({
      id: 'page-1',
      chapterId: 'chapter-1',
      ordinal: 1,
      logicalSourcePath: '001.png',
      width: 100,
      height: 200,
      sourceRevision: 1,
      documentRevision: 2,
      renderedRevision: 2,
      renderStatus: 'ready',
      detectionState: 'completed',
      sourceUrl: '/api/v2/assets/source-1',
      thumbnailSourceUrl: '/api/v2/assets/thumb-1',
      cleanUrl: '/api/v2/assets/clean-1',
      translatedUrl: '/api/v2/assets/translated-1',
    })

    await useTranslation().translateAllImages()
    taskCenterStore.latestEvent = {
      eventId: 998,
      jobId: 'job-1',
      type: 'page_completed',
      payload: {
        pageId: 'page-1',
        progress: { totalItems: 2, completedItems: 1 },
      },
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    await vi.waitFor(() => {
      expect(imageStore.images[0]?.translatedAssetUrl).toBe(
        '/api/v2/assets/translated-1',
      )
    })
    expect(imageStore.images[0]).toMatchObject({
      translationStatus: 'completed',
    })
    expect(imageStore.images[1]?.translationStatus).toBe('processing')
    expect(mocks.getPageSummary).toHaveBeenCalledOnce()
    expect(mocks.getPageSummary).toHaveBeenCalledWith('page-1')
    expect(mocks.listChapterPages).not.toHaveBeenCalled()
    expect(mocks.getPageDocument).toHaveBeenCalledWith('page-1')
  })

  it('rehydrates the current page document when a durable job finishes', async () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 1,
      id: 'page-1',
    })
    mocks.listChapterPages.mockResolvedValue({
      items: [{
        id: 'page-1',
        chapterId: 'chapter-1',
        ordinal: 1,
        logicalSourcePath: '001.png',
        width: 100,
        height: 200,
        sourceRevision: 1,
        documentRevision: 2,
        renderedRevision: 2,
        renderStatus: 'ready',
        detectionState: 'completed',
        sourceUrl: '/api/v2/assets/source-1',
        thumbnailSourceUrl: '/api/v2/assets/thumb-1',
        cleanUrl: '/api/v2/assets/clean-1',
        translatedUrl: '/api/v2/assets/translated-1',
      }],
    })
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      documentRevision: 2,
      defaultFontId: fontFamily,
      pageStyleDefaults: {
        ...pageStyleDefaults,
        layoutDirection: 'horizontal',
        textColor: '#123456',
        useAutoTextColor: false,
      },
      bubbles: [{
        bubbleId: 'bubble-1',
        ordinal: 1,
        fontId: null,
        payload: {
          originalText: 'こんにちは',
          translatedText: '你好',
          coords: [1, 2, 30, 40],
        },
      }],
    })

    await useTranslation().translateCurrentImage()
    taskCenterStore.latestEvent = {
      eventId: 999,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    await vi.waitFor(() => {
      expect(imageStore.currentImage?.bubbleStates).toHaveLength(1)
      expect(bubbleStore.bubbles).toHaveLength(1)
    })
    expect(imageStore.currentImage?.translatedAssetUrl).toBe(
      '/api/v2/assets/translated-1',
    )
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('你好')
    expect(imageStore.currentImage).toMatchObject({
      layoutDirection: 'horizontal',
      textColor: '#123456',
      useAutoTextColor: false,
    })
    expect(settingsStore.settings.textStyle).toMatchObject({
      layoutDirection: 'horizontal',
      textColor: '#123456',
      useAutoTextColor: false,
    })
  })

  it('does not reload the open chapter for an unrelated terminal job', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    useTranslation()
    taskCenterStore.history = [{
      jobId: 'job-export',
      batchId: 'batch-export',
      batchDisplayName: '导出',
      kind: 'export',
      retryOfJobId: null,
      retryMode: null,
      status: 'completed',
      queueRank: null,
      bookId: 'book-1',
      chapterId: 'chapter-1',
      pageId: null,
      blockedReason: null,
      progress: {},
      createdAt: new Date().toISOString(),
      startedAt: null,
      finishedAt: new Date().toISOString(),
    }]
    taskCenterStore.latestEvent = {
      eventId: 1000,
      jobId: 'job-export',
      type: 'job_finished',
      payload: {},
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    expect(mocks.listChapterPages).not.toHaveBeenCalled()
    expect(mocks.getPageDocument).not.toHaveBeenCalled()
  })

  it('reconciles newly imported pages when a backend container job finishes', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    imageStore.setCurrentImageIndex(0)
    mocks.listChapterPages.mockResolvedValue({
      items: [
        {
          id: 'page-1',
          chapterId: 'chapter-1',
          ordinal: 1,
          logicalSourcePath: '001.png',
          width: 100,
          height: 200,
          sourceRevision: 1,
          documentRevision: 1,
          renderedRevision: null,
          renderStatus: 'pending',
          detectionState: 'pending',
          sourceUrl: '/api/v2/assets/source-1',
          thumbnailSourceUrl: '/api/v2/assets/thumb-1',
          cleanUrl: null,
          translatedUrl: null,
        },
        {
          id: 'page-2',
          chapterId: 'chapter-1',
          ordinal: 2,
          logicalSourcePath: '002.png',
          width: 120,
          height: 220,
          sourceRevision: 1,
          documentRevision: 0,
          renderedRevision: null,
          renderStatus: 'pending',
          detectionState: 'pending',
          sourceUrl: '/api/v2/assets/source-2',
          thumbnailSourceUrl: '/api/v2/assets/thumb-2',
          cleanUrl: null,
          translatedUrl: null,
        },
      ],
    })
    taskCenterStore.history = [{
      jobId: 'container-job',
      batchId: 'container-batch',
      batchDisplayName: 'Container import',
      kind: 'container_import',
      retryOfJobId: null,
      retryMode: null,
      status: 'completed',
      queueRank: null,
      bookId: 'book-1',
      chapterId: 'chapter-1',
      pageId: null,
      blockedReason: null,
      blockedByJobId: null,
      progress: { totalItems: 2, completedItems: 2 },
      target: { pageCount: 2 },
      createdAt: null,
      startedAt: null,
      finishedAt: null,
    }]

    useTranslation()
    taskCenterStore.latestEvent = {
      eventId: 1001,
      jobId: 'container-job',
      type: 'job_finished',
      payload: {},
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    await vi.waitFor(() => expect(imageStore.images).toHaveLength(2))
    expect(imageStore.images[1]).toMatchObject({
      id: 'page-2',
      fileName: '002.png',
      sourceAssetUrl: '/api/v2/assets/source-2',
    })
    expect(imageStore.currentImage?.id).toBe('page-1')
  })

  it('reconciles a tracked job from durable queue and history snapshots', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    const translation = useTranslation()
    await translation.translateCurrentImage()

    taskCenterStore.queue = [{
      jobId: 'job-1',
      batchId: 'batch-1',
      batchDisplayName: 'Book / Chapter',
      kind: 'translation',
      retryOfJobId: null,
      retryMode: null,
      status: 'running',
      queueRank: null,
      bookId: 'book-1',
      chapterId: 'chapter-1',
      pageId: null,
      blockedReason: null,
      blockedByJobId: null,
      progress: {
        totalItems: 1,
        completedItems: 0,
        failedItems: 0,
        jobStatus: 'running',
        executionMode: 'sequential',
        pools: [],
      },
      target: { book: 'Book', chapter: 'Chapter', pageCount: 1 },
      createdAt: null,
      startedAt: null,
      finishedAt: null,
    }]
    await nextTick()

    expect(translation.progress.value.status).toBe('running')
    expect(translation.progress.value.label).toBe('后端正在处理')

    const runningJob = taskCenterStore.queue[0]
    taskCenterStore.queue = []
    taskCenterStore.history = [{
      ...runningJob,
      jobId: 'job-1',
      batchId: 'batch-1',
      batchDisplayName: 'Book / Chapter',
      kind: 'translation',
      retryOfJobId: null,
      retryMode: null,
      status: 'completed',
      queueRank: null,
      progress: {
        totalItems: 1,
        completedItems: 1,
        failedItems: 0,
        jobStatus: 'completed',
        executionMode: 'sequential',
        pools: [],
      },
      target: { book: 'Book', chapter: 'Chapter', pageCount: 1 },
      createdAt: null,
    }]
    await nextTick()

    expect(translation.progress.value.status).toBe('completed')
    expect(translation.progress.value.isInProgress).toBe(false)
  })

  it('retries durable failed items from the latest matching backend job', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    imageStore.setTranslationStatus(0, 'failed')
    mocks.jobsList.mockImplementation(async (scope: string) => ({
      items: scope === 'history'
        ? [{
            jobId: 'source-job',
            batchId: 'source-batch',
            batchDisplayName: 'source',
            kind: 'translation',
            retryOfJobId: null,
            retryMode: null,
            status: 'completed_with_errors',
            queueRank: null,
            bookId: null,
            chapterId: 'chapter-1',
            pageId: null,
            blockedReason: null,
            blockedByJobId: null,
            progress: { totalItems: 1, failedItems: 1 },
            target: {},
            createdAt: null,
            startedAt: null,
            finishedAt: null,
          }]
        : [],
      queueRevision: 1,
    }))

    const retried = await useTranslation().retryFailedImages()

    expect(retried).toBe(true)
    expect(mocks.jobsRetryFailed).toHaveBeenCalledWith('source-job', 'current')
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalledWith(
      '失败项已按当前设置加入后端任务中心',
    )
  })

  it('contains no browser pipeline, session persistence, or image payload processing', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useTranslationPipeline.ts'),
      'utf8',
    )

    expect(source).toContain('createChapterTranslationJob')
    expect(source).toContain('createChapterRemoveTextJob')
    expect(source).not.toContain('usePipeline')
    expect(source).not.toContain('executeRender')
    expect(source).not.toContain('sessionStore')
    expect(source).not.toContain('base64')
  })
})
