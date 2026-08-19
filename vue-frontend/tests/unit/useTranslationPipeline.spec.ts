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
import type { components } from '@/api/generated/v2'

type JobProgress = components['schemas']['JobProgress']

function jobProgress(overrides: Partial<JobProgress> = {}): JobProgress {
  return {
    executionMode: 'sequential',
    jobStatus: 'queued',
    totalItems: 0,
    completedItems: 0,
    failedItems: 0,
    skippedItems: 0,
    cancelledItems: 0,
    pools: [],
    ...overrides,
  }
}

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

vi.mock('@/api/v2/jobs', async importOriginal => {
  const actual = await importOriginal<typeof import('@/api/v2/jobs')>()
  return {
    ...actual,
    jobsApi: {
      ...actual.jobsApi,
      cancel: vi.fn(),
      list: mocks.jobsList,
      retryFailed: mocks.jobsRetryFailed,
    },
  }
})

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
    mocks.listChapterPages.mockResolvedValue({
      items: [],
      nextCursor: null,
      pageOrderRevision: 1,
    })
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      chapterId: 'chapter-1',
      documentRevision: 2,
      defaultFontId: fontFamily,
      pageStyleDefaults,
      pageStyleSchemaVersion: 2,
      renderStatus: 'not_rendered',
      bubbles: [],
    })
  })

  it('submits selected pages to one durable backend job without changing navigation', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 7,
      id: 'page-1',
    })
    addTestImage(imageStore, '002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-1',
      documentRevision: 3,
      id: 'page-2',
    })
    imageStore.setCurrentImageIndex(0)

    const result = await useTranslation().translateSelectedImages({ pages: [2] })

    expect(result).toBe(true)
    expect(mocks.createChapterTranslationJob).toHaveBeenCalledWith(
      'chapter-1',
      ['page-2'],
      expect.objectContaining({
        mode: 'standard',
        styleSourcePageId: 'page-1',
        styleSourceDocumentRevision: 7,
      }),
    )
    expect(imageStore.currentImageIndex).toBe(0)
    expect(imageStore.images[1]?.translationStatus).toBe('processing')
    expect(mocks.toast.success).toHaveBeenCalledWith(
      '任务已加入后端任务中心，可安全关闭页面',
    )
  })

  it('keeps an accepted translation successful when task-center refresh fails', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 7,
      id: 'page-1',
    })
    mocks.jobsList.mockRejectedValue(new Error('snapshot unavailable'))

    await expect(useTranslation().translateCurrentImage()).resolves.toBe(true)

    expect(mocks.createChapterTranslationJob).toHaveBeenCalledOnce()
    expect(mocks.toast.error).not.toHaveBeenCalled()
    expect(mocks.toast.success).toHaveBeenCalledWith(
      '任务已加入后端任务中心，可安全关闭页面',
    )
  })

  it('locks job creation immediately and keeps the submitted execution mode', async () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    settingsStore.settings.parallel.enabled = true
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 7,
      id: 'page-1',
    })
    let resolveCreation!: () => void
    mocks.createChapterTranslationJob.mockImplementationOnce(() => new Promise(resolve => {
      resolveCreation = () => resolve({
        batchId: 'batch-1',
        jobIds: ['job-1'],
        status: 'queued',
      })
    }))
    const translation = useTranslation()

    const firstCreation = translation.translateCurrentImage()
    await vi.waitFor(() => expect(mocks.createChapterTranslationJob).toHaveBeenCalledOnce())
    expect(imageStore.isTranslationInProgress).toBe(true)
    settingsStore.settings.parallel.enabled = false

    await expect(translation.translateCurrentImage()).resolves.toBe(false)
    expect(mocks.createChapterTranslationJob).toHaveBeenCalledOnce()
    expect(mocks.toast.info).toHaveBeenCalledWith('已有翻译任务正在创建或执行')

    resolveCreation()
    await expect(firstCreation).resolves.toBe(true)
    expect(translation.progress.value.executionMode).toBe('parallel')
  })

  it('ignores malformed realtime progress instead of coercing it to zero', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 7,
      id: 'page-1',
    })
    const translation = useTranslation()
    await translation.translateCurrentImage()

    taskCenterStore.latestEvent = {
      eventId: 997,
      jobId: 'job-1',
      type: 'pipeline_progress',
      payload: {
        progress: {
          ...jobProgress({ jobStatus: 'running', totalItems: 1 }),
          completedItems: 'not-a-number',
        },
      },
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    expect(translation.progress.value).toMatchObject({
      current: 0,
      total: 1,
      status: 'queued',
    })
  })

  it('does not create a job when chapter settings preflight fails', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 4,
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
      documentRevision: 4,
      id: 'page-1',
    })

    const result = await useTranslation().removeTextOnly()

    expect(result).toBe(true)
    expect(mocks.createChapterRemoveTextJob).toHaveBeenCalledWith(
      'chapter-1',
      ['page-1'],
      'parallel',
      { pageId: 'page-1', documentRevision: 4 },
    )
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
  })

  it('does not silently fall back to per-page defaults without a committed style source', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })

    const result = await useTranslation().translateCurrentImage()

    expect(result).toBe(false)
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
    expect(mocks.toast.error).toHaveBeenCalledWith(
      '当前页文字样式尚未写入后端，未创建任务',
    )
  })

  it('rejects pages that are not owned by one backend chapter', async () => {
    const imageStore = useImageStore()
    addTestImage(imageStore, '001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 1,
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
        progress: jobProgress({
          jobStatus: 'running',
          totalItems: 2,
          completedItems: 1,
        }),
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
      nextCursor: null,
      pageOrderRevision: 1,
    })
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      chapterId: 'chapter-1',
      documentRevision: 2,
      defaultFontId: fontFamily,
      pageStyleDefaults: {
        ...pageStyleDefaults,
        layoutDirection: 'horizontal',
        textColor: '#123456',
        useAutoTextColor: false,
      },
      pageStyleSchemaVersion: 2,
      renderStatus: 'ready',
      bubbles: [{
        bubbleId: 'bubble-1',
        ordinal: 1,
        fontId: fontFamily,
        updatedRevision: 2,
        payload: {
          originalText: 'こんにちは',
          translatedText: '你好',
          textboxText: '',
          coords: [1, 2, 30, 40],
          polygon: [],
          fontSize: 24,
          textDirection: 'vertical',
          autoTextDirection: 'vertical',
          textColor: '#123456',
          fillColor: '#ffffff',
          rotationAngle: 0,
          position: { x: 0, y: 0 },
          strokeEnabled: false,
          strokeColor: '#ffffff',
          strokeWidth: 0,
          lineSpacing: 1.2,
          inlineAlign: 'center',
          blockAlign: 'center',
          inpaintMethod: 'solid',
          autoFgColor: null,
          autoBgColor: null,
          colorConfidence: 0,
          textlines: [],
          ocrResult: null,
        },
      }],
    })

    await expect(useTranslation().translateCurrentImage()).resolves.toBe(true)
    taskCenterStore.latestEvent = {
      eventId: 999,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: new Date().toISOString(),
    }
    await nextTick()

    await vi.waitFor(() => expect(mocks.listChapterPages).toHaveBeenCalledWith(
      'chapter-1',
      { all: true },
    ))
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

  it('preserves a page selected while terminal chapter refresh is pending', async () => {
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
    let resolvePages!: () => void
    mocks.listChapterPages.mockImplementationOnce(() => new Promise(resolve => {
      resolvePages = () => resolve({
        items: [
          {
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
          },
          {
            id: 'page-2',
            chapterId: 'chapter-1',
            ordinal: 2,
            logicalSourcePath: '002.png',
            width: 100,
            height: 200,
            sourceRevision: 1,
            documentRevision: 2,
            renderedRevision: 2,
            renderStatus: 'ready',
            detectionState: 'completed',
            sourceUrl: '/api/v2/assets/source-2',
            thumbnailSourceUrl: '/api/v2/assets/thumb-2',
            cleanUrl: '/api/v2/assets/clean-2',
            translatedUrl: '/api/v2/assets/translated-2',
          },
        ],
        nextCursor: null,
        pageOrderRevision: 1,
      })
    }))
    mocks.getPageDocument.mockImplementation(async (pageId: string) => {
      const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
      return {
        pageId,
        chapterId: 'chapter-1',
        documentRevision: 2,
        defaultFontId: fontFamily,
        pageStyleDefaults,
        pageStyleSchemaVersion: 2,
        renderStatus: 'ready',
        bubbles: [],
      }
    })
    const translation = useTranslation()
    await translation.translateCurrentImage()

    taskCenterStore.latestEvent = {
      eventId: 1002,
      jobId: 'job-1',
      type: 'job_finished',
      payload: {},
      createdAt: new Date().toISOString(),
    }
    await vi.waitFor(() => expect(mocks.listChapterPages).toHaveBeenCalledOnce())
    imageStore.setCurrentImageIndex(1)
    resolvePages()

    await vi.waitFor(() => expect(mocks.getPageDocument).toHaveBeenCalledWith('page-2'))
    expect(imageStore.currentImage?.id).toBe('page-2')
    expect(mocks.getPageDocument).not.toHaveBeenCalledWith('page-1')
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
      progress: jobProgress({ jobStatus: 'completed' }),
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
      nextCursor: null,
      pageOrderRevision: 2,
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
      progress: jobProgress({
        jobStatus: 'completed',
        totalItems: 2,
        completedItems: 2,
      }),
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
      documentRevision: 1,
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
      progress: jobProgress({
        totalItems: 1,
        completedItems: 0,
        jobStatus: 'running',
      }),
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
      progress: jobProgress({
        totalItems: 1,
        completedItems: 1,
        jobStatus: 'completed',
      }),
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
            progress: jobProgress({
              jobStatus: 'completed_with_errors',
              totalItems: 1,
              failedItems: 1,
            }),
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
