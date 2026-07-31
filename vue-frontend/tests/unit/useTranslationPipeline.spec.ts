import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslation } from '@/composables/useTranslationPipeline'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'

const mocks = vi.hoisted(() => ({
  createChapterTranslationJob: vi.fn(),
  getPageDocument: vi.fn(),
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
  createChapterTranslationJob: mocks.createChapterTranslationJob,
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
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
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      documentRevision: 2,
      defaultFontId: 'font-default',
      pageStyleDefaults: {},
      bubbles: [],
    })
  })

  it('submits selected pages to one durable backend job without changing navigation', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    imageStore.addImage('002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-1',
      id: 'page-2',
    })
    imageStore.setCurrentImageIndex(0)

    const result = await useTranslation().translatePages([1], 'standard')

    expect(result.success).toBe(true)
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

  it('rejects pages that are not owned by one backend chapter', async () => {
    const imageStore = useImageStore()
    imageStore.addImage('001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    imageStore.addImage('002.png', '/api/v2/assets/source-2', {
      chapterId: 'chapter-2',
      id: 'page-2',
    })

    const result = await useTranslation().translatePages([0, 1], 'standard')

    expect(result.success).toBe(false)
    expect(mocks.createChapterTranslationJob).not.toHaveBeenCalled()
  })

  it('rehydrates the current page document when a durable job finishes', async () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const taskCenterStore = useTaskCenterStore()
    imageStore.addImage('001.png', '/api/v2/assets/source-1', {
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
        thumbnailTranslatedUrl: '/api/v2/assets/translated-thumb-1',
      }],
    })
    mocks.getPageDocument.mockResolvedValue({
      pageId: 'page-1',
      documentRevision: 2,
      defaultFontId: 'font-default',
      pageStyleDefaults: {},
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

    await useTranslation().translatePages([0], 'standard')
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
  })

  it('reconciles a tracked job from durable queue and history snapshots', async () => {
    const imageStore = useImageStore()
    const taskCenterStore = useTaskCenterStore()
    imageStore.addImage('001.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      id: 'page-1',
    })
    const translation = useTranslation()
    await translation.translatePages([0], 'standard')

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
    imageStore.addImage('001.png', '/api/v2/assets/source-1', {
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
    expect(source).not.toContain('usePipeline')
    expect(source).not.toContain('executeRender')
    expect(source).not.toContain('sessionStore')
    expect(source).not.toContain('base64')
  })
})
