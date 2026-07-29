import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTranslation } from '@/composables/useTranslationPipeline'
import { useImageStore } from '@/stores/imageStore'

const mocks = vi.hoisted(() => ({
  createChapterTranslationJob: vi.fn(),
  jobsList: vi.fn(),
  jobsRetryFailed: vi.fn(),
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
  listChapterPages: vi.fn(),
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
