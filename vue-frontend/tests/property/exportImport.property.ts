import { readFileSync } from 'node:fs'

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useExportImport } from '@/composables/useExportImport'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({
  commitChapterTextImport: vi.fn(),
  createChapterExportJob: vi.fn(),
  getChapterTextExportUrl: vi.fn(),
  jobGet: vi.fn(),
  jobList: vi.fn(),
  previewChapterTextImport: vi.fn(),
  triggerUrlDownload: vi.fn(),
  toast: {
    error: vi.fn(),
    info: vi.fn(),
    removeToast: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  },
}))

vi.mock('@/api/v2/translation', () => ({
  commitChapterTextImport: mocks.commitChapterTextImport,
  createChapterExportJob: mocks.createChapterExportJob,
  getChapterTextExportUrl: mocks.getChapterTextExportUrl,
  previewChapterTextImport: mocks.previewChapterTextImport,
}))

vi.mock('@/api/v2/jobs', () => ({
  jobsApi: {
    get: mocks.jobGet,
    list: mocks.jobList,
  },
}))

vi.mock('@/utils/browserDownload', () => ({
  triggerUrlDownload: mocks.triggerUrlDownload,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => mocks.toast,
}))

function seedChapter() {
  const store = useImageStore()
  addTestImage(store, '001.png', '/api/v2/assets/source-1', {
    chapterId: 'chapter-1',
    id: 'page-1',
    sourceAssetUrl: '/api/v2/assets/source-1',
  })
  addTestImage(store, '002.png', '/api/v2/assets/source-2', {
    chapterId: 'chapter-1',
    id: 'page-2',
    sourceAssetUrl: '/api/v2/assets/source-2',
  })
  return store
}

describe('backend-owned export/import contracts', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mocks.getChapterTextExportUrl.mockReturnValue('/api/v2/chapters/chapter-1/text-export')
    mocks.toast.info.mockReturnValue(101)
    mocks.jobList.mockResolvedValue({ items: [], queueRevision: 1 })
  })

  it('downloads text from the backend export endpoint', () => {
    seedChapter()

    useExportImport().exportText()

    expect(mocks.getChapterTextExportUrl).toHaveBeenCalledWith('chapter-1')
    expect(mocks.triggerUrlDownload).toHaveBeenCalledWith(
      '/api/v2/chapters/chapter-1/text-export',
      'chapter-chapter-1-text.json',
    )
  })

  it('previews text on the backend and commits only matched changed pages', async () => {
    seedChapter()
    const matchingPage = {
      baseDocumentRevision: 7,
      changes: [{
        bubbleId: 'bubble-1',
        differences: { translatedText: { after: '你好', before: 'hello' } },
        fields: { translatedText: '你好' },
      }],
      issues: [],
      pageId: 'page-1',
      sourceAssetId: 'source-1',
      sourceChecksum: 'sha256:source-1',
      status: 'match' as const,
    }
    mocks.previewChapterTextImport.mockResolvedValue({
      chapterId: 'chapter-1',
      conflictedPages: 1,
      matchedPages: 1,
      pages: [
        matchingPage,
        {
          ...matchingPage,
          baseDocumentRevision: null,
          changes: [],
          issues: ['revision_conflict'],
          pageId: 'page-2',
          status: 'conflict',
        },
      ],
      schemaVersion: 2,
    })
    mocks.commitChapterTextImport.mockResolvedValue({
      batchId: 'batch-1',
      jobIds: ['job-import-1'],
      status: 'queued',
    })

    const file = new File(['{}'], 'translation.json', { type: 'application/json' })
    await useExportImport().importText(file)

    expect(mocks.previewChapterTextImport).toHaveBeenCalledWith('chapter-1', file)
    expect(mocks.commitChapterTextImport).toHaveBeenCalledWith('chapter-1', [matchingPage])
    expect(mocks.jobList).toHaveBeenCalledTimes(2)
    expect(mocks.toast.success).toHaveBeenCalledWith(
      '已提交 1 页文本导入，可安全关闭页面；跳过 1 页冲突',
    )
  })

  it('creates a durable backend export and downloads its artifact', async () => {
    seedChapter()
    mocks.createChapterExportJob.mockResolvedValue({
      batchId: 'batch-export',
      jobIds: ['job-export-1'],
      status: 'queued',
    })
    mocks.jobGet.mockResolvedValue({
      artifacts: [{
        assetId: 'asset-export',
        expiresAt: null,
        kind: 'chapter_export',
        url: '/api/v2/assets/asset-export',
      }],
      id: 'job-export-1',
      progress: { completedItems: 2, failedItems: 0, totalItems: 2 },
      status: 'completed',
    })
    mocks.jobList.mockImplementation((scope: string) => Promise.resolve({
      items: scope === 'queue'
        ? [{
            jobId: 'job-export-1',
            progress: { completedItems: 2, failedItems: 0, totalItems: 2 },
            status: 'completed',
          }]
        : [],
      queueRevision: 1,
    }))

    await useExportImport().downloadAllImages('cbz')

    expect(mocks.createChapterExportJob).toHaveBeenCalledWith(
      'chapter-1',
      'cbz',
      ['page-1', 'page-2'],
    )
    expect(mocks.jobGet).toHaveBeenCalledWith('job-export-1')
    expect(mocks.triggerUrlDownload).toHaveBeenCalledWith(
      '/api/v2/assets/asset-export?download=1&filename=chapter-export.cbz',
    )
    expect(mocks.toast.removeToast).toHaveBeenCalledWith(101)
  })

  it('contains no browser-side payload generation or legacy download sessions', () => {
    const source = readFileSync('src/composables/useExportImport.ts', 'utf8')

    expect(source).not.toContain('executeRender')
    expect(source).not.toContain('FileReader')
    expect(source).not.toContain('readAsDataURL')
    expect(source).not.toContain('downloadStartSession')
    expect(source).not.toContain('exportTextToJson')
    expect(source).not.toContain('WatchStopHandle')
    expect(source).not.toContain('30 * 60')
    expect(source).not.toContain('Math.round(complete / total')
    expect(source).toContain('complete / total * 90 + 5')
    expect(source).toContain('createChapterExportJob')
    expect(source).toContain('previewChapterTextImport')
    expect(source).toContain('taskCenterStore.waitForJob')
  })
})
