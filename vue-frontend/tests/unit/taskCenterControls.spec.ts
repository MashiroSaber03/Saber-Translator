import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import TaskBatchAnalysisModal from '@/components/task-center/TaskBatchAnalysisModal.vue'
import TaskCenterDrawer from '@/components/task-center/TaskCenterDrawer.vue'
import TaskStatusBadge from '@/components/task-center/TaskStatusBadge.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { BookData } from '@/types/bookshelf'
import type { V2JobDetail } from '@/api/v2/jobs'

const { createAnalysisMock, getBookDetailMock, postMock } = vi.hoisted(() => ({
  createAnalysisMock: vi.fn(),
  getBookDetailMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  const apiClient = {
    delete: vi.fn(),
    get: vi.fn(),
    patch: vi.fn(),
    post: postMock,
    put: vi.fn(),
    upload: vi.fn(),
  }
  return { apiClient, default: apiClient }
})

vi.mock('@/api/bookshelf', () => ({
  getBookDetail: getBookDetailMock,
}))

vi.mock('@/api/v2/insight', () => ({
  createInsightAnalysisJob: createAnalysisMock,
}))

let wrapper: VueWrapper | null = null
let pinia: Pinia

beforeEach(() => {
  vi.clearAllMocks()
  pinia = createPinia()
  setActivePinia(pinia)
  const store = useBookshelfStore()
  store.books = [
    {
      id: 'book-1',
      title: '测试漫画',
      chapterCount: 2,
    },
  ]
  getBookDetailMock.mockResolvedValue({
    id: 'book-1',
    title: '测试漫画',
    chapters: [
      {
        id: 'chapter-1',
        title: '第一话',
        order: 0,
        imageCount: 3,
        page_count: 3,
      },
      {
        id: 'chapter-empty',
        title: '空章节',
        order: 1,
        imageCount: 0,
        page_count: 0,
      },
    ],
  })
  createAnalysisMock.mockResolvedValue({
    batchId: 'batch-1',
    jobIds: ['job-1'],
    runId: 'run-1',
    status: 'queued',
  })
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = null
  document.body.innerHTML = ''
  document.body.style.overflow = ''
})

describe('task center controls', () => {
  it('loads books for an initially open batch modal and reports the load failure', async () => {
    const store = useBookshelfStore()
    store.books = []
    vi.spyOn(store, 'loadBooks').mockRejectedValue(new Error('书籍服务暂不可用'))

    wrapper = mount(TaskBatchAnalysisModal, {
      attachTo: document.body,
      props: { modelValue: true },
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(document.body.textContent).toContain('书籍服务暂不可用')
  })

  it('keeps durable interrupted counts after the task snapshot loads', () => {
    const store = useTaskCenterStore()
    store.snapshotLoaded = true

    wrapper = mount(TaskStatusBadge, {
      props: {
        bookId: 'book-1',
        summary: { interrupted: 3 },
      },
      global: { plugins: [pinia] },
    })

    expect(wrapper.get('button').text()).toBe('中断 3')
  })

  it('submits Worker model release through the system endpoint', async () => {
    postMock.mockResolvedValue({
      commandId: 'command-1',
      kind: 'release_models',
      status: 'pending',
    })
    const { releaseWorkerModelCache } = await import('@/api/v2/system')

    await expect(releaseWorkerModelCache()).resolves.toMatchObject({
      kind: 'release_models',
      status: 'pending',
    })
    expect(postMock).toHaveBeenCalledWith('/api/v2/system/release-models')
  })

  it('creates a chapter-scoped Insight batch from the task center', async () => {
    wrapper = mount(TaskBatchAnalysisModal, {
      attachTo: document.body,
      props: { modelValue: true },
      global: { plugins: [pinia] },
    })
    const selects = wrapper.findAllComponents(UiSelect)
    selects[0]!.vm.$emit('change', 'book-1')
    await flushPromises()
    selects[1]!.vm.$emit('change', 'chapter')
    await flushPromises()

    const chapterChoice = wrapper.findAllComponents(UiCheckbox)[0]
    expect(chapterChoice?.props('label')).toBe('第一话')
    chapterChoice!.vm.$emit('change', true)
    await flushPromises()
    const submit = [...document.body.querySelectorAll('button')].find(
      button => button.textContent?.trim() === '加入任务队列'
    )
    expect(submit).toBeTruthy()
    submit!.click()
    await flushPromises()

    expect(createAnalysisMock).toHaveBeenCalledWith({
      bookId: 'book-1',
      scope: 'chapter',
      chapterIds: ['chapter-1'],
    })
    expect(wrapper.emitted('created')?.[0]?.[0]).toMatchObject({
      batchId: 'batch-1',
      jobIds: ['job-1'],
    })
  })

  it('keeps chapters from the latest selected book when requests finish out of order', async () => {
    let resolveFirst: ((value: BookData) => void) | undefined
    let resolveSecond: ((value: BookData) => void) | undefined
    getBookDetailMock.mockImplementation(
      (bookId: string) =>
        new Promise(resolve => {
          if (bookId === 'book-1') resolveFirst = resolve
          else resolveSecond = resolve
        })
    )
    wrapper = mount(TaskBatchAnalysisModal, {
      attachTo: document.body,
      props: { modelValue: true },
      global: { plugins: [pinia] },
    })
    const bookSelect = wrapper.findAllComponents(UiSelect)[0]!

    bookSelect.vm.$emit('change', 'book-1')
    bookSelect.vm.$emit('change', 'book-2')
    resolveSecond?.({
      id: 'book-2',
      title: '第二本',
      chapters: [{ id: 'chapter-2', title: '第二话', order: 0, imageCount: 2 }],
    })
    await flushPromises()
    resolveFirst?.({
      id: 'book-1',
      title: '第一本',
      chapters: [{ id: 'chapter-1', title: '第一话', order: 0, imageCount: 1 }],
    })
    await flushPromises()
    wrapper.findAllComponents(UiSelect)[1]!.vm.$emit('change', 'chapter')
    await flushPromises()

    const labels = wrapper.findAllComponents(UiCheckbox).map(item => item.props('label'))
    expect(labels).toEqual(['第二话'])
  })

  it('shows batch continuation for paused members outside the waiting zone', async () => {
    const store = useTaskCenterStore()
    store.drawerOpen = true
    store.queue = [
      {
        jobId: 'paused-job',
        batchId: 'batch-1',
        kind: 'translation',
        retryOfJobId: null,
        retryMode: null,
        status: 'paused',
        queueRank: 1,
        progress: {
          executionMode: 'sequential',
          jobStatus: 'paused',
          totalItems: 1,
          completedItems: 0,
          failedItems: 0,
          skippedItems: 0,
          cancelledItems: 0,
          pools: [],
        },
        target: {},
        createdAt: null,
      },
    ]
    wrapper = mount(TaskCenterDrawer, {
      attachTo: document.body,
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const labels = [...document.body.querySelectorAll('button')].map(button =>
      button.textContent?.trim()
    )
    expect(labels).toContain('全部继续')
  })

  it('shows every durable outcome in a parallel step pool', async () => {
    const store = useTaskCenterStore()
    store.drawerOpen = true
    store.queue = [
      {
        jobId: 'parallel-job',
        batchId: null,
        kind: 'translation',
        retryOfJobId: null,
        retryMode: null,
        status: 'running',
        queueRank: 1,
        progress: {
          executionMode: 'parallel',
          jobStatus: 'running',
          totalItems: 2,
          completedItems: 0,
          failedItems: 0,
          skippedItems: 0,
          cancelledItems: 0,
          pools: [
            {
              kind: 'translate',
              total: 8,
              completed: 4,
              failed: 1,
              skipped: 1,
              cancelled: 0,
              waiting: 1,
              processing: 1,
              lockWaiting: false,
              current: [
                {
                  itemId: 'item-1',
                  pageId: 'page-1',
                  itemOrdinal: 1,
                  stepId: 'step-1',
                  stepOrdinal: 4,
                },
              ],
            },
          ],
        },
        target: {},
        createdAt: null,
      },
    ]
    wrapper = mount(TaskCenterDrawer, {
      attachTo: document.body,
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(document.body.querySelector('.task-job__pools')?.textContent).toContain(
      '完成 4 / 8 · 失败 1 · 跳过 1 · 取消 0 · 处理中 1 · 等待 1'
    )
  })

  it('uses page progress in Insight details instead of counting the finalization item', async () => {
    const store = useTaskCenterStore()
    const insightJob = {
      jobId: 'insight-job',
      batchId: null,
      kind: 'insight_analysis' as const,
      retryOfJobId: null,
      retryMode: null,
      status: 'running' as const,
      queueRank: 1,
      progress: {
        executionMode: 'sequential' as const,
        jobStatus: 'running' as const,
        totalItems: 7,
        completedItems: 2,
        failedItems: 0,
        skippedItems: 0,
        cancelledItems: 0,
        pools: [
          {
            kind: 'insight_analyze_page',
            total: 6,
            completed: 1,
            failed: 1,
            skipped: 1,
            cancelled: 0,
            waiting: 3,
            processing: 0,
            lockWaiting: false,
            current: [],
          },
        ],
      },
      target: { book: '测试漫画', pageCount: 6 },
      createdAt: null,
    }
    store.drawerOpen = true
    store.queue = [insightJob]
    store.selectedDetailJobId = insightJob.jobId
    store.selectedDetail = {
      ...insightJob,
      counts: {
        total: 7,
        pending: 5,
        running: 0,
        completed: 2,
        failed: 0,
        skipped: 0,
        cancelled: 0,
      },
      durationMs: null,
      error: null,
      configSummary: {},
      items: [],
      failedItems: [],
      artifacts: [],
      recentEvents: [],
    } satisfies V2JobDetail

    wrapper = mount(TaskCenterDrawer, {
      attachTo: document.body,
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const detailText = document.body.querySelector('.task-job__detail')?.textContent ?? ''
    expect(detailText).toContain('页进度')
    expect(detailText).toContain('3 / 6')
    expect(detailText).not.toContain('2 / 7')
  })

  it('treats the open task center as the active dialog', async () => {
    const trigger = document.createElement('button')
    document.body.appendChild(trigger)
    trigger.focus()
    const store = useTaskCenterStore()
    store.drawerOpen = true
    wrapper = mount(TaskCenterDrawer, {
      attachTo: document.body,
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const panel = document.body.querySelector<HTMLElement>('.task-center__panel')
    expect(document.body.style.overflow).toBe('hidden')
    expect(document.activeElement).toBe(panel)

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    await flushPromises()

    expect(store.drawerOpen).toBe(false)
    expect(document.body.style.overflow).toBe('')
    expect(document.activeElement).toBe(trigger)
  })
})
