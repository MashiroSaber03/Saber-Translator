import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia, type Pinia } from 'pinia'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import TaskBatchAnalysisModal from '@/components/task-center/TaskBatchAnalysisModal.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'

const {
  createAnalysisMock,
  getBookDetailMock,
  postMock,
} = vi.hoisted(() => ({
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
})

describe('task center controls', () => {
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
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/system/release-models',
    )
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
      button => button.textContent?.trim() === '加入任务队列',
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
})
