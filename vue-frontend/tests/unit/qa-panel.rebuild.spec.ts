import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { V2Job } from '@/api/v2/jobs'

const {
  rebuildEmbeddingsMock,
  rebuildCompressedContextMock,
  regenerateOverviewMock,
  getQAStatusMock,
  sendChatMock,
  showToastMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  rebuildEmbeddingsMock: vi.fn(),
  rebuildCompressedContextMock: vi.fn(),
  regenerateOverviewMock: vi.fn(),
  getQAStatusMock: vi.fn(),
  sendChatMock: vi.fn(),
  showToastMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  sendChat: sendChatMock,
  rebuildEmbeddings: rebuildEmbeddingsMock,
  rebuildCompressedContext: rebuildCompressedContextMock,
  regenerateOverview: regenerateOverviewMock,
  getQAStatus: getQAStatusMock,
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

import QAPanel from '@/components/insight/QAPanel.vue'

function vectorJob(overrides: Partial<V2Job> = {}): V2Job {
  return {
    jobId: 'vector-job-1',
    kind: 'vector_rebuild',
    retryOfJobId: null,
    retryMode: null,
    status: 'running',
    queueRank: 1,
    bookId: 'book-1',
    progress: {
      executionMode: 'sequential',
      jobStatus: 'running',
      totalItems: 10,
      completedItems: 3,
      failedItems: 0,
      skippedItems: 0,
      cancelledItems: 0,
      pools: [],
    },
    target: {},
    createdAt: null,
    ...overrides,
  }
}

describe('QAPanel vector rebuild task projection', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)

    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    insightStore.setLoading(false)

    rebuildEmbeddingsMock.mockReset()
    rebuildCompressedContextMock.mockReset()
    regenerateOverviewMock.mockReset()
    getQAStatusMock.mockReset()
    getQAStatusMock.mockResolvedValue({
      available: true,
      reason: null,
    })
    sendChatMock.mockReset()
    showToastMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('starts once, then projects progress and completion from the global task center', async () => {
    rebuildEmbeddingsMock.mockResolvedValue({
      success: true,
      task_id: 'vector-job-1',
    })
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    const taskCenterStore = useTaskCenterStore()
    const refreshSpy = vi.spyOn(taskCenterStore, 'refresh').mockResolvedValue(undefined)

    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })

    await wrapper.find('button[title="重建向量索引"]').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '重建向量索引',
      message: '确定要重建向量索引吗？这将删除现有的向量数据并重新构建，可能需要一些时间。',
      confirmText: '重建',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(rebuildEmbeddingsMock).toHaveBeenCalledWith('book-1')
    expect(refreshSpy).toHaveBeenCalledTimes(1)

    taskCenterStore.queue = [vectorJob()]
    await flushPromises()
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建中 (3/10)')
    expect(insightStore.isLoading).toBe(true)

    taskCenterStore.queue = []
    taskCenterStore.history = [vectorJob({
      status: 'completed',
      queueRank: null,
      progress: {
        executionMode: 'sequential',
        jobStatus: 'completed',
        totalItems: 10,
        completedItems: 10,
        failedItems: 0,
        skippedItems: 0,
        cancelledItems: 0,
        pools: [],
      },
    })]
    await flushPromises()

    expect(showToastMock).toHaveBeenCalledWith('向量索引重建完成', 'success', 6000)
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(insightStore.isLoading).toBe(false)
  })

  it('consumes backend QA readiness and blocks exact questions until vectors are rebuilt', async () => {
    getQAStatusMock.mockResolvedValue({
      available: false,
      reason: 'vector_missing',
      repairAction: 'vector_rebuild',
    })
    rebuildEmbeddingsMock.mockResolvedValue({
      success: true,
      task_id: 'vector-job-1',
    })
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    const taskCenterStore = useTaskCenterStore()
    vi.spyOn(taskCenterStore, 'refresh').mockResolvedValue(undefined)

    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(getQAStatusMock).toHaveBeenCalledWith('book-1', 'precise')
    expect(wrapper.get('.qa-panel__status').text()).toContain('精确问答暂不可用')
    expect(wrapper.get('.qa-panel__status').text()).toContain('重建向量')
    expect(wrapper.get('textarea[placeholder="输入你的问题..."]').attributes('disabled')).toBeDefined()

    await wrapper.get('.qa-panel__status button').trigger('click')
    await flushPromises()

    expect(rebuildEmbeddingsMock).toHaveBeenCalledWith('book-1')
    expect(sendChatMock).not.toHaveBeenCalled()
  })

  it('resumes an already-running rebuild from the global task snapshot on mount', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.queue = [vectorJob()]

    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(rebuildEmbeddingsMock).not.toHaveBeenCalled()
    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建中 (3/10)')
    expect(insightStore.isLoading).toBe(true)
  })

  it('does not leave the QA panel loading for an interrupted rebuild', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.queue = [vectorJob({ status: 'interrupted' })]

    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(insightStore.isLoading).toBe(false)
  })

  it('drops the old rebuild projection when another book is selected', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const insightStore = useInsightStore()
    insightStore.currentBookId = 'book-1'
    const taskCenterStore = useTaskCenterStore()
    taskCenterStore.queue = [vectorJob()]

    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()
    expect(insightStore.isLoading).toBe(true)

    insightStore.currentBookId = 'book-2'
    await flushPromises()

    expect(wrapper.find('button[title="重建向量索引"]').text()).toContain('重建向量')
    expect(insightStore.isLoading).toBe(false)
  })

  it('contains no page-local timer or legacy task-status polling', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/QAPanel.vue'), 'utf8')

    expect(source).not.toContain('setInterval(')
    expect(source).not.toContain('setTimeout(')
    expect(source).not.toContain('getRebuildEmbeddingsStatus')
    expect(source).toContain('taskCenterStore.queue')
  })
})
