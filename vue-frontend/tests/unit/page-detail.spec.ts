import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { useInsightStore } from '@/stores/insightStore'

const { getPageDataMock, reanalyzePageMock, getPageImageUrlMock } = vi.hoisted(() => ({
  getPageDataMock: vi.fn(),
  reanalyzePageMock: vi.fn(),
  getPageImageUrlMock: vi.fn(() => '/page.png'),
}))

vi.mock('@/api/insight', () => ({
  getPageData: getPageDataMock,
  reanalyzePage: reanalyzePageMock,
  getPageImageUrl: getPageImageUrlMock,
}))

import PageDetail from '@/components/insight/PageDetail.vue'

describe('PageDetail', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    store.setAnalysisStatus('idle')
    store.dataRefreshKey = 0

    getPageDataMock.mockReset()
    getPageDataMock.mockResolvedValue({
      success: true,
      analysis: {
        page_num: 3,
        page_summary: '旧摘要',
        panels: [],
      },
    })

    reanalyzePageMock.mockReset()
    reanalyzePageMock.mockResolvedValue({
      success: true,
      task_id: 'task-123',
    })
  })

  it('starts async reanalyze and refreshes on dataRefreshKey', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    store.selectPage(3)
    store.setBookTotalPages(20)
    store.setAnalysisStatus('idle')
    store.dataRefreshKey = 0

    const setStatusSpy = vi.spyOn(store, 'setAnalysisStatus')
    const setTaskSpy = vi.spyOn(store, 'setCurrentTaskId')

    const wrapper = mount(PageDetail, {
      global: {
        plugins: [pinia],
      },
    })
    await flushPromises()

    expect(getPageDataMock).toHaveBeenCalledTimes(1)

    const reanalyzeButton = wrapper.findAll('button').find(button => button.text().includes('重新分析'))
    expect(reanalyzeButton).toBeTruthy()

    await reanalyzeButton!.trigger('click')
    await flushPromises()

    expect(reanalyzePageMock).toHaveBeenCalledWith('book-1', 3)
    expect(setTaskSpy).toHaveBeenCalledWith('task-123')
    expect(setStatusSpy).toHaveBeenCalledWith('running')

    // 不应在启动后立即当作同步完成并刷新详情
    expect(getPageDataMock).toHaveBeenCalledTimes(1)

    store.triggerDataRefresh()
    await nextTick()
    await flushPromises()

    // 分析完成信号到达后自动刷新当前页详情
    expect(getPageDataMock).toHaveBeenCalledTimes(2)
  })
})
