import { defineComponent, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import { useContinuationState } from './useContinuationState'

const { prepareContinuationMock, getCharactersMock, syncContinuationAnalysisMock } = vi.hoisted(() => ({
  prepareContinuationMock: vi.fn(),
  getCharactersMock: vi.fn(),
  syncContinuationAnalysisMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  prepareContinuation: prepareContinuationMock,
  getCharacters: getCharactersMock,
  syncContinuationAnalysis: syncContinuationAnalysisMock,
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

describe('useContinuationState', () => {
  it('keeps continuation blocked when preparation reports missing prerequisites', async () => {
    prepareContinuationMock.mockResolvedValue({
      ready: false,
      message: '续写功能需要故事概要',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockResolvedValue({ items: [], nextCursor: null })
    const state = useContinuationState(ref('book-1'))
    await state.initializeData()

    expect(state.isDataReady.value).toBe(false)
    expect(state.errorMessage.value).toContain('故事概要')
  })

  it('shows a persistent error when character loading fails during initialization', async () => {
    prepareContinuationMock.mockResolvedValue({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockRejectedValue(new Error('角色接口不可用'))

    const state = useContinuationState(ref('book-1'))
    await state.initializeData()

    expect(state.isDataReady.value).toBe(true)
    expect(state.errorMessage.value).toContain('角色接口不可用')
    expect(state.messageType.value).toBe('error')
  })

  it('resets stale continuation state before applying a fresh empty payload', async () => {
    prepareContinuationMock.mockResolvedValue({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockResolvedValue({
      items: [
        {
          name: '主角',
          aliases: [],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      ],
      nextCursor: null,
    })
    const state = useContinuationState(ref('book-1'))
    state.pageCount.value = 22
    state.styleRefPages.value = 7
    state.continuationDirection.value = '保留旧方向'
    state.initialReferenceTokens.value = ['old-reference']
    state.chapterScript.value = {
      chapter_title: '旧脚本',
      page_count: 8,
      script_text: '旧内容',
      generated_at: '2026-05-11T00:00:00',
    }
    state.pages.value = [
      {
        page_number: 1,
        continuity_text: '旧承接',
        story_text: '旧剧情',
        dialogue_text: '旧对白',
        characters: ['旧角色'],
        final_prompt: '已有最终提示词',
        image_url: '/tmp/previous.png',
        previous_url: '',
        status: 'generated',
      },
    ]

    await state.initializeData()

    expect(state.isDataReady.value).toBe(true)
    expect(state.initialReferenceTokens.value).toEqual([])
    expect(state.pageCount.value).toBe(10)
    expect(state.styleRefPages.value).toBe(3)
    expect(state.continuationDirection.value).toBe('')
    expect(state.chapterScript.value).toBeNull()
    expect(state.pages.value).toEqual([])
    expect(state.characters.value.map(character => character.name)).toEqual(['主角'])
  })

  it('ignores stale initialization responses after the selected book changes', async () => {
    const bookId = ref('book-1')
    const firstPrepare = deferred<{
      ready: boolean
      message: string
      saved_data: { script: null; pages: []; config: null; has_data: boolean }
    }>()
    const secondPrepare = deferred<{
      ready: boolean
      message: string
      saved_data: { script: null; pages: []; config: null; has_data: boolean }
    }>()
    prepareContinuationMock.mockImplementation((id: string) => (
      id === 'book-1' ? firstPrepare.promise : secondPrepare.promise
    ))
    getCharactersMock.mockImplementation((id: string) => Promise.resolve({
      items: [
        {
          name: id === 'book-1' ? '旧书角色' : '新书角色',
          aliases: [],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      ],
      nextCursor: null,
    }))

    const state = useContinuationState(bookId)
    const firstLoad = state.initializeData()
    bookId.value = 'book-2'
    const secondLoad = state.initializeData()

    secondPrepare.resolve({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    await secondLoad

    expect(state.characters.value.map(character => character.name)).toEqual(['新书角色'])

    firstPrepare.resolve({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    await firstLoad

    expect(state.characters.value.map(character => character.name)).toEqual(['新书角色'])
  })

  it('merges character forms from the next backend cursor page', async () => {
    prepareContinuationMock.mockResolvedValue({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockReset()
    getCharactersMock
      .mockResolvedValueOnce({
        items: [{
          name: '主角',
          aliases: [],
          description: '第一页数据',
          forms: [{
            form_id: 'form-1',
            form_name: '常服',
            description: '',
            reference_image: '/form-1.png',
          }],
          reference_image: '/form-1.png',
          enabled: true,
        }],
        nextCursor: 100,
      })
      .mockResolvedValueOnce({
        items: [{
          name: '主角',
          aliases: ['hero'],
          description: '最新数据',
          forms: [{
            form_id: 'form-2',
            form_name: '战斗服',
            description: '',
            reference_image: '/form-2.png',
          }],
          reference_image: '/form-2.png',
          enabled: true,
        }],
        nextCursor: null,
      })

    const state = useContinuationState(ref('book-1'))
    await state.initializeData()
    await state.loadMoreCharacterForms()

    expect(getCharactersMock).toHaveBeenNthCalledWith(1, 'book-1')
    expect(getCharactersMock).toHaveBeenNthCalledWith(2, 'book-1', 100)
    expect(state.characters.value[0]).toMatchObject({
      aliases: ['hero'],
      description: '最新数据',
      forms: [
        { form_id: 'form-1' },
        { form_id: 'form-2' },
      ],
    })
    expect(state.hasMoreCharacterForms.value).toBe(false)
  })

  it('syncs analysis data without clearing existing continuation payloads', async () => {
    prepareContinuationMock.mockResolvedValue({
      ready: true,
      message: '续写数据已就绪',
      saved_data: {
        script: {
          chapter_title: '旧脚本',
          page_count: 2,
          script_text: '原有内容',
          generated_at: '2026-05-11T00:00:00',
        },
        pages: [
          {
            page_number: 1,
            continuity_text: '原有承接',
            story_text: '原有剧情',
            dialogue_text: '',
            characters: ['主角'],
            final_prompt: '已有提示词',
            image_url: '/tmp/previous.png',
            previous_url: '',
            status: 'generated',
          },
        ],
        config: {
          page_count: 12,
          style_reference_pages: 4,
          continuation_direction: '保留当前续写',
        },
        has_data: true,
      },
    })
    getCharactersMock.mockResolvedValue({
      items: [
        {
          name: '主角',
          aliases: [],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      ],
      nextCursor: null,
    })
    syncContinuationAnalysisMock.mockResolvedValue({
      ready: true,
      message: '分析数据同步完成',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: true,
      },
    })

    const state = useContinuationState(ref('book-1'))
    await state.initializeData()
    await state.syncAnalysisData('manual')

    expect(syncContinuationAnalysisMock).toHaveBeenCalledWith('book-1')
    expect(state.chapterScript.value?.script_text).toBe('原有内容')
    expect(state.pages.value[0]?.story_text).toBe('原有剧情')
    expect(state.pageCount.value).toBe(12)
    expect(state.styleRefPages.value).toBe(4)
    expect(state.continuationDirection.value).toBe('保留当前续写')
    expect(state.isDataReady.value).toBe(true)
    expect(state.lastAnalysisSyncAt.value).toBe('')
  })

  it('shows info messages without letting a previous timer clear a newer message', () => {
    vi.useFakeTimers()

    const state = useContinuationState(ref('book-1'))

    state.showMessage('第一条', 'success')
    vi.advanceTimersByTime(2500)

    state.showMessage('第二条', 'info')
    expect(state.messageType.value).toBe('info')
    expect(state.successMessage.value).toBe('第二条')

    vi.advanceTimersByTime(1000)
    expect(state.successMessage.value).toBe('第二条')

    vi.advanceTimersByTime(2500)
    expect(state.successMessage.value).toBe('')
    expect(state.messageType.value).toBe('')

    vi.useRealTimers()
  })

  it('does not let a pending transient message timer mutate state after owner unmount', () => {
    vi.useFakeTimers()

    try {
      let state: ReturnType<typeof useContinuationState> | undefined
      const Host = defineComponent({
        setup() {
          state = useContinuationState(ref('book-1'))
          state.showMessage('短暂提示', 'success')
          return () => null
        },
      })

      const wrapper = mount(Host)
      expect(state?.successMessage.value).toBe('短暂提示')

      wrapper.unmount()
      const messageAfterUnmount = state?.successMessage.value

      vi.advanceTimersByTime(3000)

      expect(state?.successMessage.value).toBe(messageAfterUnmount)
    } finally {
      vi.useRealTimers()
    }
  })

  it('uses backend-owned v2 asset URLs directly', () => {
    const state = useContinuationState(ref('book/id one#x'))
    state.characters.value = [{
      name: '主角/形态',
      aliases: [],
      description: '',
      forms: [],
      reference_image: '/api/v2/assets/avatar-1',
    }]

    expect(state.getCharacterImageUrl('主角/形态')).toBe('/api/v2/assets/avatar-1')
  })

  it('reports analysis sync failures and releases the sync lock', async () => {
    syncContinuationAnalysisMock.mockClear()
    syncContinuationAnalysisMock.mockRejectedValueOnce(new Error('database busy'))
    const state = useContinuationState(ref('book-1'))

    await expect(state.syncAnalysisData('manual')).resolves.toBeUndefined()

    expect(state.errorMessage.value).toBe('同步分析数据失败：database busy')
    expect(state.messageType.value).toBe('error')
    expect(state.isSyncingAnalysis.value).toBe(false)
  })

  it('does not submit duplicate analysis sync commands', async () => {
    syncContinuationAnalysisMock.mockClear()
    const pending = deferred<{
      ready: boolean
      message: string
      saved_data: { script: null; pages: []; config: null; has_data: boolean }
    }>()
    syncContinuationAnalysisMock.mockReturnValueOnce(pending.promise)
    getCharactersMock.mockResolvedValue({ items: [], nextCursor: null })
    const state = useContinuationState(ref('book-1'))

    const first = state.syncAnalysisData('manual')
    const second = state.syncAnalysisData('manual')

    expect(syncContinuationAnalysisMock).toHaveBeenCalledTimes(1)
    await expect(second).resolves.toBeUndefined()
    pending.resolve({
      ready: true,
      message: 'done',
      saved_data: { script: null, pages: [], config: null, has_data: false },
    })
    await first
  })
})
