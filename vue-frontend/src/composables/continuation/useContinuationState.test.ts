import { ref } from 'vue'
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

describe('useContinuationState', () => {
  it('keeps continuation blocked when preparation reports missing prerequisites', async () => {
    prepareContinuationMock.mockResolvedValue({
      success: true,
      ready: false,
      message: '续写功能需要故事概要',
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockResolvedValue({ success: true, characters: [] })
    const state = useContinuationState(ref('book-1'))
    await state.initializeData()

    expect(state.isDataReady.value).toBe(false)
    expect(state.errorMessage.value).toContain('故事概要')
  })

  it('shows a persistent error when character loading fails during initialization', async () => {
    prepareContinuationMock.mockResolvedValue({
      success: true,
      ready: true,
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockResolvedValue({
      success: false,
      error: '角色接口不可用',
    })

    const state = useContinuationState(ref('book-1'))
    await state.initializeData()

    expect(state.isDataReady.value).toBe(true)
    expect(state.errorMessage.value).toContain('角色接口不可用')
    expect(state.messageType.value).toBe('error')
  })

  it('resets stale continuation state before applying a fresh empty payload', async () => {
    prepareContinuationMock.mockResolvedValue({
      success: true,
      ready: true,
      saved_data: {
        script: null,
        pages: [],
        config: null,
        has_data: false,
      },
    })
    getCharactersMock.mockResolvedValue({
      success: true,
      characters: [
        {
          name: '主角',
          aliases: [],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      ],
    })
    const state = useContinuationState(ref('book-1'))
    state.pageCount.value = 22
    state.styleRefPages.value = 7
    state.continuationDirection.value = '保留旧方向'
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
        final_prompt: '旧最终提示词',
        image_url: '/tmp/old.png',
        previous_url: '',
        status: 'generated',
      },
    ]

    await state.initializeData()

    expect(state.isDataReady.value).toBe(true)
    expect(state.pageCount.value).toBe(10)
    expect(state.styleRefPages.value).toBe(3)
    expect(state.continuationDirection.value).toBe('')
    expect(state.chapterScript.value).toBeNull()
    expect(state.pages.value).toEqual([])
    expect(state.characters.value.map(character => character.name)).toEqual(['主角'])
  })

  it('syncs analysis data without clearing existing continuation payloads', async () => {
    prepareContinuationMock.mockResolvedValue({
      success: true,
      ready: true,
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
            final_prompt: '原有提示词',
            image_url: '/tmp/old.png',
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
      success: true,
      characters: [
        {
          name: '主角',
          aliases: [],
          description: 'desc',
          forms: [],
          reference_image: '',
          enabled: true,
        },
      ],
    })
    syncContinuationAnalysisMock.mockResolvedValue({
      success: true,
      ready: true,
      message: '分析数据同步完成',
      story_summary_ready: true,
      timeline_ready: true,
      characters_added: 1,
      total_characters: 2,
      synced_at: '2026-05-21T20:00:00',
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
    expect(state.lastAnalysisSyncAt.value).toBe('2026-05-21T20:00:00')
  })

  it('shows info messages without letting an older timer clear a newer message', () => {
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
})
