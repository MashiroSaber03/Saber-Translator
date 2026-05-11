import { ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { useContinuationState } from './useContinuationState'

const { prepareContinuationMock, getCharactersMock, getAvailableImagesMock } = vi.hoisted(() => ({
  prepareContinuationMock: vi.fn(),
  getCharactersMock: vi.fn(),
  getAvailableImagesMock: vi.fn(),
}))

vi.mock('@/api/continuation', () => ({
  prepareContinuation: prepareContinuationMock,
  getCharacters: getCharactersMock,
  getAvailableImages: getAvailableImagesMock,
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
    getAvailableImagesMock.mockResolvedValue({ success: true, total_original_pages: 0 })

    const state = useContinuationState(ref('book-1'))
    await state.initializeData()

    expect(state.isDataReady.value).toBe(false)
    expect(state.errorMessage.value).toContain('故事概要')
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
