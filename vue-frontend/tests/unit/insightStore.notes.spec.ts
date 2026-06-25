import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const getNotesMock = vi.hoisted(() => vi.fn())

vi.mock('@/api/insight', () => ({
  getNotes: getNotesMock,
}))

describe('useInsightStore notes ownership', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    getNotesMock.mockReset()
  })

  it('clears previous book notes immediately when switching books', async () => {
    getNotesMock.mockResolvedValue({ success: true, notes: [] })
    const { useInsightStore } = await import('@/stores/insightStore')
    const store = useInsightStore()

    store.notes.push({
      id: 'note-old',
      type: 'text',
      content: 'old book note',
      createdAt: '2026-06-25T00:00:00.000Z',
      updatedAt: '2026-06-25T00:00:00.000Z',
    })

    store.setCurrentBook('book-2')

    expect(store.notes).toEqual([])
  })
})
