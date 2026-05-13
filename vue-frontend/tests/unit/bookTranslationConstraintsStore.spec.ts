import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const { updateBookMock } = vi.hoisted(() => ({
  updateBookMock: vi.fn(),
}))

vi.mock('@/api/bookshelf', () => ({
  updateBook: updateBookMock,
}))

import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'

describe('bookTranslationConstraintsStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    updateBookMock.mockReset()
    updateBookMock.mockResolvedValue({
      success: true,
      book: {
        id: 'book-1',
        translation_constraints: {
          glossary: {
            enabled: true,
            autoExtractEnabled: false,
            entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
          },
          non_translate: {
            enabled: true,
            entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
          },
        },
      },
    })
  })

  it('loads and exposes book-level constraints', () => {
    const store = useBookTranslationConstraintsStore()

    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: false,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: false,
        entries: [],
      },
    })

    expect(store.bookId).toBe('book-1')
    expect(store.isAvailable).toBe(true)
    expect(store.glossary.enabled).toBe(true)
    expect(store.nonTranslate.enabled).toBe(false)
  })

  it('resets constraints when leaving current book', () => {
    const store = useBookTranslationConstraintsStore()

    store.loadBookConstraints('book-1', {
      glossary: {
        enabled: true,
        autoExtractEnabled: false,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: true,
        entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
      },
    })
    store.resetBookConstraints()

    expect(store.bookId).toBeNull()
    expect(store.isAvailable).toBe(false)
    expect(store.glossary).toEqual({ enabled: false, autoExtractEnabled: false, entries: [] })
    expect(store.nonTranslate).toEqual({ enabled: false, entries: [] })
  })

  it('saves updated constraints through book update api', async () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: { enabled: false, autoExtractEnabled: false, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })

    const ok = await store.saveBookConstraints({
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: {
        enabled: true,
        entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
      },
    })

    expect(ok).toBe(true)
    expect(updateBookMock).toHaveBeenCalledWith(
      'book-1',
      expect.objectContaining({
        translation_constraints: {
          glossary: {
            enabled: true,
            autoExtractEnabled: true,
            entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
          },
          non_translate: {
            enabled: true,
            entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
          },
        },
      }),
    )
    expect(store.glossary.autoExtractEnabled).toBe(true)
  })

  it('does not mutate runtime constraints when save fails', async () => {
    updateBookMock.mockResolvedValueOnce({
      success: false,
      error: 'save failed',
    })

    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: { enabled: false, autoExtractEnabled: false, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })

    const ok = await store.saveBookConstraints({
      glossary: {
        enabled: true,
        autoExtractEnabled: false,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: { enabled: false, entries: [] },
    })

    expect(ok).toBe(false)
    expect(store.glossary).toEqual({ enabled: false, autoExtractEnabled: false, entries: [] })
  })

  it('preserves autoExtractEnabled when backend response omits the new field', async () => {
    updateBookMock.mockResolvedValueOnce({
      success: true,
      book: {
        id: 'book-1',
        translation_constraints: {
          glossary: {
            enabled: true,
            entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
          },
          non_translate: {
            enabled: false,
            entries: [],
          },
        },
      },
    })

    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', {
      glossary: { enabled: false, autoExtractEnabled: false, entries: [] },
      non_translate: { enabled: false, entries: [] },
    })

    const ok = await store.saveBookConstraints({
      glossary: {
        enabled: true,
        autoExtractEnabled: true,
        entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
      },
      non_translate: { enabled: false, entries: [] },
    })

    expect(ok).toBe(true)
    expect(store.glossary.autoExtractEnabled).toBe(true)
  })
})
