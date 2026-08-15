import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'

const { updateBookTranslationConstraintsMock } = vi.hoisted(() => ({
  updateBookTranslationConstraintsMock: vi.fn(),
}))

vi.mock('@/api/bookshelf', () => ({
  updateBookTranslationConstraints: updateBookTranslationConstraintsMock,
}))

import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'

function initialConstraints(): BookTranslationConstraints {
  return {
    glossary: {
      enabled: false,
      autoExtractEnabled: false,
      autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
      entries: [],
    },
    nonTranslate: { enabled: false, entries: [] },
  }
}

function savedConstraints(): BookTranslationConstraints {
  return {
    glossary: {
      enabled: true,
      autoExtractEnabled: true,
      autoExtractPrompt: '抽取术语',
      entries: [{ source: 'Alice', target: '爱丽丝', note: '', matchMode: 'text' }],
    },
    nonTranslate: {
      enabled: true,
      entries: [{ pattern: '<keep>', note: '', matchMode: 'text' }],
    },
  }
}

describe('bookTranslationConstraintsStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    updateBookTranslationConstraintsMock.mockReset()
  })

  it('loads the canonical document and its authoritative revision', () => {
    const store = useBookTranslationConstraintsStore()
    const payload = savedConstraints()

    store.loadBookConstraints('book-1', payload, 7)

    expect(store.bookId).toBe('book-1')
    expect(store.revision).toBe(7)
    expect(store.isAvailable).toBe(true)
    expect(store.glossary).toEqual(payload.glossary)
    expect(store.nonTranslate).toEqual(payload.nonTranslate)
  })

  it('resets the full constraint context when leaving the current book', () => {
    const store = useBookTranslationConstraintsStore()
    store.loadBookConstraints('book-1', savedConstraints(), 7)

    store.resetBookConstraints()

    expect(store.bookId).toBeNull()
    expect(store.revision).toBeNull()
    expect(store.isAvailable).toBe(false)
    expect(store.constraints).toEqual(initialConstraints())
  })

  it('saves once through the dedicated CAS resource and adopts its response', async () => {
    const store = useBookTranslationConstraintsStore()
    const next = savedConstraints()
    updateBookTranslationConstraintsMock.mockResolvedValueOnce({
      constraints: next,
      revision: 8,
    })
    store.loadBookConstraints('book-1', initialConstraints(), 7)

    await store.saveBookConstraints(next)

    expect(updateBookTranslationConstraintsMock).toHaveBeenCalledOnce()
    expect(updateBookTranslationConstraintsMock).toHaveBeenCalledWith('book-1', next, 7)
    expect(store.constraints).toEqual(next)
    expect(store.revision).toBe(8)
    expect(store.isSaving).toBe(false)
  })

  it('propagates backend failures without mutating the current document', async () => {
    const store = useBookTranslationConstraintsStore()
    const initial = initialConstraints()
    const failure = new Error('revision conflict')
    updateBookTranslationConstraintsMock.mockRejectedValueOnce(failure)
    store.loadBookConstraints('book-1', initial, 7)

    await expect(store.saveBookConstraints(savedConstraints())).rejects.toBe(failure)

    expect(store.constraints).toEqual(initial)
    expect(store.revision).toBe(7)
    expect(store.isSaving).toBe(false)
  })

  it('does not apply a late response to a newly loaded book context', async () => {
    const store = useBookTranslationConstraintsStore()
    let resolveSave!: (value: { constraints: BookTranslationConstraints; revision: number }) => void
    updateBookTranslationConstraintsMock.mockReturnValueOnce(new Promise(resolve => {
      resolveSave = resolve
    }))
    store.loadBookConstraints('book-1', initialConstraints(), 7)

    const pendingSave = store.saveBookConstraints(savedConstraints())
    const otherBookConstraints = initialConstraints()
    store.loadBookConstraints('book-2', otherBookConstraints, 3)
    resolveSave({ constraints: savedConstraints(), revision: 8 })
    await pendingSave

    expect(store.bookId).toBe('book-2')
    expect(store.revision).toBe(3)
    expect(store.constraints).toEqual(otherBookConstraints)
  })

  it('rejects saves before a constraint document has been loaded', async () => {
    const store = useBookTranslationConstraintsStore()

    await expect(store.saveBookConstraints(savedConstraints()))
      .rejects.toThrow('书籍翻译约束尚未加载')
    expect(updateBookTranslationConstraintsMock).not.toHaveBeenCalled()
  })
})
