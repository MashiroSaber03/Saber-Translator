import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'

const validSessionNameArb = fc.string({ minLength: 1, maxLength: 30 })
  .filter(value => /^[a-z0-9_-]+$/i.test(value))

function createSessionStore() {
  setActivePinia(createPinia())
  return useSessionStore()
}

describe('session context properties', () => {
  it('sets and clears bookshelf context', () => {
    fc.assert(
      fc.property(fc.uuid(), fc.uuid(), (bookId, chapterId) => {
        const sessionStore = createSessionStore()

        expect(sessionStore.isBookshelfMode).toBe(false)

        sessionStore.setContext(bookId, chapterId)
        expect(sessionStore.isBookshelfMode).toBe(true)
        expect(sessionStore.currentBookId).toBe(bookId)
        expect(sessionStore.currentChapterId).toBe(chapterId)

        sessionStore.clearContext()
        expect(sessionStore.isBookshelfMode).toBe(false)
        expect(sessionStore.currentBookId).toBeNull()
        expect(sessionStore.currentChapterId).toBeNull()
      }),
      { numRuns: 20 }
    )
  })

  it('parses bookshelf context from URL parameters', () => {
    fc.assert(
      fc.property(fc.uuid(), fc.uuid(), (bookId, chapterId) => {
        const sessionStore = createSessionStore()
        const searchParams = new URLSearchParams()
        searchParams.set('book', bookId)
        searchParams.set('chapter', chapterId)

        sessionStore.parseContextFromUrl(searchParams)

        expect(sessionStore.isBookshelfMode).toBe(true)
        expect(sessionStore.currentBookId).toBe(bookId)
        expect(sessionStore.currentChapterId).toBe(chapterId)
      }),
      { numRuns: 20 }
    )
  })

  it('sets and clears the current session name', () => {
    fc.assert(
      fc.property(validSessionNameArb, sessionName => {
        const sessionStore = createSessionStore()

        expect(sessionStore.currentSessionName).toBeNull()

        sessionStore.setSessionName(sessionName)
        expect(sessionStore.currentSessionName).toBe(sessionName)

        sessionStore.clearSessionName()
        expect(sessionStore.currentSessionName).toBeNull()
      }),
      { numRuns: 20 }
    )
  })

  it('returns data URLs without network conversion', async () => {
    const sessionStore = createSessionStore()
    const base64Data = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

    await expect(sessionStore.imageUrlToBase64(base64Data)).resolves.toBe(base64Data)
  })

  it('resets context, session list, progress, and errors together', () => {
    fc.assert(
      fc.property(fc.uuid(), fc.uuid(), validSessionNameArb, (bookId, chapterId, sessionName) => {
        const sessionStore = createSessionStore()

        sessionStore.setContext(bookId, chapterId)
        sessionStore.setSessionName(sessionName)
        sessionStore.setSessionList([{ name: 'test', savedAt: new Date().toISOString(), imageCount: 1, version: '2.0' }])
        sessionStore.startBatchSave(10, 'test-session-id')
        sessionStore.setError('测试错误')

        expect(sessionStore.isBookshelfMode).toBe(true)
        expect(sessionStore.currentSessionName).toBe(sessionName)
        expect(sessionStore.sessionList).toHaveLength(1)
        expect(sessionStore.batchSaveState.isInProgress).toBe(true)
        expect(sessionStore.error).toBe('测试错误')

        sessionStore.reset()

        expect(sessionStore.isBookshelfMode).toBe(false)
        expect(sessionStore.currentSessionName).toBeNull()
        expect(sessionStore.sessionList).toHaveLength(0)
        expect(sessionStore.batchSaveState.isInProgress).toBe(false)
        expect(sessionStore.error).toBeNull()
      }),
      { numRuns: 20 }
    )
  })
})
