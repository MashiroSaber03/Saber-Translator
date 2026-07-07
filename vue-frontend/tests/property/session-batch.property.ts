import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'

function createSessionStore() {
  setActivePinia(createPinia())
  return useSessionStore()
}

describe('session batch save properties', () => {
  it('tracks batch progress from start through completion', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 100 }),
        fc.uuid(),
        (totalCount, sessionId) => {
          const sessionStore = createSessionStore()

          expect(sessionStore.batchSaveState.isInProgress).toBe(false)
          expect(sessionStore.batchSaveProgress).toBe(0)

          sessionStore.startBatchSave(totalCount, sessionId)
          expect(sessionStore.batchSaveState.isInProgress).toBe(true)
          expect(sessionStore.batchSaveState.totalCount).toBe(totalCount)
          expect(sessionStore.batchSaveState.sessionId).toBe(sessionId)

          const midProgress = Math.floor(totalCount / 2)
          sessionStore.updateBatchSaveProgress(midProgress)

          expect(sessionStore.batchSaveState.currentIndex).toBe(midProgress)
          expect(sessionStore.batchSaveProgress).toBe(Math.round((midProgress / totalCount) * 100))

          sessionStore.completeBatchSave()
          expect(sessionStore.batchSaveState.isInProgress).toBe(false)
          expect(sessionStore.batchSaveState.sessionId).toBeNull()
          expect(sessionStore.batchSaveProgress).toBe(0)
        }
      ),
      { numRuns: 20 }
    )
  })
})
